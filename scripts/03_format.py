#!/usr/bin/env python3
"""Step 2b — format clean products into HuggingFace Messages JSONL for Gemma 3 SFT.

This is the heart of the 2-step design. We fine-tune ONLY step 2 (text -> BOM), but the
text the fine-tuned model sees at inference is a *caption* produced by the vision model in
step 1 — NOT the curated marketing `description`. To avoid a train/inference mismatch, the
prompt is split into two blocks:

    Visual:           <- caption-style text: what a photo would actually show. We derive it
                         from the curated description by rule-stripping brand names and
                         material/jargon words, leaving shape/colour/closure/size cues. This
                         mimics the generic register a zero-shot captioner emits.
    Known metadata:   <- brand, category, pack size, regions as structured fields. These come
                         from the product record at inference, not the photo, so they stay
                         explicit and un-stripped.

The assistant target is the packaging BOM JSON ({PP, SP, TP}), same shape every time.

Splits go by SKU (never leak a product across splits). Train gets N caption *variants* per
product (different surface phrasings of the same Visual block) against an identical target,
so the model learns caption-invariance. Val/test get exactly one caption each for clean eval.

Reads:
    data/processed/phase2/products_clean.jsonl
    data/processed/phase2/materials_catalog.json
Writes:
    data/processed/phase3/{train,val,test}.jsonl
    data/processed/phase3/split_manifest.json
    data/processed/phase3/dataset_stats.json

Usage:
    uv run scripts/03_format.py
    uv run scripts/03_format.py --train-augments 4
    uv run scripts/03_format.py --val-frac 0.1 --test-frac 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
log = logging.getLogger("phase3")

TYPE_ORDER = ("Plastic", "Paper/Board", "Metal", "Glass", "Wood", "Textile", "Other")
CHARS_PER_TOKEN = 4
P95_TOKEN_BUDGET = 3800


# ====================================================================
#  Schema doc — closed-vocabulary catalog embedded in every prompt
# ====================================================================

SCHEMA_DOC_HEADER = """The packaging Bill-of-Materials JSON must have this exact structure:

{
  "PP": [                              // Primary packaging — always present
    {
      "component_name": string,        // e.g. "Lipstick Tube", "Cap", "Carton Box"
      "rigid_or_soft": "Rigid"|"Soft"|null,
      "is_reusable": true|false|null,  // designed to be refilled/reused?
      "dimensions_mm": {               // outer size in millimetres; null if unknown
        "l": number|null, "w": number|null, "h": number|null
      },
      "materials": [
        {
          "material_name": string,     // MUST be from the allowed list below
          "material_abbrev": string|null,
          "material_type": string,     // "Plastic"|"Paper/Board"|"Metal"|"Glass"|"Wood"|"Textile"|"Other"
          "mass_g": number,            // mass of this material in grams
          "recycled_content_percent": number|null  // 0-100, recycled feedstock share
        }
      ]
    }
  ],
  "SP": [...],                         // Secondary packaging — [] if none
  "TP": [...]                          // Tertiary packaging — [] if none
}

Provide PP always. Provide SP and TP whenever plausible for a retail product — for
ordinary retail cosmetics they almost always apply. Only leave a tier empty ([]) when it
genuinely does not apply (e.g. a bulk-only or sample item). Give dimensions_mm whenever you
can reasonably estimate them from the product form and pack size."""

SCHEMA_DOC_FOOTER = """
material_name MUST be EXACTLY one of the values below, paired with its listed
material_abbrev (or null) and material_type. Do NOT invent new material names.

Allowed materials by type:"""


def build_schema_doc(materials: list[dict[str, Any]]) -> str:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in materials:
        by_type[m["material_type"]].append(m)
    parts = [SCHEMA_DOC_HEADER, SCHEMA_DOC_FOOTER]
    for t in TYPE_ORDER:
        items = by_type.get(t, [])
        if not items:
            continue
        parts.append(f"\n{t}:")
        for m in sorted(items, key=lambda x: (x["material_name"] or "").lower()):
            suffix = f" ({m['material_abbrev']})" if m.get("material_abbrev") else ""
            parts.append(f"  - {m['material_name']}{suffix}")
    return "\n".join(parts)


# ====================================================================
#  Caption synthesis — turn a curated description into caption-style text
# ====================================================================
#
# We do NOT call an LLM (deterministic, zero cost, per the chosen approach). The
# transformation strips signals a camera can't see — brand names, specific material
# names, sustainability/marketing claims — keeping the visually-grounded cues. The
# result reads like a generic photo caption, matching what step 1's zero-shot
# captioner will emit at inference.

# Marketing / non-visual words to drop (a photo can't show "sustainable" or "clean").
_MARKETING = re.compile(
    r"\b(sustainable|sustainably|eco[- ]?friendly|clean|natural|luxurious|luxury|premium|"
    r"minimal(ist)?|elegant|sophisticated|recyclable|recycled|responsibly|ethical|"
    r"high[- ]?quality|innovative|signature|iconic|sleek|modern|refillable)\b",
    re.IGNORECASE,
)
# Material words to drop from the Visual block (the model must infer materials, not be told).
_MATERIAL_WORDS = re.compile(
    r"\b(plastic|polypropylene|PP|PE|HDPE|LDPE|PET|PETE|ABS|acrylic|aluminium|aluminum|"
    r"metal|steel|tin|glass|paper|cardboard|board|carton|corrugated|wood(en)?|bamboo|"
    r"textile|fabric|cotton)\b",
    re.IGNORECASE,
)


def _strip_brand(text: str, brand: str | None) -> str:
    if brand:
        text = re.sub(re.escape(brand), "the product", text, flags=re.IGNORECASE)
    return text


def caption_from_description(description: str | None, brand: str | None) -> str:
    """Rule-based caption: strip brand, marketing claims, and explicit material names."""
    if not description:
        return "A lipstick product shown with its packaging."
    text = _strip_brand(description, brand)
    text = _MARKETING.sub("", text)
    text = _MATERIAL_WORDS.sub("packaging", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    if not text:
        text = "A lipstick product shown with its packaging"
    # Normalise to a caption-y opener.
    if not re.match(r"(?i)^(a|an|the|this)\b", text):
        text = "A " + text[0].lower() + text[1:]
    return text + "."


# Surface variants of the Visual block — different phrasings of the SAME caption, so the
# model learns the BOM mapping is invariant to caption wording (train-time augmentation).
def caption_variants(base_caption: str, n: int) -> list[str]:
    openers = [
        "{c}",
        "Photo shows: {c}",
        "Visible in the image: {c}",
        "The picture depicts {lower}",
        "Image description: {c}",
    ]
    out = []
    for tmpl in openers[:n]:
        lower = base_caption[0].lower() + base_caption[1:] if base_caption else base_caption
        out.append(tmpl.format(c=base_caption, lower=lower))
    return out


# ====================================================================
#  Prompt assembly — two-block structure
# ====================================================================

PROMPT_TEMPLATE = """You are a packaging expert. Given a product image description and its known
metadata, predict the complete packaging Bill-of-Materials.

Visual: {visual}

Known metadata:
- Category: {category} > {subcategory}
- Brand: {brand}
- Pack size: {pack_volume} {pack_volume_unit}
- Made in: {mfr_region}
- Sold in: {eol_region}

{schema_doc}

Return ONLY the JSON, no commentary."""


def fill_prompt(visual: str, inp: dict[str, Any], schema_doc: str) -> str:
    pack_vol = inp.get("pack_volume")
    return PROMPT_TEMPLATE.format(
        visual=visual,
        category=inp.get("category") or "Unknown",
        subcategory=inp.get("subcategory") or "Unknown",
        brand=inp.get("brand") or "Unknown brand",
        pack_volume=("unspecified" if pack_vol is None else str(pack_vol)),
        pack_volume_unit=(inp.get("pack_volume_unit") or "" if pack_vol is not None else ""),
        mfr_region=inp.get("mfr_region") or "Unknown",
        eol_region=inp.get("eol_region") or "Unknown",
        schema_doc=schema_doc,
    )


# ====================================================================
#  Target JSON
# ====================================================================

def _round_dims(dims: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalise dimensions_mm to {l,w,h} floats (or None). Used to derive shape + volume."""
    if not isinstance(dims, dict):
        return None
    out = {}
    for k in ("l", "w", "h"):
        v = dims.get(k)
        out[k] = round(float(v), 2) if isinstance(v, (int, float)) else None
    if all(out[k] is None for k in ("l", "w", "h")):
        return None
    return out


def build_target_json(target: dict[str, Any]) -> str:
    """Canonical BOM target. Includes the richer fields the recommendation feature needs:
    per-component dimensions_mm / rigid_or_soft / is_reusable, and per-material
    recycled_content_percent. Sustainability metrics (carbon/water/recyclability) are NOT
    here — those are looked up deterministically from the catalog at enrichment time, never
    predicted by the model.
    """
    out: dict[str, list[dict[str, Any]]] = {"PP": [], "SP": [], "TP": []}
    for layer in ("PP", "SP", "TP"):
        for comp in target.get(layer, []):
            out[layer].append({
                "component_name": comp.get("component_name"),
                "rigid_or_soft": comp.get("rigid_or_soft"),
                "is_reusable": comp.get("is_reusable"),
                "dimensions_mm": _round_dims(comp.get("dimensions_mm")),
                "materials": [
                    {
                        "material_name": m.get("material_name"),
                        "material_abbrev": m.get("material_abbrev"),
                        "material_type": m.get("material_type"),
                        "mass_g": (round(float(m["mass_g"]), 4)
                                   if m.get("mass_g") is not None else None),
                        "recycled_content_percent": (
                            round(float(m["recycled_content_percent"]), 2)
                            if m.get("recycled_content_percent") is not None else None),
                    }
                    for m in comp.get("materials") or []
                ],
            })
    return json.dumps(out, indent=2, ensure_ascii=False)


# ====================================================================
#  Examples
# ====================================================================

def build_examples(product: dict[str, Any], n_aug: int, schema_doc: str) -> list[dict[str, Any]]:
    inp = product["input"]
    target_str = build_target_json(product["target"])
    base_caption = caption_from_description(inp.get("description"), inp.get("brand"))
    visuals = caption_variants(base_caption, n_aug)
    examples = []
    for visual in visuals:
        examples.append({
            "sku": product["sku"],
            "messages": [
                {"role": "user", "content": fill_prompt(visual, inp, schema_doc)},
                {"role": "assistant", "content": target_str},
            ],
        })
    return examples


# ====================================================================
#  SKU split
# ====================================================================

def split_skus(skus, *, val_frac, test_frac, seed):
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")
    rng = random.Random(seed)
    s = sorted(skus)
    rng.shuffle(s)
    n = len(s)
    nv, nt = int(round(n * val_frac)), int(round(n * test_frac))
    val, test, train = set(s[:nv]), set(s[nv:nv + nt]), set(s[nv + nt:])
    assert not (train & val) and not (train & test) and not (val & test)
    return train, val, test


# ====================================================================
#  Validation
# ====================================================================

def validate_split(path: Path):
    errors, schemas, char_lengths = [], set(), []
    with path.open() as f:
        for i, line in enumerate(f):
            try:
                ex = json.loads(line)
            except Exception as e:  # noqa: BLE001
                errors.append(f"line {i}: outer JSON parse failed: {e}")
                continue
            total = 0
            for msg in ex.get("messages", []):
                c = msg.get("content")
                if isinstance(c, str):
                    total += len(c)
                if msg.get("role") == "assistant":
                    try:
                        tgt = json.loads(c)
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"line {i}: assistant not valid JSON: {e}")
                        continue
                    if isinstance(tgt, dict):
                        schemas.add(frozenset(tgt.keys()))
                    else:
                        errors.append(f"line {i}: assistant JSON not an object")
            char_lengths.append(total)
    return errors, schemas, char_lengths


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    log.info("wrote %s (%d examples)", path, len(items))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=Path("data/processed/phase2/products_clean.jsonl"))
    ap.add_argument("--materials-catalog", type=Path,
                    default=Path("data/processed/phase2/materials_catalog.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed/phase3"))
    ap.add_argument("--train-augments", type=int, default=4, help="Caption variants per train product (1-5)")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.input.exists():
        log.error("input not found: %s — run scripts/02_preprocess.py first", args.input)
        return 2
    if not (1 <= args.train_augments <= 5):
        log.error("--train-augments must be 1..5")
        return 2

    catalog = json.loads(args.materials_catalog.read_text())
    schema_doc = build_schema_doc(catalog["materials"])
    log.info("schema doc: %d chars (~%d tokens), %d materials",
             len(schema_doc), len(schema_doc) // CHARS_PER_TOKEN, catalog["n_materials"])

    products = [json.loads(l) for l in args.input.open()]
    log.info("loaded %d clean products", len(products))

    train_skus, val_skus, test_skus = split_skus(
        [p["sku"] for p in products],
        val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )
    log.info("split: train=%d val=%d test=%d", len(train_skus), len(val_skus), len(test_skus))

    train, val, test = [], [], []
    for p in products:
        if p["sku"] in train_skus:
            train.extend(build_examples(p, args.train_augments, schema_doc))
        elif p["sku"] in val_skus:
            val.extend(build_examples(p, 1, schema_doc))
        elif p["sku"] in test_skus:
            test.extend(build_examples(p, 1, schema_doc))
    log.info("examples: train=%d val=%d test=%d", len(train), len(val), len(test))

    out = args.output_dir
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)

    # Validate
    all_passed = True
    split_stats = {}
    for name, path in [("train", out / "train.jsonl"), ("val", out / "val.jsonl"),
                       ("test", out / "test.jsonl")]:
        errors, schemas, lengths = validate_split(path)
        if len(schemas) > 1:
            errors.append(f"{name}: multiple schemas: {schemas}")
        elif schemas and next(iter(schemas)) != frozenset({"PP", "SP", "TP"}):
            errors.append(f"{name}: wrong top-level keys: {set(next(iter(schemas)))}")
        if lengths:
            p95 = sorted(lengths)[int(len(lengths) * 0.95)]
            p95_tok = p95 // CHARS_PER_TOKEN
            if p95_tok > P95_TOKEN_BUDGET:
                errors.append(f"{name}: p95 est tokens {p95_tok} > {P95_TOKEN_BUDGET}")
            p50 = int(statistics.median(lengths))
        else:
            p50 = p95 = p95_tok = 0
        if errors:
            log.error("[%s] FAIL: %s", name, errors[:3])
            all_passed = False
        else:
            log.info("[%s] OK examples=%d chars[p50/p95]=%d/%d est_tok[p95]=%d",
                     name, len(lengths), p50, p95, p95_tok)
        split_stats[name] = {"examples": len(lengths), "char_p50": p50, "char_p95": p95,
                             "est_tokens_p95": p95_tok, "errors": errors}

    (out / "split_manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "input_clean_products": len(products),
        "augmentation": {"train": args.train_augments, "val": 1, "test": 1},
        "skus": {"train": sorted(train_skus), "val": sorted(val_skus), "test": sorted(test_skus)},
    }, indent=2))
    (out / "dataset_stats.json").write_text(json.dumps({
        "validation_passed": all_passed,
        "materials": {"n": catalog["n_materials"], "schema_doc_chars": len(schema_doc)},
        "splits": {k: {"skus": len(s), **split_stats[k]}
                   for k, s in [("train", train_skus), ("val", val_skus), ("test", test_skus)]},
    }, indent=2))
    log.info("wrote split_manifest.json + dataset_stats.json")

    if not all_passed:
        log.error("validation FAILED")
        return 1
    log.info("done. train=%d val=%d test=%d", len(train), len(val), len(test))
    return 0


if __name__ == "__main__":
    sys.exit(main())
