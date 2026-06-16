#!/usr/bin/env python3
"""Step 2a — preprocess the 10K chunk into a clean, normalised product set.

DPK-style stages (the same conceptual pipeline data-prep-kit uses — ingest, dedup,
filter, profile — implemented as light local pandas/Python stages, no Ray/Parquet
overhead for a 10K job):

    1. INGEST   — read data/raw/chunk_10k.jsonl, reshape the synthetic record into the
                  canonical {sku, input, target} shape the formatter expects.
    2. VALIDATE — drop records with no primary packaging, missing name, or unusable BOM.
    3. DEDUP    — MinHash near-dup removal on the product signature (name+brand+description
                  +BOM structure), so the model isn't over-trained on synthetic clones.
    4. PROFILE  — emit a dataset profile (counts, material distribution, mass stats).
    5. CATALOG  — derive the closed-vocabulary materials catalog from the data itself
                  (the synthetic set has no separate catalog CSV). This is the list the
                  formatter embeds in every prompt — the single biggest accuracy lever.

Reads:
    data/raw/chunk_10k.jsonl
Writes:
    data/processed/phase2/products_clean.jsonl     ({sku, input, target} per line)
    data/processed/phase2/materials_catalog.json   (closed vocabulary by type)
    data/processed/phase2/profile.json             (dataset profile)

Usage:
    uv run scripts/02_preprocess.py
    uv run scripts/02_preprocess.py --no-dedup
    uv run scripts/02_preprocess.py --dedup-threshold 0.9
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import orjson
from datasketch import MinHash, MinHashLSH

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
log = logging.getLogger("phase2")

# Fixed category for this run — the 20260609-165708 set is all Lipstick/Lips.
# Kept as fields so the formatter's prompt has category/subcategory like v1.
PRODUCT_GROUP = "Lips"
PRODUCT_SUBGROUP = "Lipstick"

VALID_MATERIAL_TYPES = {
    "Plastic", "Paper/Board", "Metal", "Glass", "Wood", "Textile", "Other",
}


# ====================================================================
#  Stage 1 — INGEST / reshape
# ====================================================================

def reshape(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Synthetic nested record -> canonical {sku, input, target}.

    input  = the fields a user/product-record provides (what the model conditions on)
    target = the PP/SP/TP Bill-of-Materials (what the model must predict)
    """
    prod = rec.get("product") or {}
    sku = rec.get("sku")
    if not sku:
        return None

    inp = {
        "name": prod.get("name"),
        "brand": prod.get("brand"),
        "category": PRODUCT_GROUP,
        "subcategory": PRODUCT_SUBGROUP,
        "pack_volume": prod.get("pack_volume"),
        "pack_volume_unit": prod.get("pack_volume_unit"),
        "mfr_region": prod.get("mfr_region"),
        "eol_region": prod.get("eol_region"),
        "description": prod.get("description"),
    }

    target: dict[str, list[dict[str, Any]]] = {"PP": [], "SP": [], "TP": []}
    for layer in ("PP", "SP", "TP"):
        for comp in rec.get(layer) or []:
            mats = []
            for m in comp.get("materials") or []:
                mats.append({
                    "material_name": m.get("material_name"),
                    "material_abbrev": m.get("material_abbrev"),
                    "material_type": m.get("material_type"),
                    "mass_g": m.get("mass_g"),
                    "recycled_content_percent": m.get("recycled_content_percent"),
                })
            target[layer].append({
                "component_name": comp.get("component_name"),
                "rigid_or_soft": comp.get("rigid_or_soft"),
                "is_reusable": comp.get("is_reusable"),
                "dimensions_mm": comp.get("dimensions_mm"),
                "materials": mats,
            })

    return {"sku": sku, "input": inp, "target": target}


# ====================================================================
#  Stage 2 — VALIDATE
# ====================================================================

def is_valid(p: dict[str, Any]) -> tuple[bool, str]:
    inp, target = p["input"], p["target"]
    if not inp.get("name"):
        return False, "no_name"
    if not target.get("PP"):
        return False, "no_primary_packaging"
    # every material must have a name + a type in the enum, and a numeric mass
    for layer in ("PP", "SP", "TP"):
        for comp in target[layer]:
            if not comp.get("materials"):
                continue  # a component with no materials is tolerated, just skipped downstream
            for m in comp["materials"]:
                if not m.get("material_name"):
                    return False, "material_no_name"
                if m.get("material_type") not in VALID_MATERIAL_TYPES:
                    return False, f"bad_material_type:{m.get('material_type')}"
                if not isinstance(m.get("mass_g"), (int, float)):
                    return False, "material_no_mass"
    return True, "ok"


# ====================================================================
#  Stage 3 — DEDUP (MinHash near-duplicate removal)
# ====================================================================

def product_signature(p: dict[str, Any]) -> set[str]:
    """Shingle set summarising a product, for MinHash similarity."""
    inp = p["input"]
    toks: list[str] = []
    for k in ("name", "brand", "description"):
        v = (inp.get(k) or "").lower()
        toks.extend(v.split())
    # BOM structure: component names + material names (order-independent)
    for layer in ("PP", "SP", "TP"):
        for comp in p["target"][layer]:
            toks.append((comp.get("component_name") or "").lower())
            for m in comp.get("materials") or []:
                toks.append((m.get("material_name") or "").lower())
    return set(t for t in toks if t)


def dedup(products: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Drop near-duplicates with Jaccard similarity >= threshold (MinHash LSH)."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes: dict[str, MinHash] = {}
    kept: list[dict[str, Any]] = []
    dropped = 0

    for p in products:
        sig = product_signature(p)
        mh = MinHash(num_perm=128)
        for tok in sig:
            mh.update(tok.encode("utf-8"))
        if lsh.query(mh):  # a near-duplicate already kept
            dropped += 1
            continue
        lsh.insert(p["sku"], mh)
        minhashes[p["sku"]] = mh
        kept.append(p)

    log.info("dedup: kept %d, dropped %d near-dups (threshold=%.2f)", len(kept), dropped, threshold)
    return kept


# ====================================================================
#  Stage 5 — derive materials CATALOG (closed vocabulary)
# ====================================================================

def build_catalog(products: list[dict[str, Any]]) -> dict[str, Any]:
    """Closed vocabulary of (material_name, material_abbrev, material_type) seen in the data.

    The synthetic set has no canonical catalog CSV, so we derive it. When the same
    material_name appears with inconsistent abbrev/type, we take the most common pairing.
    """
    combos: Counter = Counter()
    for p in products:
        for layer in ("PP", "SP", "TP"):
            for comp in p["target"][layer]:
                for m in comp.get("materials") or []:
                    combos[(m["material_name"], m.get("material_abbrev"), m["material_type"])] += 1

    # Resolve to one (abbrev, type) per material_name = the most frequent combo.
    by_name: dict[str, tuple[tuple[str | None, str], int]] = {}
    for (name, abbrev, mtype), cnt in combos.items():
        prev = by_name.get(name)
        if prev is None or cnt > prev[1]:
            by_name[name] = ((abbrev, mtype), cnt)

    materials = [
        {"material_name": name, "material_abbrev": abbrev, "material_type": mtype}
        for name, ((abbrev, mtype), _cnt) in sorted(by_name.items())
    ]
    type_counts = Counter(m["material_type"] for m in materials)
    return {
        "n_materials": len(materials),
        "type_counts": dict(type_counts),
        "materials": materials,
    }


# ====================================================================
#  Stage 4 — PROFILE
# ====================================================================

def profile(products: list[dict[str, Any]], catalog: dict[str, Any]) -> dict[str, Any]:
    layer_comp_counts = {"PP": [], "SP": [], "TP": []}
    total_masses: list[float] = []
    mat_type_freq: Counter = Counter()
    for p in products:
        tmass = 0.0
        for layer in ("PP", "SP", "TP"):
            layer_comp_counts[layer].append(len(p["target"][layer]))
            for comp in p["target"][layer]:
                for m in comp.get("materials") or []:
                    mat_type_freq[m["material_type"]] += 1
                    if isinstance(m.get("mass_g"), (int, float)):
                        tmass += float(m["mass_g"])
        total_masses.append(tmass)

    def _avg(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    return {
        "n_products": len(products),
        "avg_components_per_layer": {k: _avg(v) for k, v in layer_comp_counts.items()},
        "avg_total_mass_g": _avg(total_masses),
        "material_type_frequency": dict(mat_type_freq),
        "catalog_size": catalog["n_materials"],
        "catalog_type_counts": catalog["type_counts"],
    }


# ====================================================================
#  Main
# ====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=Path("data/raw/chunk_10k.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/phase2"))
    ap.add_argument("--no-dedup", action="store_true", help="Skip MinHash dedup stage")
    ap.add_argument("--dedup-threshold", type=float, default=0.92,
                    help="Jaccard similarity above which products are near-dups (default: %(default)s)")
    args = ap.parse_args()

    if not args.input.exists():
        log.error("input not found: %s — run scripts/01_extract_chunk.py first", args.input)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 — INGEST / reshape
    raw = 0
    reshaped: list[dict[str, Any]] = []
    with args.input.open("rb") as f:
        for line in f:
            raw += 1
            rec = orjson.loads(line)
            r = reshape(rec)
            if r is not None:
                reshaped.append(r)
    log.info("ingest: read %d, reshaped %d", raw, len(reshaped))

    # Stage 2 — VALIDATE
    valid: list[dict[str, Any]] = []
    reasons: Counter = Counter()
    for p in reshaped:
        ok, why = is_valid(p)
        if ok:
            valid.append(p)
        else:
            reasons[why] += 1
    log.info("validate: kept %d, dropped %d", len(valid), len(reshaped) - len(valid))
    if reasons:
        log.info("  drop reasons: %s", dict(reasons))

    # Stage 3 — DEDUP
    clean = valid if args.no_dedup else dedup(valid, args.dedup_threshold)

    # Stage 5 — CATALOG (from the clean set)
    catalog = build_catalog(clean)
    log.info("catalog: %d materials, types=%s", catalog["n_materials"], catalog["type_counts"])

    # Stage 4 — PROFILE
    prof = profile(clean, catalog)

    # Write outputs
    out_clean = args.out_dir / "products_clean.jsonl"
    with out_clean.open("wb") as f:
        for p in clean:
            f.write(orjson.dumps(p))
            f.write(b"\n")
    log.info("wrote %s (%d products)", out_clean, len(clean))

    (args.out_dir / "materials_catalog.json").write_text(json.dumps(catalog, indent=2))
    (args.out_dir / "profile.json").write_text(json.dumps(prof, indent=2))
    log.info("wrote materials_catalog.json + profile.json")
    log.info("profile: %s", json.dumps(prof, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
