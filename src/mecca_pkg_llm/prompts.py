"""Prompt contracts — shared by training-data formatting AND inference.

CRITICAL: the Step-2 prompt built here MUST match what scripts/03_format.py produced for
training, or the model runs off-distribution. The format logic is duplicated intentionally
(scripts/ run standalone with no src import on the training box); keep the two in sync.

Step-1 captioning prompt is co-designed with the training-data caption style: it instructs
the model to emit a short, generic, visually-grounded description WITHOUT naming materials
or brand — the same register the rule-based caption synthesiser used at format time.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

TYPE_ORDER = ("Plastic", "Paper/Board", "Metal", "Glass", "Wood", "Textile", "Other")

CAPTION_PROMPT = (
    "Describe ONLY what is visibly shown in this product photo in 1-2 sentences. "
    "Mention container shape, colour, closure type, and approximate size. "
    "Do NOT name the brand, do NOT name specific materials, and do NOT guess exact "
    "dimensions. Write a plain, factual caption."
)

DESCRIBE_PROMPT = (
    "You are a packaging expert. Describe the product shown in this image in clear, "
    "self-explanatory prose, focusing on details that matter for predicting its packaging "
    "bill of materials. Cover, where visible or inferable:\n"
    "  - product type / category (e.g. lipstick, serum, sunscreen)\n"
    "  - primary container form (tube, bottle, jar, pump, pouch, compact, stick, sachet)\n"
    "  - closure / cap / applicator (screw cap, pump, dropper, flip-top, brush)\n"
    "  - apparent materials and finishes (glass, plastic, metal, paperboard; matte/gloss, foil)\n"
    "  - any secondary packaging (unit carton, box, sleeve) and labels/printing\n"
    "  - approximate size or fill volume if it can be estimated\n"
    "Write a single descriptive paragraph. Do NOT output JSON or lists, and do NOT guess a "
    "brand or SKU you cannot actually see."
)

DESCRIBE_TEXT_PROMPT = (
    "You are a packaging expert. Rewrite the following product note into one clear, "
    "self-explanatory paragraph describing the product and its packaging (container form, "
    "closure, apparent materials, any carton/sleeve, approximate size). Do NOT output JSON "
    "or lists. Do NOT invent a brand or facts not implied by the note.\n\nNote: {note}"
)

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


def load_schema_doc(catalog_path: str | Path) -> str:
    catalog = json.loads(Path(catalog_path).read_text())
    return build_schema_doc(catalog["materials"])


def build_bom_prompt(visual: str, metadata: dict[str, Any], schema_doc: str) -> str:
    """Build the Step-2 prompt. `visual` is the caption (from Step 1 or a test set),
    `metadata` carries the non-visual known fields.
    """
    pack_vol = metadata.get("pack_volume")
    return PROMPT_TEMPLATE.format(
        visual=visual,
        category=metadata.get("category") or "Unknown",
        subcategory=metadata.get("subcategory") or "Unknown",
        brand=metadata.get("brand") or "Unknown brand",
        pack_volume=("unspecified" if pack_vol is None else str(pack_vol)),
        pack_volume_unit=(metadata.get("pack_volume_unit") or "" if pack_vol is not None else ""),
        mfr_region=metadata.get("mfr_region") or "Unknown",
        eol_region=metadata.get("eol_region") or "Unknown",
        schema_doc=schema_doc,
    )
