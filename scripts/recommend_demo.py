#!/usr/bin/env python3
"""Batch box-recommendation over demo-reccommandation/data.json -> flattened CSV.

For each product in data.json we run the live 2-step pipeline against the SageMaker endpoint:
  Step 1  caption the product photo (.webp)
  Step 2  feed (caption + the data.json marketing description) into the fine-tuned BOM model,
          then enrich the BOM deterministically (carbon/water/recyclability from the catalog).

Two CSVs are written:
  - recommendations.csv          : ONE ROW PER PACKAGING COMPONENT (tube, cap, carton, ...)
  - recommendations_summary.csv  : ONE ROW PER PRODUCT (totals from the enrichment rollup)

The active endpoint is the 4B *dev* model, whose BOM JSON only parses for a fraction of
products. Parse failures are NOT dropped: the product still gets a row in the summary CSV
(parsed=False, with a snippet of the raw output) so you can see what happened.

Usage:
  uv run scripts/recommend_demo.py --limit 3      # test on the first 3 products
  uv run scripts/recommend_demo.py                # all products
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from mecca_pkg_llm.inference import GemmaEndpointClient  # noqa: E402
from mecca_pkg_llm.prompts import load_schema_doc  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
log = logging.getLogger("recommend")

REPO_ROOT = Path(__file__).resolve().parents[1]
LAYERS = ("PP", "SP", "TP")

# Fixed metadata for this demo set (all MECCA lipsticks). pack_volume is unknown for these
# SKUs, so we leave it None -> the prompt says "unspecified" and the model estimates from form.
BASE_META = {
    "category": "Lips",
    "subcategory": "Lipstick",
    "pack_volume": None,
    "pack_volume_unit": "",
    "mfr_region": "Unknown",
    "eol_region": "Australia",
}

COMPONENT_FIELDS = [
    "id", "product_name", "image", "link", "parsed", "caption",
    "tier", "component_name", "rigid_or_soft", "is_reusable",
    "dim_l_mm", "dim_w_mm", "dim_h_mm", "shape", "volume_cm3",
    "materials", "n_materials", "component_mass_g",
    "component_carbon_kg", "component_water_l",
]

SUMMARY_FIELDS = [
    "id", "product_name", "image", "link", "parsed",
    "n_components", "total_mass_g", "total_carbon_kg", "total_water_l",
    "recyclability_pct", "recycled_content_pct",
    "materials_covered", "bom_raw_snippet",
]


def _num(v):
    return v if isinstance(v, (int, float)) else None


def _component_rows(product, caption, bom):
    """Yield one flattened dict per packaging component across PP/SP/TP."""
    base = {
        "id": product["id"],
        "product_name": product.get("name", ""),
        "image": product.get("image", ""),
        "link": product.get("link", ""),
        "parsed": True,
        "caption": caption,
    }
    for tier in LAYERS:
        for comp in bom.get(tier) or []:
            dims = comp.get("dimensions_mm") or {}
            mats = comp.get("materials") or []
            # "Polypropylene:8.5g; Glass:2.0g"
            mat_cells, comp_mass, comp_carbon, comp_water = [], 0.0, 0.0, 0.0
            for m in mats:
                name = m.get("material_name") or "?"
                mass = _num(m.get("mass_g")) or 0.0
                comp_mass += mass
                mat_cells.append(f"{name}:{mass}g")
                env = m.get("_env") or {}
                if _num(env.get("carbon_kg")) is not None:
                    comp_carbon += env["carbon_kg"]
                if _num(env.get("water_l")) is not None:
                    comp_water += env["water_l"]
            row = dict(base)
            row.update({
                "tier": tier,
                "component_name": comp.get("component_name", ""),
                "rigid_or_soft": comp.get("rigid_or_soft"),
                "is_reusable": comp.get("is_reusable"),
                "dim_l_mm": _num(dims.get("l")),
                "dim_w_mm": _num(dims.get("w")),
                "dim_h_mm": _num(dims.get("h")),
                "shape": comp.get("_shape"),
                "volume_cm3": comp.get("_volume_cm3"),
                "materials": "; ".join(mat_cells),
                "n_materials": len(mats),
                "component_mass_g": round(comp_mass, 2),
                "component_carbon_kg": round(comp_carbon, 4) if comp_carbon else 0.0,
                "component_water_l": round(comp_water, 2) if comp_water else 0.0,
            })
            yield row


def _summary_row(product, parsed, rollup, coverage, bom_raw):
    total = (rollup or {}).get("total") or {}
    return {
        "id": product["id"],
        "product_name": product.get("name", ""),
        "image": product.get("image", ""),
        "link": product.get("link", ""),
        "parsed": parsed,
        "n_components": total.get("n_components"),
        "total_mass_g": total.get("mass_g"),
        "total_carbon_kg": total.get("carbon_kg"),
        "total_water_l": total.get("water_l"),
        "recyclability_pct": total.get("recyclability_pct"),
        "recycled_content_pct": total.get("recycled_content_pct"),
        "materials_covered": (
            f"{(coverage or {}).get('with_env', 0)}/{(coverage or {}).get('materials', 0)}"
            if coverage else ""
        ),
        "bom_raw_snippet": "" if parsed else (bom_raw or "")[:300].replace("\n", " "),
    }


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--endpoint-name", default=os.getenv("ENDPOINT_NAME", "gemma3-dev-bom-ep"))
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    ap.add_argument("--profile", default=os.getenv("AWS_PROFILE"))
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "demo-reccommandation" / "data.json")
    ap.add_argument(
        "--catalog", type=Path,
        default=REPO_ROOT / "data" / "processed" / "phase2" / "materials_catalog.json",
    )
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "demo-reccommandation" / "recommendations.csv",
    )
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N products")
    args = ap.parse_args()

    products = json.loads(args.data.read_text())
    if args.limit is not None:
        products = products[: args.limit]
    log.info("processing %d products via endpoint %s", len(products), args.endpoint_name)

    schema_doc = load_schema_doc(args.catalog)
    client = GemmaEndpointClient(
        args.endpoint_name, region=args.region, profile=args.profile, schema_doc=schema_doc
    )

    summary_out = args.out.with_name(args.out.stem + "_summary.csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = n_components = 0
    with args.out.open("w", newline="") as comp_f, summary_out.open("w", newline="") as sum_f:
        comp_w = csv.DictWriter(comp_f, fieldnames=COMPONENT_FIELDS)
        sum_w = csv.DictWriter(sum_f, fieldnames=SUMMARY_FIELDS)
        comp_w.writeheader()
        sum_w.writeheader()

        for product in products:
            pid = product["id"]
            img_path = REPO_ROOT / product["image"]
            if not img_path.exists():
                log.warning("[%s] image missing: %s -- skipping", pid, img_path)
                continue

            log.info("=== [%s] %s ===", pid, product.get("name", ""))
            image_bytes = img_path.read_bytes()

            # Step 1: caption the real photo.
            caption = client.caption(image_bytes)
            # Combine image caption + the data.json marketing description as the BOM visual.
            description = (product.get("description") or "").strip()
            visual = f"{caption}\n\nProduct description: {description}" if description else caption

            metadata = dict(BASE_META, brand=product.get("name") or "Unknown brand")

            # Step 2 (+ deterministic enrichment).
            result = client.recommend_enriched(visual, metadata)
            bom = result.get("bom")
            parsed = bom is not None

            if parsed:
                n_ok += 1
                rows = list(_component_rows(product, caption, bom))
                comp_w.writerows(rows)
                n_components += len(rows)
                log.info("    parsed OK -- %d components", len(rows))
            else:
                n_fail += 1
                log.info("    parse FAILED -- raw: %s", (result.get("bom_raw") or "")[:120])

            sum_w.writerow(_summary_row(
                product, parsed, result.get("rollup"), result.get("coverage"),
                result.get("bom_raw"),
            ))

    log.info(
        "done: %d parsed, %d failed, %d component rows", n_ok, n_fail, n_components
    )
    log.info("wrote %s", args.out)
    log.info("wrote %s", summary_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
