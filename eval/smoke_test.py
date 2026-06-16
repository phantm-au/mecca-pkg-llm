#!/usr/bin/env python3
"""Step 4 — end-to-end smoke test: real image -> caption -> BOM (the full 2-step flow).

Unlike evaluate.py (which feeds text captions to test the Step-2 model in isolation), this
exercises BOTH steps against the live endpoint with REAL product photos, so you can eyeball
whether Step-1 captioning is good enough to drive Step-2.

Since the synthetic dataset has no product images, you supply your own: point --images at a
folder of lipstick photos (jpg/png/webp). Optionally pair each image with metadata via a
sidecar JSON (same basename, .json) holding {brand, pack_volume, pack_volume_unit,
mfr_region, eol_region}; otherwise sensible defaults are used.

Usage:
  uv run eval/smoke_test.py --endpoint-name gemma3-bom-ep --images ./sample_photos
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mecca_pkg_llm import metrics  # noqa: E402
from mecca_pkg_llm.inference import GemmaEndpointClient  # noqa: E402
from mecca_pkg_llm.prompts import load_schema_doc  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
log = logging.getLogger("smoke")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_META = {
    "category": "Lips", "subcategory": "Lipstick", "brand": "Unknown brand",
    "pack_volume": 3.5, "pack_volume_unit": "g", "mfr_region": "Unknown", "eol_region": "Australia",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint-name", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--images", type=Path, required=True, help="Folder of product photos")
    ap.add_argument("--catalog", type=Path, default=Path("data/processed/phase2/materials_catalog.json"))
    ap.add_argument("--out", type=Path, default=Path("eval/results/smoke.jsonl"))
    args = ap.parse_args()

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        log.error("no images found in %s (looked for %s)", args.images, IMG_EXTS)
        return 2
    log.info("found %d images", len(images))

    schema_doc = load_schema_doc(args.catalog)
    client = GemmaEndpointClient(args.endpoint_name, region=args.region,
                                 profile=args.profile, schema_doc=schema_doc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.out.open("w")
    for img_path in images:
        meta = dict(DEFAULT_META)
        sidecar = img_path.with_suffix(".json")
        if sidecar.exists():
            meta.update(json.loads(sidecar.read_text()))

        image_bytes = img_path.read_bytes()
        log.info("=== %s ===", img_path.name)
        result = client.run(image_bytes, meta)
        bom = metrics.try_parse(result["bom_raw"])
        parsed_ok = bom is not None
        schema_ok = parsed_ok and frozenset(bom.keys()) == frozenset({"PP", "SP", "TP"})

        print(f"\n--- {img_path.name} ---")
        print(f"caption : {result['caption']}")
        print(f"parsed  : {parsed_ok}   schema_ok: {schema_ok}")
        if parsed_ok:
            n_comp = sum(len(bom.get(l) or []) for l in ("PP", "SP", "TP"))
            print(f"BOM     : {n_comp} components across PP/SP/TP")
            print(json.dumps(bom, indent=2)[:600])
        else:
            print(f"RAW     : {result['bom_raw'][:300]}")

        out_f.write(json.dumps({
            "image": img_path.name, "metadata": meta,
            "caption": result["caption"], "bom_raw": result["bom_raw"],
            "parsed": parsed_ok, "schema_ok": schema_ok,
        }, ensure_ascii=False) + "\n")
    out_f.close()
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
