#!/usr/bin/env python3
"""Step 4 — evaluate the fine-tuned model on the held-out test set (Step-2 only).

Runs TWO conditions against the SAME test products, both through the model's text path:

  (i)  caption     — the caption-style Visual block (what the model was trained on AND what
                     the vision model emits at inference). This is the PRODUCTION-REALISTIC
                     number; the ship decision is based on it.
  (ii) description — the raw curated marketing description as the Visual block (an upper
                     bound, comparable to v1). The gap (i) vs (ii) is the distribution tax.

Both reuse data/processed/phase3/test.jsonl (gold targets) and the original product fields
for the metadata block. We rebuild the prompt locally so we can swap the Visual block.

Inference goes through a live SageMaker endpoint (the model is text-only here, so the
endpoint just needs the merged Gemma model served via TGI). Provide --endpoint-name.

Outputs eval/results/<timestamp>/metrics.json (+ per-example predictions for inspection).

Usage:
  uv run eval/evaluate.py --endpoint-name gemma3-bom-ep --limit 100
  uv run eval/evaluate.py --endpoint-name gemma3-bom-ep --condition caption
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mecca_pkg_llm import metrics  # noqa: E402
from mecca_pkg_llm.inference import GemmaEndpointClient  # noqa: E402
from mecca_pkg_llm.prompts import load_schema_doc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import importlib.util  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
log = logging.getLogger("eval")


def _load_caption_fn():
    """Import caption_from_description from scripts/03_format.py without renaming files."""
    spec = importlib.util.spec_from_file_location(
        "format03", str(Path(__file__).resolve().parents[1] / "scripts" / "03_format.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.caption_from_description


def load_test_products(clean_path: Path, test_skus: set[str]) -> dict[str, dict]:
    """Load the clean products for the test SKUs (we need raw fields + gold target)."""
    out = {}
    with clean_path.open() as f:
        for line in f:
            p = json.loads(line)
            if p["sku"] in test_skus:
                out[p["sku"]] = p
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint-name", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--phase3-dir", type=Path, default=Path("data/processed/phase3"))
    ap.add_argument("--clean", type=Path, default=Path("data/processed/phase2/products_clean.jsonl"))
    ap.add_argument("--catalog", type=Path, default=Path("data/processed/phase2/materials_catalog.json"))
    ap.add_argument("--condition", choices=["caption", "description", "both"], default="both")
    ap.add_argument("--limit", type=int, default=100, help="Max test products to score")
    ap.add_argument("--out-dir", type=Path, default=Path("eval/results"))
    args = ap.parse_args()

    manifest = json.loads((args.phase3_dir / "split_manifest.json").read_text())
    test_skus = set(manifest["skus"]["test"])
    products = load_test_products(args.clean, test_skus)
    skus = sorted(products)[: args.limit]
    log.info("scoring %d/%d test products", len(skus), len(test_skus))

    schema_doc = load_schema_doc(args.catalog)
    caption_from_description = _load_caption_fn()
    client = GemmaEndpointClient(args.endpoint_name, region=args.region,
                                 profile=args.profile, schema_doc=schema_doc)

    conditions = ["caption", "description"] if args.condition == "both" else [args.condition]
    results: dict[str, list] = {c: [] for c in conditions}
    predictions: dict[str, list] = {c: [] for c in conditions}

    for i, sku in enumerate(skus, 1):
        p = products[sku]
        inp = p["input"]
        gold = {layer: p["target"][layer] for layer in ("PP", "SP", "TP")}
        for cond in conditions:
            if cond == "caption":
                visual = caption_from_description(inp.get("description"), inp.get("brand"))
            else:
                visual = inp.get("description") or "(no description)"
            try:
                pred_text = client.recommend(visual, inp)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] invoke failed for %s: %s", cond, sku, e)
                pred_text = ""
            results[cond].append(metrics.score_one(pred_text, gold))
            predictions[cond].append({"sku": sku, "visual": visual, "pred": pred_text})
        if i % 20 == 0:
            log.info("  %d/%d done", i, len(skus))

    out = args.out_dir / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    summary = {cond: metrics.aggregate(results[cond]) for cond in conditions}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    for cond in conditions:
        (out / f"predictions_{cond}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in predictions[cond])
        )

    print("\n" + "=" * 64)
    print(" Evaluation results  (Step-2 text -> BOM)")
    print("=" * 64)
    for cond in conditions:
        m = summary[cond]
        print(f"\n[{cond}]  n={m['n']}")
        print(f"  parse_success_rate     : {m['parse_success_rate']:.3f}")
        print(f"  schema_compliance_rate : {m['schema_compliance_rate']:.3f}")
        print(f"  material_type_set_f1   : {m['material_type_set_f1']:.3f}")
        print(f"  material_abbrev_set_f1 : {m['material_abbrev_set_f1']:.3f}")
        print(f"  mass_g_total_mape      : {m['mass_g_total_mape']:.3f}")
        print(f"  component_count_mae    : {m['component_count_mae']:.3f}")
    if len(conditions) == 2:
        gap = (summary["description"]["material_abbrev_set_f1"]
               - summary["caption"]["material_abbrev_set_f1"])
        print(f"\n  >> distribution tax (abbrev_f1 description - caption): {gap:+.3f}")
        print("     (large positive gap = model over-relies on rich descriptions; "
              "the 'caption' number is what production will see)")
    print("=" * 64)
    print(f"\nwrote {out}/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
