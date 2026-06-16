#!/usr/bin/env python3
"""Offline tests - run WITHOUT a live endpoint. Validates the pieces that must be correct
before spending GPU money:

  1. metrics: perfect/wrong/garbage scoring behaves (sanity).
  2. prompt consistency: the inference-time Step-2 prompt EXACTLY matches the prompt the
     training formatter produced for the same product+caption. A mismatch here means the
     model trains on one prompt and is served another -> silent accuracy loss.
  3. inference client wiring: a mocked endpoint drives caption() + recommend() + run().

Usage:
  uv run eval/test_offline.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mecca_pkg_llm import metrics  # noqa: E402
from mecca_pkg_llm.prompts import build_bom_prompt, load_schema_doc  # noqa: E402


def _load_format_mod():
    spec = importlib.util.spec_from_file_location(
        "format03", str(ROOT / "scripts" / "03_format.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metrics():
    gold = {"PP": [{"component_name": "Tube",
                    "materials": [{"material_name": "Polypropylene", "material_type": "Plastic",
                                   "mass_g": 8.0}]}], "SP": [], "TP": []}
    perfect = metrics.aggregate([metrics.score_one(json.dumps(gold), gold)])
    assert perfect["material_abbrev_set_f1"] == 1.0 and perfect["mass_g_total_mape"] == 0.0
    garbage = metrics.aggregate([metrics.score_one("nope", gold)])
    assert garbage["parse_success_rate"] == 0.0
    print("OK test_metrics")


def test_prompt_consistency():
    """The eval rebuilds the Step-2 prompt via prompts.build_bom_prompt; the formatter
    builds it via its own fill_prompt. For the same product+caption they MUST be identical.
    """
    fmt = _load_format_mod()
    catalog = ROOT / "data/processed/phase2/materials_catalog.json"
    if not catalog.exists():
        print("SKIP test_prompt_consistency (run scripts/02_preprocess.py first)")
        return
    cat = json.loads(catalog.read_text())
    schema_doc_fmt = fmt.build_schema_doc(cat["materials"])
    schema_doc_inf = load_schema_doc(catalog)
    assert schema_doc_fmt == schema_doc_inf, "schema docs differ between format and inference"

    inp = {"name": "X", "brand": "Aurelia", "category": "Lips", "subcategory": "Lipstick",
           "pack_volume": 3.5, "pack_volume_unit": "g", "mfr_region": "China",
           "eol_region": "Australia", "description": "A clean sustainable matte lipstick."}
    caption = fmt.caption_from_description(inp["description"], inp["brand"])
    # formatter's prompt (training side) vs inference prompt — must match for the same visual
    prompt_fmt = fmt.fill_prompt(caption, inp, schema_doc_fmt)
    prompt_inf = build_bom_prompt(caption, inp, schema_doc_inf)
    assert prompt_fmt == prompt_inf, (
        "TRAINING and INFERENCE prompts differ!\n--- train ---\n"
        + prompt_fmt[:400] + "\n--- infer ---\n" + prompt_inf[:400]
    )
    print("OK test_prompt_consistency  (train prompt == inference prompt)")


def test_inference_client_mocked():
    import mecca_pkg_llm.inference as inf

    class FakeRuntime:
        def invoke_endpoint(self, **kw):
            body = json.loads(kw["Body"])
            # echo a canned response shaped like TGI Messages API
            has_image = any(
                isinstance(m.get("content"), list)
                and any(c.get("type") == "image_url" for c in m["content"])
                for m in body["messages"]
            )
            text = "A small lipstick tube." if has_image else '{"PP": [], "SP": [], "TP": []}'

            class B:
                def read(self):
                    return json.dumps({"choices": [{"message": {"content": text}}]}).encode()
            return {"Body": B()}

    client = inf.GemmaEndpointClient.__new__(inf.GemmaEndpointClient)
    client.runtime = FakeRuntime()
    client.endpoint_name = "fake"
    client.schema_doc = "SCHEMA"

    cap = client.caption(b"\xff\xd8fakejpeg")
    assert "lipstick" in cap.lower()
    bom = client.recommend(cap, {"brand": "X"})
    assert json.loads(bom).keys() == {"PP", "SP", "TP"}
    full = client.run(b"\xff\xd8fakejpeg", {"brand": "X"})
    assert full["caption"] and full["bom_raw"]
    print("OK test_inference_client_mocked")


if __name__ == "__main__":
    test_metrics()
    test_prompt_consistency()
    test_inference_client_mocked()
    print("\nAll offline tests passed.")
