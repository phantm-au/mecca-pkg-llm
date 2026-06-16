#!/usr/bin/env python3
"""Streamlit UI for the packaging-recommendation model — image OR text, 2-step.

Run:
  cd mecca-pkg-llm
  uv run streamlit run ui/app.py

Two input modes:
  - Image: upload a product photo -> Step 1 caption (zero-shot) -> Step 2 BOM (fine-tuned).
  - Text:  type a description/caption directly -> Step 2 BOM only.

Sidebar manages the SageMaker endpoint (status / deploy / DELETE) so you never leave a GPU
endpoint running by accident. A red banner warns while a real-time endpoint is InService.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mecca_pkg_llm.inference import GemmaEndpointClient  # noqa: E402
from mecca_pkg_llm.prompts import load_schema_doc  # noqa: E402

sys.path.insert(0, str(ROOT / "ui"))
from endpoint import EndpointManager, load_env  # noqa: E402

st.set_page_config(page_title="Packaging Recommender (Gemma 3)", page_icon="📦", layout="wide")

env = load_env(ROOT / ".env")
REGION = env.get("AWS_REGION", "us-east-1")
PROFILE = env.get("AWS_PROFILE")
CATALOG = ROOT / "data/processed/phase2/materials_catalog.json"


@st.cache_resource
def get_manager():
    return EndpointManager(region=REGION, profile=PROFILE, role_arn=env.get("SAGEMAKER_ROLE_ARN"))


@st.cache_data
def get_schema_doc():
    return load_schema_doc(CATALOG) if CATALOG.exists() else ""


def get_client(endpoint_name: str) -> GemmaEndpointClient:
    return GemmaEndpointClient(endpoint_name, region=REGION, profile=PROFILE,
                              schema_doc=get_schema_doc())


# ---------------- Sidebar: endpoint control ----------------
st.sidebar.header("⚙️ Endpoint")
endpoint_name = st.sidebar.text_input("Endpoint name",
                                      value=env.get("ENDPOINT_NAME", "gemma3-dev-bom-ep"))
mgr = get_manager()

if st.sidebar.button("Check status"):
    st.session_state["ep_status"] = mgr.status(endpoint_name)

status = st.session_state.get("ep_status")
if status:
    if status["exists"] and status["status"] == "InService":
        st.sidebar.success(f"InService — created {status.get('created','')[:19]}")
    elif status["exists"]:
        st.sidebar.info(f"Status: {status['status']}")
    else:
        st.sidebar.warning("Endpoint not found — deploy it (CLI) or below.")

with st.sidebar.expander("Deploy (spins up GPU $$)", expanded=False):
    model_s3 = st.text_input("Merged model S3 prefix", value=env.get("MODEL_S3", ""))
    instance = st.text_input("Instance type", value=env.get("ENDPOINT_INSTANCE", "ml.g5.2xlarge"))
    num_gpus = st.number_input("GPUs", min_value=1, max_value=8,
                               value=int(env.get("ENDPOINT_NUM_GPUS", "1")))
    serverless = False  # disabled: see caption below
    st.caption("Real-time on ml.g5.2xlarge (~$1.50/hr) is the validated path. "
               "Deploy takes ~8–12 min; the button blocks until InService. "
               "(Serverless is unavailable — the merged model is stored as loose files "
               "/ ModelDataSource, which serverless endpoints don't support.)")
    if st.button("Deploy endpoint"):
        if not model_s3:
            st.error("Provide the merged model S3 prefix (see training/last_training_job.txt).")
        else:
            with st.spinner("Deploying… (~8–12 min, container pull + model load)"):
                mgr.deploy(model_s3, endpoint_name, instance_type=instance,
                           num_gpus=int(num_gpus), serverless=serverless,
                           hf_token=env.get("HF_TOKEN"))
            st.session_state["ep_status"] = {"exists": True, "status": "InService"}
            st.success("Deployed — InService.")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️  DELETE endpoint (stop billing)", type="primary"):
    mgr.delete(endpoint_name)
    st.session_state["ep_status"] = {"exists": False, "status": "NotFound"}
    st.sidebar.success("Deleted. Billing stopped.")

# Cost reminder banner.
if status and status.get("status") == "InService":
    st.warning("⚠️ A real-time endpoint is **InService** and billing per hour. "
               "Delete it from the sidebar when you're done.")

st.title("Packaging / Box Recommendation")
st.caption("Gemma 3 · image and/or text → rich description → packaging BOM (fine-tuned) → "
           "deterministic sustainability metrics from the materials catalog.")

if not CATALOG.exists():
    st.error("materials_catalog.json missing — run scripts/02_preprocess.py first.")
    st.stop()

st.markdown("Upload a product image, write a description, or both.")
uploaded = st.file_uploader("Product image (optional)", type=["jpg", "jpeg", "png", "webp"])
image_bytes = uploaded.read() if uploaded else None
if image_bytes:
    st.image(image_bytes, width=260)
text_note = st.text_area("Description (optional)", value="", height=80,
                         placeholder="e.g. A creamy matte lipstick in a slim tube with a magnetic cap.")
text_note = text_note.strip() or None

metadata = {
    "category": "Lips", "subcategory": "Lipstick", "brand": "Unknown brand",
    "pack_volume": None, "pack_volume_unit": "",
    "mfr_region": "Unknown", "eol_region": "Unknown",
}


def _fmt(v, unit="", nd=1):
    return "—" if v is None else f"{v:,.{nd}f}{unit}"


def _render_results(res: dict, description: str | None):
    if description:
        st.markdown("### 📝 Product description")
        st.write(description)

    bom = res.get("bom")
    if bom is None:
        raw = res.get("bom_raw", "")
        stripped = raw.strip()
        if stripped.startswith("{") and not stripped.endswith("}"):
            st.error("Output was cut off before the JSON finished (token limit). Raise "
                     "max_new_tokens in src/mecca_pkg_llm/inference.py.")
        else:
            st.error("Model output was not valid JSON.")
        st.code(raw)
        return

    roll = res["rollup"]
    tot = roll["total"]

    st.markdown("### ♻️ Sustainability (total)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total mass", _fmt(tot["mass_g"], " g"))
    c2.metric("Carbon", _fmt(tot["carbon_kg"], " kg CO₂e", nd=3))
    c3.metric("Water", _fmt(tot["water_l"], " L"))
    c4.metric("Recyclability", _fmt(tot["recyclability_pct"], "%"))
    c5.metric("Recycled content", _fmt(tot["recycled_content_pct"], "%"))
    st.caption("Carbon / water / recyclability are looked up from the materials catalog "
               "(real per-material intensities × predicted mass) — not generated by the model.")

    st.markdown("### 📦 Packaging bill of materials")
    for layer, label in [("PP", "Primary"), ("SP", "Secondary"), ("TP", "Tertiary")]:
        comps = bom.get(layer) or []
        if not comps:
            continue
        tr = roll[layer]
        st.markdown(f"**{label} ({layer})** — {tr['n_components']} components · "
                    f"{_fmt(tr['mass_g'], ' g')} · {_fmt(tr['carbon_kg'], ' kg CO₂e', nd=3)} · "
                    f"recyclability {_fmt(tr['recyclability_pct'], '%')}")
        rows = []
        for c in comps:
            dims = c.get("dimensions_mm") or {}
            dim_str = ("{}×{}×{}".format(dims.get("l"), dims.get("w"), dims.get("h"))
                       if dims else "—")
            shape = c.get("_shape", "—")
            vol = c.get("_volume_cm3")
            for m in c.get("materials") or []:
                env = m.get("_env") or {}
                rows.append({
                    "Component": c.get("component_name"),
                    "Shape": shape,
                    "Dims (mm)": dim_str,
                    "Vol (cm³)": vol,
                    "Material": m.get("material_name"),
                    "Type": m.get("material_type"),
                    "Mass (g)": m.get("mass_g"),
                    "Recycled %": m.get("recycled_content_percent"),
                    "Carbon (kg)": env.get("carbon_kg"),
                    "Water (L)": env.get("water_l"),
                })
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

    cov = res.get("coverage") or {}
    if cov and cov.get("with_env", 0) < cov.get("materials", 0):
        st.warning(f"Env data found for {cov['with_env']}/{cov['materials']} materials — "
                   "totals are partial.")
    with st.expander("Raw BOM JSON"):
        st.code(json.dumps(bom, indent=2))


if st.button("Recommend packaging", type="primary"):
    if not endpoint_name:
        st.error("Set an endpoint name in the sidebar.")
        st.stop()
    if image_bytes is None and not text_note:
        st.error("Provide an image, a description, or both.")
        st.stop()
    client = get_client(endpoint_name)
    try:
        description = None
        if image_bytes is not None:
            with st.spinner("Step 1: describing image…"):
                description = client.describe_image(image_bytes)
                visual = client.caption(image_bytes)
            if text_note:
                visual = f"{visual} {text_note}".strip()
        else:
            with st.spinner("Step 1: normalising description…"):
                description = client.describe_text(text_note)
            visual = text_note

        with st.spinner("Step 2: predicting packaging BOM + sustainability…"):
            res = client.recommend_enriched(visual, metadata)
    except Exception as e:  # noqa: BLE001
        st.error(f"Inference failed: {e}")
        st.stop()

    _render_results(res, description)
