# The Streamlit Test App

```bash
cd mecca-pkg-llm
uv run streamlit run ui/app.py
```

## `ui/app.py` - the UI

### Setup & caching

Loads `.env` for `AWS_REGION`, `AWS_PROFILE`, `SAGEMAKER_ROLE_ARN`, `ENDPOINT_NAME`,
`MODEL_S3`. Cached resources:

- [`get_manager()`](../../ui/app.py#L40) - `@st.cache_resource` `EndpointManager` (persistent
  boto3 session).
- [`get_schema_doc()`](../../ui/app.py#L45) - `@st.cache_data` parsed materials catalog (the
  closed-vocabulary schema doc embedded in the Step-2 prompt).
- [`get_client()`](../../ui/app.py#L50) - a `GemmaEndpointClient` bound to the chosen endpoint.

### Inputs ([L114-131](../../ui/app.py#L114))

- `st.file_uploader` - optional image (`jpg/jpeg/png/webp`).
- `st.text_area` - optional free-text note (stripped to `None` if blank).
- **No metadata UI**: `metadata` is hardcoded to the training-time fallbacks
  (`category="Lips"`, `subcategory="Lipstick"`, `brand="Unknown brand"`, `pack_volume=None`,
  regions `"Unknown"`) - the model was trained with these defaults, so unknowns are in-domain.

### Inference flow (on "Recommend packaging")

1. Validate: endpoint name set, and at least one of image/text provided.
2. **Step 1 - build the `Visual:` block:**
   - Image → [`client.describe_image()`](../../src/mecca_pkg_llm/inference.py#L99) (rich paragraph
     for display) **and** [`client.caption()`](../../src/mecca_pkg_llm/inference.py#L85) (terse
     caption that drives Step 2). If text is also given, it's folded into the caption.
   - Text only → [`client.describe_text()`](../../src/mecca_pkg_llm/inference.py#L112) normalises
     the note; the note itself is used as the caption.
3. **Step 2 + enrich:**
   [`client.recommend_enriched(visual, metadata)`](../../src/mecca_pkg_llm/inference.py#L128) →
   `{bom_raw, bom, rollup, coverage}`. This parses the BOM and runs
   [`enrich_bom()`](../../src/mecca_pkg_llm/enrich.py#L75) (catalog join for carbon/water/
   recyclability, shape/volume from dimensions).
4. Render results.

### Results rendering

- **Sustainability (total)** - a 5-metric row: total mass (g), carbon (kg CO₂e), water (L),
  recyclability (%), recycled content (%). All from the enrichment **rollup**, i.e. catalog
  lookups scaled by predicted mass - not model output.
- **Packaging BOM** - three expandable tiers (PP/SP/TP); per tier a dataframe of components
  (name, shape, dims, volume, material, type, mass, recycled %, carbon, water) plus per-tier
  totals.
- **Raw BOM JSON** - expandable.
- **Error path** - if the model output doesn't parse (e.g. truncated JSON), the app shows an
  error + the raw output instead of crashing; partial catalog coverage is surfaced as a
  warning with coverage stats.

### Sidebar - endpoint control & budget protection

- **Check status** ([L61](../../ui/app.py#L61)) → `mgr.status()`; InService / other / not-found.
- **Deploy** expander ([L73-92](../../ui/app.py#L73)): model S3 prefix (`MODEL_S3`), instance
  type (default `ml.g5.2xlarge` ~$1.50/hr - the validated path), GPU count. Deploy blocks until
  InService (~8–12 min). **Serverless is disabled** (the merged model is loose files; see
  below).
- **🗑️ DELETE endpoint (stop billing)** ([L95-98](../../ui/app.py#L95)) - the primary budget
  control.
- **Cost banner** ([L101-103](../../ui/app.py#L101)) - a persistent ⚠️ warning whenever an
  endpoint is InService.
- Session state in `st.session_state["ep_status"]` avoids re-polling on every rerun.

## `ui/endpoint.py` - endpoint lifecycle ([EndpointManager](../../ui/endpoint.py#L48))

Serves the merged model via the HuggingFace **TGI** container exposing the OpenAI-style
Messages API that [inference.py](../../src/mecca_pkg_llm/inference.py) calls (and which supports
Gemma 3 `image_url` parts for Step-1 captioning).

### CLI

```bash
uv run ui/endpoint.py deploy  --model-s3 s3://.../model/ --name gemma3-bom-ep
uv run ui/endpoint.py status  --name gemma3-bom-ep
uv run ui/endpoint.py delete  --name gemma3-bom-ep     # do this when done!
uv run ui/endpoint.py list                              # list running endpoints
```

### `deploy()` ([L60-122](../../ui/endpoint.py#L60))

- **Idempotency cleanup** ([L69-77](../../ui/endpoint.py#L69)): deletes any stale same-named
  endpoint-config/model from a prior failed deploy (does **not** touch a live InService
  endpoint - delete that explicitly).
- **Container env** ([L81-89](../../ui/endpoint.py#L81)): `HF_MODEL_ID=/opt/ml/model`,
  `SM_NUM_GPUS`, `MAX_INPUT_TOKENS=6000`, `MAX_TOTAL_TOKENS=8000`,
  `MESSAGES_API_ENABLED=true`.
- **Model data** as `S3Prefix` / `CompressionType: None` (uncompressed loose files);
  `image_uri` = TGI **3.2.3** resolved with explicit `region`.
- **Serverless is rejected** ([L104-113](../../ui/endpoint.py#L104)): serverless endpoints
  don't support `ModelDataSource` (loose files), so `deploy(serverless=True)` raises a clear
  `ValueError` rather than a cryptic AWS error. (Real-time only, unless you repackage as
  `model.tar.gz`.)
- Real-time deploy waits for InService with a 900 s container health-check timeout.

> Note: the module docstring mentions a "scale-to-zero serverless option," but the implemented
> behavior **fails fast on serverless** for this loose-files artifact. Real-time is the
> validated path.

### `status()` / `delete()` / `list_endpoints()`

`status()` distinguishes real errors from "not found" (ValidationException). `delete()` removes
both the endpoint and its config and logs that billing stops. `list_endpoints()` enumerates
running endpoints.

### Default instance

`DEFAULT_INSTANCE = "ml.g6.12xlarge"` ([L34](../../ui/endpoint.py#L34)) for the CLI (sized for a
12B model needing sharding); the **UI** defaults to the cheaper `ml.g5.2xlarge` for the 4B dev
model. A forgotten real-time `ml.g6.12xlarge` ≈ **$120+/day** - the entire reason for the
sidebar's delete button and banner.

---

*Previous: [Evaluation](04-evaluation.md) · Back to the [docs hub](../README.md) ·
Plain-language version: [non-technical/05-streamlit-app.md](../non-technical/05-streamlit-app.md).
Operational teardown checklist: [RUNBOOK.md](../../RUNBOOK.md) §6.*
