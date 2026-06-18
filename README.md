# mecca-pkg-llm - Packaging-recommendation model (Gemma 3, SageMaker)

Fine-tune **Google Gemma 3** (VL for **packaging / box recommendation**, trained and
served **in SageMaker**.

## Documentation

Full reference docs live in [documentation/](documentation/) - start at the
[documentation hub](documentation/README.md). Each topic comes in a **plain-language** and a
**technical** version:

| # | Topic | Plain-language | Technical |
|---|---|---|---|
| 1 | Dataset | [non-technical](documentation/non-technical/01-dataset.md) | [technical](documentation/technical/01-dataset.md) |
| 2 | Preprocessing | [non-technical](documentation/non-technical/02-preprocessing.md) | [technical](documentation/technical/02-preprocessing.md) |
| 3 | SFT (fine-tuning) | [non-technical](documentation/non-technical/03-sft.md) | [technical](documentation/technical/03-sft.md) |
| 4 | Evaluation | [non-technical](documentation/non-technical/04-evaluation.md) | [technical](documentation/technical/04-evaluation.md) |
| 5 | Streamlit test app | [non-technical](documentation/non-technical/05-streamlit-app.md) | [technical](documentation/technical/05-streamlit-app.md) |

For the operational, end-to-end run sequence (and the ⚠️ endpoint-teardown checklist), see
[RUNBOOK.md](RUNBOOK.md).

## The 2-step inference flow (key idea)

The dataset has **no images mapped to products**, so we do **NOT** do multimodal fine-tuning.
Instead inference is two passes of the *same* Gemma model:

1. **Step 1 - caption (zero-shot, not fine-tuned):** product image → short visual description, using
   Gemma's *inherent* vision ability.
2. **Step 2 - recommend (fine-tuned, text-only):** visual description + known product metadata +
   materials catalog → packaging Bill-of-Materials (BOM) JSON. **Only this step is fine-tuned.**

To keep Step-1 captioning intact, training **freezes the vision encoder AND the multimodal
projector** - LoRA touches only the language-model layers.

### The caption/training-text match (the load-bearing detail)

At inference Step-2 sees a *caption*, but the dataset has curated marketing `description` text.
To avoid a train/inference mismatch, the Step-2 prompt is split into two blocks:

- **`Visual:`** - caption-style text (the description, rule-stripped of brand/material jargon) - what
  the vision model will actually emit.
- **`Known metadata:`** - brand, category, pack size, regions as structured fields - these come from
  the product record at inference, *not* the photo.

## Pipeline (mirrors mecca-llm phase 2→5)

> **Step 0 - get the source dataset (required before step 1).** The raw dataset is **not in git**
> (it's ~440 MB). Download it from Google Drive and drop the files into the `dataset/` folder:
>
> Expected files in `dataset/`: `products.jsonl` (49,881 products), `material_view.csv`,
> `batch_job.json`, `manifest.json`. See [dataset/.gitkeep](dataset/.gitkeep) for details.
> `scripts/01_extract_chunk.py` reads `dataset/products.jsonl`, so step 1 will fail without it.
> *(You only need this for the data-prep/training steps - not for deploying an already-trained model.)*

| Step | Script | Output |
|---|---|---|
| 1. Extract 10K chunk | `scripts/01_extract_chunk.py` | `data/raw/chunk_10k.jsonl` |
| 2. Preprocess (DPK-style) | `scripts/02_preprocess.py` | `data/processed/phase2/products_clean.jsonl` + `materials_catalog.json` |
| 2. Format (2-block prompt) | `scripts/03_format.py` | `data/processed/phase3/{train,val,test}.jsonl` |
| 3. Train | `training/launcher.py` → `training/training_code/train.py` | merged Gemma model in S3 |
| 4. Evaluate | `eval/evaluate.py` | metrics (curated vs caption condition) |
| 4. Smoke test | `eval/smoke_test.py` | end-to-end image→BOM on a few samples |
| 5. Serve / UI | `ui/app.py` + `ui/endpoint.py` | Streamlit, on-demand endpoint |

## Setup

```bash
cd mecca-pkg-llm
uv sync                      # creates .venv with local-side deps
cp .env.example .env         # fill in AWS_PROFILE, region, bucket, role ARN, HF_TOKEN
```

### Getting an `HF_TOKEN` (HuggingFace)

Gemma 3 is a **gated** model, so you need a HuggingFace read token to download it:

1. Create / log in to a HuggingFace account: https://huggingface.co/join
2. Accept the Gemma license on **both** model pages (one click each, usually instant approval):
   - [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) - cheap dev runs
   - [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) - the real run
3. Create a **Read** token at https://huggingface.co/settings/tokens (`New token` → type `Read`).
4. Copy the value (it starts with `hf_`) into `HF_TOKEN=` in your `.env`.

The other `.env` values (AWS profile/region, bucket, role ARN) can be shared with you directly.

Then run steps in order - see each script's `--help`.

## Cost guardrails

- Use the **4B** model for dev iterations (`--model_id google/gemma-3-4b-it`), **12B** for the real run.
- Prefer the Async/Serverless option in `ui/endpoint.py` - serverless scales to zero between calls.

### ⚠️ Deploy the endpoint only while using it, and DELETE it when done

A SageMaker endpoint **bills for every hour it exists**, whether or not you send it any requests.
A forgotten real-time `ml.g6.12xlarge` endpoint costs **~$120+/day**. Always:

```bash
# Deploy only when you're about to use it (eval, smoke test, or the UI):
uv run ui/endpoint.py deploy --name gemma3-bom-ep --model-s3 s3://<bucket>/.../model/ --serverless

# ...use it...

# Tear it down the moment you're finished:
uv run ui/endpoint.py delete --name gemma3-bom-ep
uv run ui/endpoint.py list        # confirm nothing is left InService
```

The Streamlit UI also has **Deploy / Delete** buttons in the sidebar (`ui/app.py`) that spin the
endpoint up on demand and tear it down on exit - but **double-check with `endpoint.py list`** that
nothing is still running before you walk away. See the
[RUNBOOK teardown checklist](RUNBOOK.md) (step 6) for the full procedure.
