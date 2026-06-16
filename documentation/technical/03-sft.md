# Supervised Fine-Tuning (SFT) - technical

## The 2-step design (why we fine-tune text only)

The dataset has **no image↔product mapping**, so there is **no multimodal fine-tuning**.
Inference is two passes of the *same* Gemma model
([README](../../README.md), [src/mecca_pkg_llm/inference.py](../../src/mecca_pkg_llm/inference.py)):

1. **Step 1 - caption (zero-shot):** product image → terse visual caption, using Gemma's
   inherent vision ability. **Not fine-tuned.**
2. **Step 2 - recommend (fine-tuned, text-only):** caption + known metadata + materials
   catalog → BOM JSON. **Only this step is trained.**

To keep Step 1 intact, training freezes the vision encoder **and** the multimodal projector;
LoRA touches only language-model layers.

---

## Part A - Formatting ([scripts/03_format.py](../../scripts/03_format.py))

Converts `products_clean.jsonl` into HuggingFace **Messages** JSONL (`user` prompt →
`assistant` BOM target).

### The two-block prompt

The Step-2 prompt splits inputs by where they come from at inference time
([PROMPT_TEMPLATE](../../scripts/03_format.py#L186), mirrored in
[src/mecca_pkg_llm/prompts.py](../../src/mecca_pkg_llm/prompts.py#L92)):

- **`Visual:`** - caption-style text: what a *photo* would actually show. Derived from the
  curated description by rule-stripping (below). This is the train/inference-match anchor.
- **`Known metadata:`** - `Category > Subcategory`, `Brand`, `Pack size`, `Made in`,
  `Sold in` - structured fields from the product record, kept un-stripped.

Followed by the embedded **schema doc** (the closed 66-material catalog + JSON structure
spec, ~3152 chars) and `Return ONLY the JSON`.

> **Load-bearing invariant:** the prompt built here MUST byte-match what
> [prompts.build_bom_prompt()](../../src/mecca_pkg_llm/prompts.py#L130) builds at inference. The
> logic is intentionally duplicated (training box has no `src` import); a mismatch is silent
> accuracy loss. [eval/test_offline.py](../../eval/test_offline.py) asserts they match - see
> [Evaluation](04-evaluation.md).

### Rule-based caption synthesis

[`caption_from_description()`](../../scripts/03_format.py#L149) turns marketing copy into a
generic caption deterministically (no LLM call): it strips the brand
([`_strip_brand()`](../../scripts/03_format.py#L143)), marketing words
([`_MARKETING`](../../scripts/03_format.py#L128): *sustainable, eco-friendly, luxurious,
premium, recyclable, refillable…*), and explicit material words
([`_MATERIAL_WORDS`](../../scripts/03_format.py#L135): *plastic, PP, PET, aluminium, glass,
paper, carton…* → `"packaging"`), then normalises whitespace and to a caption-y opener. This
mimics the generic register Step-1's zero-shot captioner emits.

### Caption augmentation

[`caption_variants()`](../../scripts/03_format.py#L167) wraps the same base caption in up to 5
surface forms (`"{c}"`, `"Photo shows: {c}"`, `"Visible in the image: {c}"`, …) against an
**identical** target, so the model learns caption-invariance. Train default = **4 variants**
per product; val/test get **1** each for clean eval.

### Target JSON

[`build_target_json()`](../../scripts/03_format.py#L235) emits the canonical, richer BOM -
per component: `component_name, rigid_or_soft, is_reusable, dimensions_mm` (rounded via
[`_round_dims()`](../../scripts/03_format.py#L222)); per material: `material_name,
material_abbrev, material_type, mass_g, recycled_content_percent`. **Sustainability metrics
are deliberately absent** - those are catalog lookups at enrichment time, never predicted.

### SKU-level split

[`split_skus()`](../../scripts/03_format.py#L292) splits by **SKU** (a product never leaks
across splits): default `val_frac=0.10`, `test_frac=0.10`, `seed=42`. Per
[data/processed/phase3/dataset_stats.json](../../data/processed/phase3/dataset_stats.json):

| Split | SKUs | Examples | char p50 / p95 | est tokens p95 |
|---|---|---|---|---|
| train | 8,000 | 32,000 | 9,010 / 10,282 | 2,570 |
| val | 1,000 | 1,000 | 8,988 / 10,285 | 2,571 |
| test | 1,000 | 1,000 | 8,986 / 10,238 | 2,559 |

`validate_split()` enforces every assistant message parses as JSON with exactly
`{PP, SP, TP}` keys and the p95 token estimate stays under `P95_TOKEN_BUDGET=3800`.

### CLI

```bash
uv run scripts/03_format.py
uv run scripts/03_format.py --train-augments 4
uv run scripts/03_format.py --val-frac 0.1 --test-frac 0.1 --seed 42
```

---

## Part B - Training

### Launcher ([training/launcher.py](../../training/launcher.py)) - laptop side

Reads `.env`, uploads `phase3/{train,val}.jsonl` to S3, and submits a SageMaker HuggingFace
training job running `train.py`. Cost discipline for the **$300 budget**:

| Size | Model | Instance | ~$/hr | QLoRA |
|---|---|---|---|---|
| `dev` | `google/gemma-3-4b-it` | `ml.g6.2xlarge` | 1.50 | off (bf16 LoRA) |
| `real` | `google/gemma-3-12b-it` | `ml.g6e.2xlarge` (L40S 48 GB) | 2.80 | on |

([SIZE_PRESETS](../../training/launcher.py#L34), [hourly table](../../training/launcher.py#L42))
The launcher prints an estimate ([`estimate_runtime_hours()`](../../training/launcher.py#L72))
and **asks for confirmation** before spending; `--dry-run` previews only. For cheap dev
shakedowns it caps train rows to 800 by default ([L120](../../training/launcher.py#L120)).
Records the job + model S3 URI in
[training/last_training_job.txt](../../training/last_training_job.txt).

```bash
uv run training/launcher.py --size dev --dry-run     # preview cost
uv run training/launcher.py --size dev               # 4B shakedown (~$1-2)
uv run training/launcher.py --size real --epochs 2   # 12B real run
```

### Training entry point ([train.py](../../training/training_code/train.py)) - runs in the GPU container

1. **Load** Gemma 3 via `Gemma3ForConditionalGeneration`, optionally 4-bit NF4 QLoRA
   (`BitsAndBytesConfig`, bf16 compute), `attn_implementation="eager"`
   ([L102-124](../../training/training_code/train.py#L102)). A float8 compatibility shim
   aliases missing dtypes before importing transformers/peft
   ([L35-37](../../training/training_code/train.py#L35)).
2. **Freeze vision + projector** -
   [`freeze_vision_and_projector()`](../../training/training_code/train.py#L74) sets
   `requires_grad=False` on params matching `vision_tower`, `multi_modal_projector`,
   `multimodal_projector`. **`frozen==0` is logged as a red flag** (marker names drifted).
   ⇒ the image→embedding path stays bit-identical to base Gemma, so Step-1 captioning is
   provably unaffected.
3. **LoRA on LM layers only** ([L154-167](../../training/training_code/train.py#L154)):
   `target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]`,
   `r=32, alpha=64, dropout=0.05`, plus `exclude_modules` regex keeping LoRA off any
   vision/projector linears (belt-and-suspenders with the freeze).
4. **Train** with TRL `SFTTrainer` + the text-only collator (below).
5. **Merge + save** ([L213-242](../../training/training_code/train.py#L213)): merge LoRA into
   the base and save a clean servable artifact (safetensors + processor) to `/opt/ml/model`.
   For QLoRA the 4-bit base can't be merged directly, so it reloads the bf16 base, attaches
   the adapter, merges, and saves full bf16 weights.

### Assistant-only loss ([collator.py](../../training/training_code/collator.py))

[`Gemma3TextCollator`](../../training/training_code/collator.py#L24) applies Gemma's chat
template, tokenizes the full conversation, then masks the prompt tokens
(`labels[:prompt_len] = -100`) so loss is computed on the **assistant BOM answer only**. Pad
tokens are also masked. Gemma's `<start_of_turn>`/`<end_of_turn>` markers make the prompt-only
prefix align exactly with where the answer begins.

The dataset itself is a thin JSONL reader
([dataset.py](../../training/training_code/dataset.py) - `MessagesDataset` yields raw
`{messages:[...]}` dicts; the collator does all templating/tokenization).

---

*Previous: [Preprocessing](02-preprocessing.md) · Next: [Evaluation →](04-evaluation.md) ·
Plain-language version: [non-technical/03-sft.md](../non-technical/03-sft.md). Operational run
sequence: [RUNBOOK.md](../../RUNBOOK.md).*
