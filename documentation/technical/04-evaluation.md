# Evaluation

## `eval/evaluate.py` - two-condition Step-2 evaluation

Evaluates the text→BOM model **in isolation** (Step 2 only) against the held-out test set,
running two conditions on the **same** products
([docstring](../../eval/evaluate.py#L1)):

| Condition | `Visual:` block | Meaning |
|---|---|---|
| **caption** | caption-style text via [`caption_from_description()`](../../scripts/03_format.py#L149) (the same fn used at format time) | **Production-realistic - the ship decision** |
| **description** | the raw curated marketing description | Upper bound / v1-comparable |

The **distribution tax** = `description.material_abbrev_set_f1 − caption.material_abbrev_set_f1`
([L140-142](../../eval/evaluate.py#L140)). A large positive gap ⇒ the model over-relies on
rich descriptions; production sees the caption number.

### Flow

1. Read test SKUs from
   [data/processed/phase3/split_manifest.json](../../data/processed/phase3/split_manifest.json)
   ([L84-85](../../eval/evaluate.py#L84)).
2. [`load_test_products()`](../../eval/evaluate.py#L59) loads raw fields + gold BOM from
   `products_clean.jsonl`; sort + apply `--limit` ([L87](../../eval/evaluate.py#L87)).
3. For each product × condition, build the `Visual:` block, call
   [`client.recommend(visual, inp)`](../../src/mecca_pkg_llm/inference.py#L121) (text-path only),
   and score via [`metrics.score_one()`](../../src/mecca_pkg_llm/metrics.py#L88)
   ([L99-114](../../eval/evaluate.py#L99)). Invoke failures are caught and scored as empty
   output.
4. Aggregate with [`metrics.aggregate()`](../../src/mecca_pkg_llm/metrics.py#L120) and write
   outputs ([L118-125](../../eval/evaluate.py#L118)).

### Outputs

- `eval/results/<timestamp>/metrics.json` - per-condition aggregate scorecard.
- `eval/results/<timestamp>/predictions_{caption,description}.jsonl` - per-example
  `{sku, visual, pred}` for inspection.

### CLI

```bash
uv run eval/evaluate.py --endpoint-name gemma3-bom-ep --limit 100
uv run eval/evaluate.py --endpoint-name gemma3-bom-ep --condition caption
```

| Flag | Default | Meaning |
|---|---|---|
| `--endpoint-name` | *(required)* | SageMaker endpoint |
| `--condition` | `both` | `caption` / `description` / `both` |
| `--limit` | `100` | Max test products to score |
| `--out-dir` | `eval/results` | Output root |

## The metrics ([src/mecca_pkg_llm/metrics.py](../../src/mecca_pkg_llm/metrics.py))

[`score_one()`](../../src/mecca_pkg_llm/metrics.py#L88) parses the prediction
([`try_parse()`](../../src/mecca_pkg_llm/metrics.py#L22) tolerates ```` ```json ```` fences and
trailing prose) and computes, all **set-based / order-insensitive**:

| Metric | What it measures |
|---|---|
| `parse_success_rate` | fraction of outputs that are valid JSON |
| `schema_compliance_rate` | fraction with exactly `{PP, SP, TP}` top-level keys |
| `material_type_set_f1` | set-F1 over `(layer, material_type)` |
| `material_abbrev_set_f1` | set-F1 over `(layer, material_name)` - **the discriminating signal** |
| `mass_g_total_mape` | mean abs % error on total predicted mass vs gold |
| `component_count_mae` | mean abs error in component count per layer |

[`aggregate()`](../../src/mecca_pkg_llm/metrics.py#L120) averages these across all examples.

## Real numbers (current, expected-low)

From [eval/results/20260610-152712/metrics.json](../../eval/results/20260610-152712/metrics.json):

| Metric | caption | description |
|---|---|---|
| n | 30 | 30 |
| parse_success_rate | **0.200** | 0.167 |
| schema_compliance_rate | 0.200 | 0.167 |
| material_type_set_f1 | 0.182 | 0.146 |
| material_abbrev_set_f1 | **0.138** | 0.118 |
| mass_g_total_mape | 0.306 | 0.184 |
| component_count_mae | 0.500 | 0.333 |

**Why so low (and why that's fine):** the deployed endpoint is the **4B `dev` model trained
on the OLD target schema**, and full BOMs frequently **truncate** mid-JSON (the parse failure
dominates the score). This is the pre-real-run baseline. The remediation is documented in
[RUNBOOK.md](../../RUNBOOK.md) ("Retrain required for accurate dimensions / recycled content")
- the inference client already bumped `max_new_tokens` to 4096 to fit a complete BOM
([inference.py](../../src/mecca_pkg_llm/inference.py#L121)).

## `eval/smoke_test.py` - end-to-end image→BOM

Exercises **both** Step 1 and Step 2 against the live endpoint with **real photos**: per image
it calls [`client.run(image_bytes, meta)`](../../src/mecca_pkg_llm/inference.py#L142) (caption +
rich describe + `recommend_enriched`), validates the BOM parses/schema, prints the caption and
a snippet, and appends to `eval/results/smoke.jsonl`. Metadata can be supplied per image via a
sidecar `<image>.json` (`brand, pack_volume, pack_volume_unit, mfr_region, eol_region`);
otherwise sensible defaults are used.

```bash
uv run eval/smoke_test.py --endpoint-name gemma3-bom-ep --images ./sample_photos
```

## `eval/test_offline.py` - no-GPU sanity

Three checks runnable with zero cloud cost:

1. **Metrics correctness** - perfect output (gold re-serialized) → F1 = 1.0, MAPE = 0.0;
   garbage → parse_success = 0.0.
2. **Prompt consistency (critical)** - asserts the training formatter's schema-doc + full
   prompt (`scripts/03_format.py`) **byte-match** the inference builder
   ([`build_bom_prompt()`](../../src/mecca_pkg_llm/prompts.py#L130)). A mismatch is silent
   accuracy loss; this is the guardrail for the duplicated prompt logic noted in
   [SFT](03-sft.md).
3. **Client wiring** - injects a fake SageMaker runtime and validates `caption()`,
   `recommend()`, and `run()` plumb correctly.

```bash
uv run eval/test_offline.py
```

> Ship decision = the **caption** condition numbers in `eval/results/*/metrics.json`.

---

*Previous: [SFT](03-sft.md) · Next: [Streamlit app →](05-streamlit-app.md) · Plain-language
version: [non-technical/04-evaluation.md](../non-technical/04-evaluation.md).*
