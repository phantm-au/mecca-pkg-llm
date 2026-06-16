# Data Preprocessing

## Pipeline overview

"DPK-style" = the same conceptual stages [data-prep-kit](https://github.com/data-prep-kit)
uses (ingest, dedup, filter, profile), implemented as light local pandas/Python - no
Ray/Parquet overhead for a 10K job
([module docstring](../../scripts/02_preprocess.py#L1)). Five stages:

| # | Stage | Function | Purpose |
|---|---|---|---|
| 1 | INGEST / reshape | `reshape()` | Nested record → canonical `{sku, input, target}` |
| 2 | VALIDATE | `is_valid()` | Drop unusable records, with drop-reason counts |
| 3 | DEDUP | `dedup()` | MinHash LSH near-duplicate removal |
| 4 | PROFILE | `profile()` | Dataset census → `profile.json` |
| 5 | CATALOG | `build_catalog()` | Closed-vocabulary materials list → `materials_catalog.json` |

(Numbered by the docstring's logical order; in `main()` the catalog is built before the
profile because the profile references catalog counts.)

## Stage 1 - INGEST / reshape

[`reshape()`](../../scripts/02_preprocess.py#L62) maps the nested synthetic record to:

- **`input`** - `name, brand, category, subcategory, pack_volume, pack_volume_unit,
  mfr_region, eol_region, description`. `category`/`subcategory` are hard-set to
  `PRODUCT_GROUP="Lips"` / `PRODUCT_SUBGROUP="Lipstick"`
  ([L50-51](../../scripts/02_preprocess.py#L50)) - this run is all lipstick.
- **`target`** - `{PP, SP, TP}`, each a list of components carrying
  `component_name, rigid_or_soft, is_reusable, dimensions_mm, materials[]`; each material
  carries `material_name, material_abbrev, material_type, mass_g, recycled_content_percent`.

Records with no `sku` return `None` and are dropped.

## Stage 2 - VALIDATE

[`is_valid()`](../../scripts/02_preprocess.py#L112) returns `(ok, reason)`. A record is
dropped if:

| Reason | Condition |
|---|---|
| `no_name` | `input.name` missing |
| `no_primary_packaging` | `target.PP` empty |
| `material_no_name` | a material has no `material_name` |
| `bad_material_type:<x>` | `material_type` ∉ `VALID_MATERIAL_TYPES` (Plastic, Paper/Board, Metal, Glass, Wood, Textile, Other) |
| `material_no_mass` | `mass_g` is not int/float |

Components with an empty `materials` list are tolerated (skipped downstream, not a reject).
Drop-reason counts are logged via a `Counter`.

## Stage 3 - DEDUP (MinHash LSH)

[`dedup()`](../../scripts/02_preprocess.py#L153) removes near-duplicates using
`datasketch.MinHashLSH`:

- **Signature** ([`product_signature()`](../../scripts/02_preprocess.py#L137)): a *set* of
  tokens from `name + brand + description` plus every `component_name` and `material_name`
  across PP/SP/TP (order-independent).
- **Parameters:** `num_perm=128`, Jaccard `threshold=0.92` (default `--dedup-threshold`).
- **Algorithm:** for each product, build a MinHash; if `lsh.query(mh)` already finds a kept
  near-dup, drop it; otherwise insert and keep. First-come-first-kept.
- Disable with `--no-dedup`.

## Stage 5 - CATALOG (closed vocabulary)

[`build_catalog()`](../../scripts/02_preprocess.py#L180): the synthetic set has no canonical
catalog CSV, so it is **derived from the data**. It counts every
`(material_name, material_abbrev, material_type)` combo, then resolves each `material_name` to
its **most frequent** `(abbrev, type)` pairing (handles inconsistent abbrev/type for the same
name). Result → `materials_catalog.json`:

```json
{
  "n_materials": 66,
  "type_counts": {"Plastic": 20, "Paper/Board": 20, "Other": 12,
                  "Metal": 5, "Wood": 5, "Textile": 3, "Glass": 1},
  "materials": [
    {"material_name": "Acrylic", "material_abbrev": null, "material_type": "Plastic"},
    {"material_name": "Acrylonitrile Butadiene Styrene", "material_abbrev": "ABS",
     "material_type": "Plastic"},
    {"material_name": "Adhesive", "material_abbrev": null, "material_type": "Other"},
    ...
  ]
}
```

This 66-material list is **embedded verbatim into every training and inference prompt** (see
[SFT →](03-sft.md)) as a closed vocabulary the model must choose from - the single biggest
accuracy lever. It is loaded by
[`build_schema_doc()`](../../src/mecca_pkg_llm/prompts.py#L109) /
[`load_schema_doc()`](../../src/mecca_pkg_llm/prompts.py#L125).

## Stage 4 - PROFILE

[`profile()`](../../scripts/02_preprocess.py#L216) →
[data/processed/phase2/profile.json](../../data/processed/phase2/profile.json):

```json
{
  "n_products": 10000,
  "avg_components_per_layer": {"PP": 4.94, "SP": 2.32, "TP": 3.81},
  "avg_total_mass_g": 19892.09,
  "material_type_frequency": {"Plastic": 30858, "Other": 32069, "Paper/Board": 52813,
                              "Wood": 10335, "Metal": 8989, "Glass": 173, "Textile": 33},
  "catalog_size": 66,
  "catalog_type_counts": {"Plastic": 20, "Other": 12, "Metal": 5, "Wood": 5,
                          "Paper/Board": 20, "Textile": 3, "Glass": 1}
}
```

### CLI

```bash
uv run scripts/02_preprocess.py
uv run scripts/02_preprocess.py --no-dedup
uv run scripts/02_preprocess.py --dedup-threshold 0.9
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `data/raw/chunk_10k.jsonl` | Input chunk |
| `--out-dir` | `data/processed/phase2` | Output directory |
| `--no-dedup` | off | Skip MinHash dedup |
| `--dedup-threshold` | `0.92` | Jaccard similarity ⇒ near-dup |

## The environmental catalog - `catalog/env_catalog.json`

A **separate** catalog (not produced by this script) that powers deterministic sustainability
enrichment later (see [Streamlit app](05-streamlit-app.md) and
[src/mecca_pkg_llm/enrich.py](../../src/mecca_pkg_llm/enrich.py)). It also covers all **66 materials**.
Top-level keys: `n_materials`, `source`, `materials`. `source` =
`"mecca-streamlit materials_catalog.json (environmental block)"`. Per-material entry:

```json
"Acrylic": {
  "material_abbrev": null,
  "material_type": "Plastic",
  "recycling_potential": 0.0,
  "carbon_kg": 3.32,          // kg CO2e per KG of material
  "water_consumption": 1171.5, // L per KG of material
  "fossil_fuel_use": 56.85
}
```

`carbon_kg` and `water_consumption` are **per-kilogram intensities**. At enrichment time the
model's predicted `mass_g` is converted to kg and multiplied by these intensities to produce
an absolute footprint - see [`enrich_bom()`](../../src/mecca_pkg_llm/enrich.py#L75). The model
never predicts these numbers; they are a catalog join keyed on `material_name`.

---

*Previous: [Dataset](01-dataset.md) · Next: [SFT →](03-sft.md) · Plain-language version:
[non-technical/02-preprocessing.md](../non-technical/02-preprocessing.md).*
