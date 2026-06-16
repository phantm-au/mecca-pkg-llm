# The Dataset - technical

## Source

The dataset originates from a synthetic generation run in the sibling repo:

```
mecca-pkg-llm/dataset/products.jsonl
```

(`DEFAULT_SOURCE` in [scripts/01_extract_chunk.py](../../scripts/01_extract_chunk.py#L44).)
The run is identified by `RUN_ID = "20260609-165708"`. The source holds ~49,881 lipstick
products. Each line is a nested JSON record:

```json
{
  "product": {
    "name": "...", "brand": "...",
    "pack_volume": 3.5, "pack_volume_unit": "g",
    "mfr_region": "...", "eol_region": "...",
    "description": "..."
  },
  "PP": [ { "component_name": "...", "rigid_or_soft": ..., "is_reusable": ...,
            "dimensions_mm": {"l":..,"w":..,"h":..},
            "materials": [ {"material_name":"...","material_abbrev":"...",
                            "material_type":"...","mass_g":...,
                            "recycled_content_percent":...}, ... ] }, ... ],
  "SP": [ ... ],
  "TP": [ ... ]
}
```

- **`product`** - the conditioning attributes (what a user / product record provides).
- **`PP` / `SP` / `TP`** - the packaging Bill-of-Materials, split into **Primary**,
  **Secondary**, and **Tertiary** packaging tiers. This is the prediction target.

## What the extract step does

[scripts/01_extract_chunk.py](../../scripts/01_extract_chunk.py) **only selects** a chunk and
stamps a stable SKU - it does **not** clean or reshape (that's step 2). Logic:

1. **Pass 1 - count** ([L70-75](../../scripts/01_extract_chunk.py#L70)): stream the source
   once to count records without holding ~200 MB in RAM.
2. **Deterministic sample** ([L82-84](../../scripts/01_extract_chunk.py#L82)):
   `random.Random(seed).sample(range(total), n)` picks `n` line indices. Seeded ⇒ reruns with
   the same source select the *same* chunk (a representative random sample, not the first N).
3. **Pass 2 - emit** ([L90-105](../../scripts/01_extract_chunk.py#L90)): stream again, write
   only chosen records, adding two fields:
   - `sku = f"SYN-{run_id}-{idx+1:06d}"` → e.g. `SYN-20260609-165708-000007`. Keyed on the
     **source line index**, so the SKU is stable across reruns with the same source.
   - `_source_index` - the original line index, for traceability.

Unparseable lines are counted and skipped (`bad`); writing zero records exits non-zero.

### CLI

```bash
uv run scripts/01_extract_chunk.py                       # defaults: n=10000, seed=42
uv run scripts/01_extract_chunk.py --n 10000 --seed 42
uv run scripts/01_extract_chunk.py --source /abs/products.jsonl --out data/raw/chunk_10k.jsonl
```

| Flag | Default | Meaning |
|---|---|---|
| `--source` | the `20260609-165708` run | Source `products.jsonl` |
| `--out` | `data/raw/chunk_10k.jsonl` | Output path |
| `--n` | `10000` | Chunk size (clamped to source size) |
| `--seed` | `42` | Shuffle seed for reproducible sampling |
| `--run-id` | `20260609-165708` | Run id used to build SKUs |

## Output schema (`data/raw/chunk_10k.jsonl`)

Same nested `{product, PP, SP, TP}` record as the source, plus top-level `sku` and
`_source_index`. The reshape into the canonical `{sku, input, target}` form happens in
[step 2 →](02-preprocessing.md).

## Dataset profile (post-clean)

After preprocessing, [data/processed/phase2/profile.json](../../data/processed/phase2/profile.json)
reports the working set. Key figures:

| Metric | Value |
|---|---|
| Products | 10,000 |
| Avg components / layer | PP **4.94**, SP **2.32**, TP **3.81** |
| Catalog size | **66** materials across **7** types |

Material-type frequency across all components (how often each type appears):

| Type | Count |
|---|---|
| Paper/Board | 52,813 |
| Other | 32,069 |
| Plastic | 30,858 |
| Wood | 10,335 |
| Metal | 8,989 |
| Glass | 173 |
| Textile | 33 |

> Note: `avg_total_mass_g` in the profile (~19,892 g) sums **all tiers including bulk
> tertiary/shipping** packaging across many components, so it is dominated by outer cases -
> not the mass of a single lipstick.

## A real sample (canonical form, post step-2 reshape)

From [data/processed/phase2/products_clean.jsonl](../../data/processed/phase2/products_clean.jsonl):

```json
{
  "sku": "SYN-20260609-165708-000007",
  "input": {
    "name": "Velour Lip Colour",
    "brand": "Botanica Beauty",
    "category": "Lips",
    "subcategory": "Lipstick",
    "pack_volume": 3.5,
    "pack_volume_unit": "g",
    "mfr_region": "China",
    "eol_region": "Australia",
    "description": "A creamy matte lipstick in a natural botanical shade with long-wearing colour."
  },
  "target": {
    "PP": [
      {
        "component_name": "Lipstick Barrel",
        "rigid_or_soft": null,
        "is_reusable": null,
        "dimensions_mm": { "l": 12.0, "w": 12.0, "h": 80.0 },
        "materials": [
          { "material_name": "Polypropylene", "material_abbrev": "PP",
            "material_type": "Plastic", "mass_g": 8.5,
            "recycled_content_percent": null }
        ]
      }
    ],
    "SP": [ ... ],   // this product: 2 secondary components
    "TP": [ ... ]    // this product: 3 tertiary components
  }
}
```

(`input` = model conditioning fields; `target` = the PP/SP/TP BOM to predict.) **All products
in this run are `category: "Lips"`, `subcategory: "Lipstick"`** - set as constants in
[scripts/02_preprocess.py](../../scripts/02_preprocess.py#L50) (`PRODUCT_GROUP` /
`PRODUCT_SUBGROUP`).

---

*Next: [Preprocessing →](02-preprocessing.md) · Plain-language version:
[non-technical/01-dataset.md](../non-technical/01-dataset.md).*
