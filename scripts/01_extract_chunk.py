#!/usr/bin/env python3
"""extract a 10K-product chunk from the synthetic dataset.

Source (the synthetic generation run):
    mecca-pkg-llm/dataset/products.jsonl   (~49,881 lipstick products)

Each source line is a nested record:
    {
      "product": {name, brand, pack_volume, pack_volume_unit, mfr_region, eol_region, description},
      "PP": [ {component_name, dimensions_mm, materials:[{material_name, material_abbrev,
               material_type, mass_g, recycled_content_percent}], ...}, ... ],
      "SP": [ ... ],
      "TP": [ ... ]
    }

Output:
    data/raw/chunk_10k.jsonl   — same records, each with an added top-level "sku".

Usage:
    uv run scripts/01_extract_chunk.py
    uv run scripts/01_extract_chunk.py --n 10000 --seed 42
    uv run scripts/01_extract_chunk.py --source /abs/path/products.jsonl --out data/raw/chunk_10k.jsonl
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import orjson

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
log = logging.getLogger("extract")

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "mecca-pkg-llm/dataset/products.jsonl"
)

# The run_id used to build stable SKUs. SKUs look like SYN-20260609-165708-000001,
# matching the scheme already present in the run's material_view.csv.
RUN_ID = "20260609-165708"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help="Source products.jsonl (default: the 20260609-165708 run)")
    ap.add_argument("--out", type=Path, default=Path("data/raw/chunk_10k.jsonl"))
    ap.add_argument("--n", type=int, default=10_000, help="Chunk size (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducible sampling")
    ap.add_argument("--run-id", default=RUN_ID, help="Run id used to build SKUs")
    args = ap.parse_args()

    if not args.source.exists():
        log.error("source not found: %s", args.source)
        return 2

    log.info("counting records in %s", args.source)
    total = 0
    with args.source.open("rb") as f:
        for _ in f:
            total += 1
    log.info("source has %d records", total)

    n = min(args.n, total)
    if n < args.n:
        log.warning("requested %d but source only has %d; taking all", args.n, total)

    rng = random.Random(args.seed)
    chosen = set(rng.sample(range(total), n))
    log.info("selected %d record indices (seed=%d)", len(chosen), args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    bad = 0
    with args.source.open("rb") as fin, args.out.open("wb") as fout:
        for idx, line in enumerate(fin):
            if idx not in chosen:
                continue
            try:
                rec = orjson.loads(line)
            except Exception as e:  # noqa: BLE001
                bad += 1
                log.debug("skip unparseable line %d: %s", idx, e)
                continue
            rec["sku"] = f"SYN-{args.run_id}-{idx + 1:06d}"
            rec["_source_index"] = idx
            fout.write(orjson.dumps(rec))
            fout.write(b"\n")
            written += 1

    log.info("wrote %s (%d records, %d unparseable skipped)", args.out, written, bad)
    if written == 0:
        log.error("no records written — check the source file")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
