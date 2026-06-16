#!/usr/bin/env python3
"""Step 3 launcher — submit a Gemma 3 QLoRA training job to SageMaker from your laptop.

Reads .env (AWS_PROFILE / AWS_REGION / BUCKET_NAME / SAGEMAKER_ROLE_ARN / HF_TOKEN).
Uploads data/processed/phase3/{train,val}.jsonl to S3 and launches the job that runs
training/training_code/train.py inside a HuggingFace GPU container.

Cost discipline ($300 budget):
  - Default --size dev uses gemma-3-4b-it on a small instance (~$5/run) to shake out the
    whole pipeline cheaply. Use --size real (gemma-3-12b-it) for the production run.
  - Prints a cost estimate and asks for confirmation before spending.

Usage:
  uv run training/launcher.py --dry-run                 # preview only
  uv run training/launcher.py --size dev                # cheap 4B shakedown run
  uv run training/launcher.py --size real               # 12B real run
  uv run training/launcher.py --size real --epochs 5
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
log = logging.getLogger("launcher")

# Model-size presets: (model_id, instance_type, default use_qlora).
SIZE_PRESETS = {
    # 4B fits comfortably; bf16 LoRA on a single A10G/L4 is fine and a bit faster.
    "dev": ("google/gemma-3-4b-it", "ml.g6.2xlarge", False),
    # 12B QLoRA on a SINGLE L40S (48 GB) — fits the whole model with no cross-GPU
    # sharding, faster AND cheaper than the 4× L4 ml.g6.12xlarge. Vision frozen.
    "real": ("google/gemma-3-12b-it", "ml.g6e.2xlarge", True),
}

INSTANCE_HOURLY_USD = {
    # L4 (ml.g6.*)
    "ml.g6.2xlarge": 1.50, "ml.g6.4xlarge": 2.00, "ml.g6.12xlarge": 5.67,
    "ml.g6.24xlarge": 8.50, "ml.g6.48xlarge": 16.0,
    # A10G (ml.g5.*)
    "ml.g5.2xlarge": 1.51, "ml.g5.4xlarge": 2.03, "ml.g5.12xlarge": 7.09,
    "ml.g5.24xlarge": 10.18, "ml.g5.48xlarge": 20.36,
    # L40S 48GB (ml.g6e.*) — single-GPU 12B QLoRA sweet spot
    "ml.g6e.2xlarge": 2.80, "ml.g6e.4xlarge": 3.60, "ml.g6e.12xlarge": 13.02,
    "ml.g6e.24xlarge": 20.0, "ml.g6e.48xlarge": 30.0,
}

# Default training-job timeout per size (the real run can take 5–11 hrs; an 8h
# default would kill it mid-train).
MAX_RUN_HOURS_BY_SIZE = {"dev": 4, "real": 12}


def load_env(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        sys.exit(f"ERROR: {env_path} not found. Copy .env.example to .env and fill it in.")
    out: dict[str, str] = {}
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def estimate_runtime_hours(n_train: int, epochs: int, qlora: bool) -> float:
    steps_per_epoch = max(1, n_train // 8)
    sec_per_step = 5.0 if qlora else 3.5  # QLoRA forward is slower (dequant)
    overhead = 30 * 60  # download big model + merge + save
    return (steps_per_epoch * epochs * sec_per_step + overhead) / 3600


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=list(SIZE_PRESETS), default="dev",
                    help="dev=4B cheap shakedown, real=12B production (default: %(default)s)")
    ap.add_argument("--phase3-dir", type=Path, default=Path("data/processed/phase3"))
    ap.add_argument("--instance-type", default=None, choices=list(INSTANCE_HOURLY_USD),
                    help="Override the preset instance type")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-run-hours", type=int, default=None,
                    help="Training-job timeout. Default by size: dev=4h, real=12h.")
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="Cap train rows uploaded (for cheap dev shakedowns). "
                         "Default for --size dev is 800; real uses all rows.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    model_id, preset_instance, preset_qlora = SIZE_PRESETS[args.size]
    instance_type = args.instance_type or preset_instance
    max_run_hours = args.max_run_hours or MAX_RUN_HOURS_BY_SIZE[args.size]

    train_jsonl = args.phase3_dir / "train.jsonl"
    val_jsonl = args.phase3_dir / "val.jsonl"
    if not train_jsonl.exists() or not val_jsonl.exists():
        log.error("missing %s or val.jsonl — run scripts/03_format.py first", train_jsonl)
        return 2
    n_train_full = sum(1 for _ in train_jsonl.open())
    n_val = sum(1 for _ in val_jsonl.open())

    # Cap train rows for cheap dev shakedowns. A shakedown validates the pipeline
    # (data -> train -> save), not accuracy, so a few hundred rows is plenty.
    max_train = args.max_train_samples
    if max_train is None and args.size == "dev":
        max_train = 800
    n_train = min(n_train_full, max_train) if max_train else n_train_full

    env = load_env(Path(".env"))
    required = ["AWS_PROFILE", "AWS_REGION", "BUCKET_NAME", "SAGEMAKER_ROLE_ARN"]
    missing = [k for k in required if not env.get(k)]
    if missing:
        log.error(".env missing keys: %s", missing)
        return 2
    hf_token = os.environ.get("HF_TOKEN") or env.get("HF_TOKEN")
    if not hf_token:
        log.error("HF_TOKEN not set. Gemma 3 is gated — accept the license at "
                  "huggingface.co/%s and set HF_TOKEN in .env or your shell.", model_id)
        return 2

    try:
        import boto3
        import sagemaker
        from sagemaker.huggingface import HuggingFace
    except ImportError as e:
        log.error("missing dep (%s). Run: uv sync", e)
        return 2

    session = boto3.Session(profile_name=env["AWS_PROFILE"], region_name=env["AWS_REGION"])
    try:
        account_id = session.client("sts").get_caller_identity()["Account"]
    except Exception as e:  # noqa: BLE001
        log.error("AWS credentials check failed: %s", e)
        return 2

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    bucket = env["BUCKET_NAME"]
    train_s3 = f"s3://{bucket}/gemma_sft/phase3/{timestamp}/train.jsonl"
    val_s3 = f"s3://{bucket}/gemma_sft/phase3/{timestamp}/val.jsonl"
    output_path = f"s3://{bucket}/gemma_sft/models/"

    rate = INSTANCE_HOURLY_USD[instance_type]
    est_hours = estimate_runtime_hours(n_train, args.epochs, preset_qlora)
    est_cost = rate * est_hours

    print("\n" + "=" * 64)
    print(f" Gemma 3 SFT — {args.size.upper()} run  ({model_id})")
    print("=" * 64)
    print(f"  AWS account:    {account_id}   region={env['AWS_REGION']}")
    print(f"  SageMaker role: {env['SAGEMAKER_ROLE_ARN']}")
    print(f"  Instance:       {instance_type}  (~${rate:.2f}/hr)  QLoRA={preset_qlora}")
    print(f"  Train/val rows: {n_train} / {n_val}")
    print(f"  Epochs:         {args.epochs}   eff_batch="
          f"{args.per_device_batch_size * args.gradient_accumulation_steps}")
    print(f"  LoRA:           r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    print(f"  Output prefix:  {output_path}")
    print(f"  Est wall/cost:  {est_hours:.1f} h  ~${est_cost:.2f}")
    print(f"  Job timeout:    {max_run_hours} h")
    print("=" * 64 + "\n")

    if args.dry_run:
        log.info("--dry-run: not uploading or launching")
        return 0
    if not args.yes and input("Proceed (y/N)? ").strip().lower() != "y":
        log.info("cancelled.")
        return 0

    s3 = session.client("s3")
    # If capping train rows, materialise a truncated copy and upload that.
    train_upload = train_jsonl
    if n_train < n_train_full:
        train_upload = args.phase3_dir / f".train_capped_{n_train}.jsonl"
        with train_jsonl.open() as fin, train_upload.open("w") as fout:
            for i, line in enumerate(fin):
                if i >= n_train:
                    break
                fout.write(line)
        log.info("capped train to %d rows -> %s", n_train, train_upload)
    log.info("uploading train/val to s3://%s/gemma_sft/phase3/%s/", bucket, timestamp)
    s3.upload_file(str(train_upload), bucket, f"gemma_sft/phase3/{timestamp}/train.jsonl")
    s3.upload_file(str(val_jsonl), bucket, f"gemma_sft/phase3/{timestamp}/val.jsonl")

    sm_session = sagemaker.Session(boto_session=session, default_bucket=bucket)
    hyperparameters = {
        "model_id": model_id,
        "epochs": args.epochs,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "warmup_ratio": 0.03,
        "use_qlora": preset_qlora,
        "merge_adapter": True,
    }
    base_job_name = f"gemma3-{args.size}-bom-{timestamp}"

    estimator = HuggingFace(
        entry_point="train.py",
        source_dir=str(Path(__file__).parent / "training_code"),
        instance_type=instance_type,
        instance_count=1,
        role=env["SAGEMAKER_ROLE_ARN"],
        sagemaker_session=sm_session,
        transformers_version="4.49",
        pytorch_version="2.5",
        py_version="py311",
        hyperparameters=hyperparameters,
        environment={
            "HF_TOKEN": hf_token,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TRANSFORMERS_VERBOSITY": "info",
        },
        max_run=max_run_hours * 3600,
        disable_output_compression=True,  # loose files → easy endpoint deploy
        base_job_name=base_job_name,
        output_path=output_path,
    )

    log.info("submitting training job: %s", base_job_name)
    estimator.fit({"train": train_s3, "val": val_s3}, wait=not args.no_wait)

    if args.no_wait:
        log.info("--no-wait: job submitted, continues in AWS")
        return 0

    job_name = estimator.latest_training_job.name
    model_s3 = f"{output_path}{job_name}/output/model/"
    out_path = Path(__file__).parent / "last_training_job.txt"
    out_path.write_text(f"{job_name}\n{model_s3}\n")
    log.info("training complete. job=%s model=%s (recorded in %s)", job_name, model_s3, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
