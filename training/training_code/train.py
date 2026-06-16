"""Training entry point — runs INSIDE the SageMaker GPU container.

Ships in source_dir=training/training_code/. SageMaker invokes it as
`python train.py --hyperparam value ...`.

Pipeline:
  1. Load Gemma 3 (4B/12B) vision-language model in 4-bit (QLoRA) + processor.
  2. FREEZE the vision encoder AND the multimodal projector/connector.
     => the image->text-embedding path stays bit-identical to base Gemma, so step-1
        zero-shot captioning is provably unaffected by this text-only SFT.
  3. Apply LoRA to the LANGUAGE-MODEL attention+MLP layers only.
  4. Train with TRL SFTTrainer + the text-only collator (assistant-only loss).
  5. Merge LoRA into the base weights and save a clean, servable artifact
     (safetensors + processor) to /opt/ml/model.

NOTE: we are NOT importing to Bedrock (Gemma isn't CMI-supported), so there is no
`transformers==4.51.3` pin and no Bedrock file-manifest check. We match whatever
transformers version the Gemma serving container expects (see requirements.txt).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch  # type: ignore[import]

# Compatibility shim: the base DLC ships torch 2.5.1, which lacks the float8 dtypes
# that newer transformers/peft reference unconditionally (peft 0.14 casts adapter
# dtype assuming torch.float8_e8m0fnu exists -> AttributeError on 2.5.1). Alias the
# missing dtypes to float32 BEFORE importing transformers/peft so the import + adapter
# casting succeed. Harmless on newer torch where the attrs already exist.
for _dt in ("float8_e8m0fnu", "float8_e4m3fn", "float8_e5m2"):
    if not hasattr(torch, _dt):
        setattr(torch, _dt, torch.float32)

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="google/gemma-3-12b-it")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_qlora", type=str, default="True", help="4-bit QLoRA (vs bf16 LoRA)")
    p.add_argument("--merge_adapter", type=str, default="True")
    return p.parse_args()


def _bool(s: str | bool) -> bool:
    return s if isinstance(s, bool) else s.strip().lower() in ("true", "1", "yes", "y")


# Substrings that identify the vision / multimodal-projector parameters in Gemma 3.
# These are frozen so they never receive gradients and stay identical to base Gemma.
# Gemma 3 (Gemma3ForConditionalGeneration) names them: `vision_tower.*` (SigLIP encoder)
# and `multi_modal_projector.*` (the connector mapping image features into the LM space).
VISION_FREEZE_MARKERS = ("vision_tower", "multi_modal_projector", "multimodal_projector")


def freeze_vision_and_projector(model) -> tuple[int, int]:
    """Set requires_grad=False on all vision-encoder and projector params.

    Returns (frozen_count, total_count) for logging — frozen_count==0 is a red flag
    that the marker names don't match this model revision.
    """
    frozen = total = 0
    for name, param in model.named_parameters():
        total += 1
        if any(marker in name for marker in VISION_FREEZE_MARKERS):
            param.requires_grad = False
            frozen += 1
    return frozen, total


def main() -> int:
    args = parse_args()
    args.use_qlora = _bool(args.use_qlora)
    args.merge_adapter = _bool(args.merge_adapter)

    import transformers  # noqa: PLC0415
    log.info("transformers=%s torch=%s cuda=%s",
             transformers.__version__, torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("GPUs: %d × %s", torch.cuda.device_count(), torch.cuda.get_device_name(0))

    from transformers import AutoProcessor, Gemma3ForConditionalGeneration  # noqa: PLC0415

    # ---- 1. Load model (QLoRA 4-bit or bf16 LoRA) ----
    quant_config = None
    if args.use_qlora:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        log.info("QLoRA: loading in 4-bit NF4 (bf16 compute)")
    else:
        log.info("bf16 LoRA: loading full-precision base")

    log.info("loading base model: %s", args.model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation="eager",  # Gemma 3 recommends eager attention for stability
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    # Gemma has no pad token by default; reuse eos for padding (masked out in labels).
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    log.info("model + processor loaded")

    # ---- 2. Freeze vision encoder + projector (the safeguard) ----
    frozen, total = freeze_vision_and_projector(model)
    log.info("froze %d/%d param tensors matching %s", frozen, total, VISION_FREEZE_MARKERS)
    if frozen == 0:
        log.warning(
            "NO vision/projector params matched the freeze markers — step-1 captioning "
            "may degrade. Inspect model.named_parameters() and update VISION_FREEZE_MARKERS."
        )

    if args.use_qlora:
        # prepare_model_for_kbit_training() handles input-grad enabling for QLoRA.
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415
        model = prepare_model_for_kbit_training(
            model, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        # bf16 LoRA path: with gradient_checkpointing on a (partially) frozen base,
        # the input embeddings don't require grad, so the checkpointed forward yields
        # a graph with no grad_fn -> "element 0 of tensors does not require grad".
        # Registering this hook makes the embedding outputs require grad. (QLoRA's
        # prepare_model_for_kbit_training does this for us above.)
        model.enable_input_require_grads()

    # ---- 3. LoRA on language-model layers only ----
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # exclude_modules keeps LoRA off any matching vision/projector linear layers,
        # belt-and-suspenders with the requires_grad freeze above.
        exclude_modules=r".*(vision_tower|multi_modal_projector).*",
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 4. Data + collator + trainer ----
    train_dir = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    val_dir = Path(os.environ.get("SM_CHANNEL_VAL", "/opt/ml/input/data/val"))

    from dataset import MessagesDataset  # noqa: PLC0415
    from collator import Gemma3TextCollator  # noqa: PLC0415

    train_ds = MessagesDataset(train_dir / "train.jsonl")
    val_ds = MessagesDataset(val_dir / "val.jsonl")
    log.info("train=%d val=%d examples", len(train_ds), len(val_ds))
    collator = Gemma3TextCollator(processor=processor, max_length=args.max_seq_length)

    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415
    training_args = SFTConfig(
        output_dir="/opt/ml/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        max_seq_length=args.max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    )

    log.info("trainer.train() starting")
    trainer.train()

    # ---- 5. Merge LoRA + save servable artifact ----
    output_dir = Path("/opt/ml/model")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_adapter and not args.use_qlora:
        log.info("merging LoRA into bf16 base")
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    elif args.merge_adapter and args.use_qlora:
        # Can't directly merge into a 4-bit base. Reload base in bf16, attach the trained
        # adapter, merge, then save full bf16 weights — what the serving container loads.
        log.info("reloading bf16 base to merge QLoRA adapter")
        adapter_dir = output_dir / "_adapter"
        model.save_pretrained(adapter_dir)
        del model
        torch.cuda.empty_cache()
        from peft import PeftModel  # noqa: PLC0415
        base = Gemma3ForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    else:
        log.info("saving LoRA adapter only (no merge)")
        model.save_pretrained(output_dir)

    processor.save_pretrained(output_dir)
    log.info("saved merged model + processor to %s", output_dir)
    log.info("contents: %s", sorted(p.name for p in output_dir.glob("*")))
    log.info("training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
