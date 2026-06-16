"""Text-only collator for Gemma 3 with assistant-only loss masking.

We fine-tune step 2 (text -> BOM) only, so there are no images in the training data.
This collator uses the Gemma 3 *tokenizer* (via processor.tokenizer) and applies the
Gemma chat template, masking the prompt so loss is computed on the assistant answer only.

Why assistant-only loss: the prompt is given context the model conditions on, not
something to predict. Standard pattern: labels[:prompt_len] = -100.

Gemma 3 specifics vs Qwen:
  - The processor is a Gemma3Processor; processor.tokenizer is the text tokenizer.
  - Gemma's chat template uses <start_of_turn>/<end_of_turn> turn markers. Applying the
    template to (full conversation) vs (prompt only, add_generation_prompt=True) gives a
    matching prefix, so the prompt-length mask aligns exactly with where the answer starts.
  - Gemma has no system role in its template; our examples are user+assistant only.
"""
from __future__ import annotations

from typing import Any

import torch  # type: ignore[import]


class Gemma3TextCollator:
    def __init__(self, processor: Any, max_length: int = 4096) -> None:
        self.processor = processor
        # Gemma3Processor exposes .tokenizer; fall back to the object itself if a bare
        # tokenizer was passed.
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.max_length = max_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_texts: list[str] = []
        prompt_texts: list[str] = []

        for ex in examples:
            messages = ex["messages"]
            full = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_texts.append(full)

            prompt_only = [m for m in messages if m.get("role") != "assistant"]
            prompt = self.tokenizer.apply_chat_template(
                prompt_only, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(prompt)

        batch = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        for i, prompt_text in enumerate(prompt_texts):
            prompt_ids = self.tokenizer(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            prompt_len = min(len(prompt_ids), labels.size(1))
            labels[i, :prompt_len] = -100

        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch
