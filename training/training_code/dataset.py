"""Minimal Messages dataset — one {messages:[...]} dict per JSONL line.

The collator does all tokenization/templating; this just yields raw examples.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset  # type: ignore[import]


class MessagesDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.examples: list[dict[str, Any]] = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]
