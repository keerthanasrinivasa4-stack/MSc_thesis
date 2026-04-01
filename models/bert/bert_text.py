"""
bert_text.py
============
BERT-based text classifier for MMHS150K hate-speech detection.

Architecture
------------
  BERT-base-uncased  →  [CLS] pooled output  →  Dropout  →  Linear(768, num_classes)

Supports both:
  - Binary  : 0=NotHate, 1=Hate   (num_classes=2)
  - 6-class : 0=NotHate 1=Racist 2=Sexist 3=Homophobe 4=Religion 5=OtherHate
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertModel


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP_6 = {
    0: "NotHate",
    1: "Racist",
    2: "Sexist",
    3: "Homophobe",
    4: "Religion",
    5: "OtherHate",
}
LABEL_MAP_BINARY = {0: "NotHate", 1: "Hate"}


def majority_vote(labels: list[int]) -> int:
    """Return the most common label; ties broken by smallest."""
    return Counter(labels).most_common(1)[0][0]


def to_binary(label6: int) -> int:
    return 0 if label6 == 0 else 1


def load_split_ids(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class MMHS150KTextDataset(Dataset):
    """
    Loads tweet text + annotation labels from MMHS150K_GT.json.

    Parameters
    ----------
    ids       : list of tweet IDs for this split
    gt        : parsed MMHS150K_GT.json dict
    tokenizer : HuggingFace tokenizer
    max_len   : max token length (default 96)
    binary    : if True, collapses 6 classes → 2 (hate / not-hate)
    """

    def __init__(
        self,
        ids: list[str],
        gt: dict,
        tokenizer: BertTokenizerFast,
        max_len: int = 96,
        binary: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.binary = binary
        self.samples: list[tuple[str, int]] = []
        self.skipped = 0

        for tid in ids:
            item = gt.get(tid)
            if item is None:
                self.skipped += 1
                continue

            text = str(item.get("tweet_text") or "").strip()

            labs = item.get("labels", [])
            if not (isinstance(labs, list) and len(labs) >= 1):
                self.skipped += 1
                continue

            # Handle entries with fewer than 3 annotators (pad with last)
            while len(labs) < 3:
                labs = labs + [labs[-1]]
            labs = [int(x) for x in labs[:3]]

            y6 = majority_vote(labs)
            y = to_binary(y6) if binary else y6
            self.samples.append((text, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        text, y = self.samples[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()} | {
            "labels": torch.tensor(y, dtype=torch.long)
        }

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss."""
        counts = Counter(y for _, y in self.samples)
        n_classes = 2 if self.binary else 6
        total = len(self.samples)
        w = [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)]
        return torch.tensor(w, dtype=torch.float)

    def sample_weights(self) -> list[float]:
        """Per-sample weights for WeightedRandomSampler."""
        counts = Counter(y for _, y in self.samples)
        return [1.0 / counts[y] for _, y in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class BertTextClassifier(nn.Module):
    """
    BERT [CLS] → Dropout → Linear classifier.

    Parameters
    ----------
    model_name  : HuggingFace model identifier (default: bert-base-uncased)
    num_classes : output classes (2 for binary, 6 for multiclass)
    dropout     : dropout rate on the pooled CLS output
    freeze_bert : if True, only the head is trained (fast but weaker)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # [CLS] pooled representation → (B, 768)
        pooled = out.pooler_output
        return self.classifier(self.dropout(pooled))
