"""
multimodal_model.py
===================
Late-fusion multimodal classifier for MMHS150K hate-speech detection.

Architecture
------------
  Text branch  : BERT-base-uncased  →  [CLS] (768)  →  Linear(768 → 512)  →  ReLU
  Image branch : ResNet50 (pretrained) →  GAP (2048) →  Linear(2048 → 512) →  ReLU
  Fusion       : Concat [512 ‖ 512]  →  Dropout(0.4)  →  Linear(1024 → num_classes)

Both branches are fine-tuned jointly from pre-trained weights.

Supports:
  - Binary  : 0 = NotHate, 1 = Hate
  - 6-class : 0 = NotHate  1 = Racist  2 = Sexist  3 = Homophobe  4 = Religion  5 = OtherHate
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms
from transformers import BertModel, BertTokenizerFast


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
    return Counter(labels).most_common(1)[0][0]


def to_binary(label6: int) -> int:
    return 0 if label6 == 0 else 1


def load_split_ids(path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Image transforms
# ─────────────────────────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — requires BOTH tweet text AND image
# ─────────────────────────────────────────────────────────────────────────────
class MMHS150KMultimodalDataset(Dataset):
    """
    Returns (text_encoding, image_tensor, label) for samples that have
    both tweet text and a corresponding resized image.

    Parameters
    ----------
    ids       : tweet IDs for this split
    gt        : parsed MMHS150K_GT.json dict
    img_dir   : path to img_resized/ folder
    tokenizer : HuggingFace BERT tokenizer
    img_tf    : torchvision transform for images
    max_len   : max token length (default 96)
    binary    : binary vs 6-class labels
    """

    def __init__(
        self,
        ids: list[str],
        gt: dict,
        img_dir: Path,
        tokenizer: BertTokenizerFast,
        img_tf=None,
        max_len: int = 96,
        binary: bool = True,
    ):
        self.img_dir   = Path(img_dir)
        self.tokenizer = tokenizer
        self.img_tf    = img_tf or EVAL_TRANSFORMS
        self.max_len   = max_len
        self.binary    = binary
        # (tweet_id, text, img_path, label)
        self.samples: list[tuple[str, str, str, int]] = []
        self.skipped   = 0

        for tid in ids:
            item = gt.get(tid)
            if item is None:
                self.skipped += 1
                continue

            text = str(item.get("tweet_text") or "").strip()
            if not text:
                self.skipped += 1
                continue

            img_path = self.img_dir / f"{tid}.jpg"
            if not img_path.exists():
                self.skipped += 1
                continue

            labs = item.get("labels", [])
            if not (isinstance(labs, list) and len(labs) >= 1):
                self.skipped += 1
                continue

            while len(labs) < 3:
                labs = labs + [labs[-1]]
            labs = [int(x) for x in labs[:3]]

            y6 = majority_vote(labs)
            y  = to_binary(y6) if binary else y6
            self.samples.append((tid, text, str(img_path), y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        _, text, img_path, y = self.samples[idx]

        # ── Text encoding ─────────────────────────────────────────────────
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        text_data = {k: v.squeeze(0) for k, v in enc.items()}

        # ── Image ─────────────────────────────────────────────────────────
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        img_tensor = self.img_tf(img)

        return text_data, img_tensor, torch.tensor(y, dtype=torch.long)

    def class_weights(self) -> torch.Tensor:
        counts    = Counter(y for *_, y in self.samples)
        n_classes = 2 if self.binary else 6
        total     = len(self.samples)
        w = [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)]
        return torch.tensor(w, dtype=torch.float)

    def sample_weights(self) -> list[float]:
        counts = Counter(y for *_, y in self.samples)
        return [1.0 / counts[y] for *_, y in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
# Model — Late-fusion BERT + ResNet50
# ─────────────────────────────────────────────────────────────────────────────
class MultimodalClassifier(nn.Module):
    """
    Late-fusion of BERT text features and ResNet50 image features.

    Text  : BERT [CLS] (768) → Linear(768, 512) → ReLU
    Image : ResNet50 GAP (2048) → Linear(2048, 512) → ReLU
    Fusion: Concat(512+512=1024) → Dropout → Linear(1024, num_classes)

    Parameters
    ----------
    num_classes    : 2 (binary) or 6 (multiclass)
    bert_name      : HuggingFace model id
    fusion_dropout : dropout before the final classifier
    freeze_bert    : freeze BERT during training (not recommended)
    freeze_cnn     : freeze ResNet backbone during training (not recommended)
    """

    PROJ_DIM = 512   # projection dimension for each branch

    def __init__(
        self,
        num_classes: int = 2,
        bert_name: str = "bert-base-uncased",
        fusion_dropout: float = 0.4,
        freeze_bert: bool = False,
        freeze_cnn: bool = False,
    ):
        super().__init__()

        # ── Text branch ───────────────────────────────────────────────────
        self.bert = BertModel.from_pretrained(bert_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.text_proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.PROJ_DIM),
            nn.ReLU(),
        )

        # ── Image branch ──────────────────────────────────────────────────
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features   # 2048
        backbone.fc = nn.Identity()             # remove original head
        self.resnet = backbone
        if freeze_cnn:
            for p in self.resnet.parameters():
                p.requires_grad = False
        self.img_proj = nn.Sequential(
            nn.Linear(in_features, self.PROJ_DIM),
            nn.ReLU(),
        )

        # ── Fusion head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(self.PROJ_DIM * 2, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Text features
        bert_out  = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        text_feat = self.text_proj(bert_out.pooler_output)   # (B, 512)

        # Image features
        img_feat  = self.img_proj(self.resnet(pixel_values)) # (B, 512)

        # Late fusion
        fused     = torch.cat([text_feat, img_feat], dim=1)  # (B, 1024)
        return self.classifier(fused)                         # (B, num_classes)
