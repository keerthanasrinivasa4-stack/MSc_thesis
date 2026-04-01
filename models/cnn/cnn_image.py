"""
cnn_image.py
============
ResNet50-based image classifier for MMHS150K hate-speech detection.

Architecture
------------
  ResNet50 (ImageNet pretrained)
    → Global Average Pooling (built-in)      (B, 2048)
    → Dropout(0.3)
    → Linear(2048, num_classes)

Supports both:
  - Binary  : 0=NotHate, 1=Hate   (num_classes=2)
  - 6-class : 0=NotHate 1=Racist 2=Sexist 3=Homophobe 4=Religion 5=OtherHate
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers  (mirrors bert_text.py)
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
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
# ImageNet mean / std
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
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class MMHS150KImageDataset(Dataset):
    """
    Loads tweet images + annotation labels from MMHS150K.

    Parameters
    ----------
    ids        : list of tweet IDs for this split
    gt         : parsed MMHS150K_GT.json dict
    img_dir    : path to img_resized/ folder (images named {id}.jpg)
    transform  : torchvision transform to apply
    binary     : if True, collapses 6 classes → 2 (hate / not-hate)
    """

    def __init__(
        self,
        ids: list[str],
        gt: dict,
        img_dir: Path,
        transform=None,
        binary: bool = True,
    ):
        self.img_dir   = Path(img_dir)
        self.transform = transform or EVAL_TRANSFORMS
        self.binary    = binary
        self.samples: list[tuple[str, int]] = []   # (image_path, label)
        self.skipped   = 0

        for tid in ids:
            item = gt.get(tid)
            if item is None:
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
            self.samples.append((str(img_path), y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, y = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Corrupted image → return black tensor
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.transform(img), torch.tensor(y, dtype=torch.long)

    def class_weights(self) -> torch.Tensor:
        counts    = Counter(y for _, y in self.samples)
        n_classes = 2 if self.binary else 6
        total     = len(self.samples)
        w = [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)]
        return torch.tensor(w, dtype=torch.float)

    def sample_weights(self) -> list[float]:
        counts = Counter(y for _, y in self.samples)
        return [1.0 / counts[y] for _, y in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class ResNetImageClassifier(nn.Module):
    """
    ResNet50 backbone with a custom classification head.

    Parameters
    ----------
    num_classes  : output classes (2 for binary, 6 for multiclass)
    dropout      : dropout rate before the final linear layer
    freeze_backbone : if True, only the head is trained (fast baseline)
    pretrained   : use ImageNet pretrained weights
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Replace the final FC layer
        in_features = backbone.fc.in_features   # 2048
        backbone.fc = nn.Identity()             # strip original head

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, 2048)
        return self.head(features)    # (B, num_classes)

    def unfreeze_backbone(self):
        """Call after warm-up to fine-tune the full network."""
        for p in self.backbone.parameters():
            p.requires_grad = True
