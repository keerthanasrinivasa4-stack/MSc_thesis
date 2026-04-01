"""
train_cnn.py
============
Full training pipeline for ResNet50 image classifier on MMHS150K.

Usage (from repo root):
    python -m models.cnn.train_cnn                   # full training (5 epochs)
    python -m models.cnn.train_cnn --quick           # sanity-check (2000/500/1000 samples)
    python -m models.cnn.train_cnn --epochs 10       # custom epochs
    python -m models.cnn.train_cnn --multiclass      # 6-way classification
    python -m models.cnn.train_cnn --freeze_epochs 2 # warm up head for 2 epochs first
    python -m models.cnn.train_cnn --help

Outputs (saved to results/cnn/<binary|6class>/):
    metrics.json               train/val/test metrics per epoch
    classification_report.txt  full sklearn report
    training_curves.png        loss & accuracy over epochs
    confusion_matrix.png       test-set confusion matrix
    per_class_f1.png           per-class F1 bar chart (test set)

Checkpoints saved to models/checkpoints/cnn/:
    cnn_<task>_best.pt         best val-F1 checkpoint
    cnn_<task>_last.pt         last epoch checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from models.cnn.cnn_image import (
    MMHS150KImageDataset,
    ResNetImageClassifier,
    LABEL_MAP_6,
    LABEL_MAP_BINARY,
    TRAIN_TRANSFORMS,
    EVAL_TRANSFORMS,
    load_split_ids,
)


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT / "data"
GT_PATH    = DATA_DIR / "MMHS150K_GT.json"
SPLIT_DIR  = DATA_DIR / "splits"
IMG_DIR    = DATA_DIR / "img_resized"
CKPT_DIR   = ROOT / "models" / "checkpoints" / "cnn"
RESULT_DIR = ROOT / "results" / "cnn"


# ─────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet50 image classifier on MMHS150K")
    p.add_argument("--epochs",         type=int,   default=5,     help="Training epochs (default: 5)")
    p.add_argument("--freeze_epochs",  type=int,   default=2,     help="Epochs to train only the head before unfreezing backbone (default: 2)")
    p.add_argument("--batch_size",     type=int,   default=64,    help="Batch size (default: 64)")
    p.add_argument("--lr",             type=float, default=1e-4,  help="Head learning rate (default: 1e-4)")
    p.add_argument("--backbone_lr",    type=float, default=1e-5,  help="Backbone LR after unfreezing (default: 1e-5)")
    p.add_argument("--dropout",        type=float, default=0.3,   help="Dropout rate (default: 0.3)")
    p.add_argument("--multiclass",     action="store_true",       help="6-class instead of binary")
    p.add_argument("--quick",          action="store_true",       help="Limit to 2000/500/1000 samples for sanity check")
    p.add_argument("--no_weighted_sampler", action="store_true",  help="Disable weighted random sampling")
    p.add_argument("--workers",        type=int,   default=2,     help="DataLoader num_workers (default: 2)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += loss_fn(logits, labels).item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc      = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1_macro, all_labels, all_preds


# ─────────────────────────────────────────────────────────────────────────────
# Plotting  (same style as BERT)
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "o-", color="#2196F3", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "o-", color="#F44336", label="Val Loss")
    axes[0].set_title("Loss per Epoch", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(epochs, history["train_acc"], "o-", color="#2196F3", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   "o-", color="#F44336", label="Val Acc")
    axes[1].set_title("Accuracy per Epoch", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.4)

    plt.suptitle("ResNet50 Image Classifier — Training Curves", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_confusion_matrix(cm, label_names: list[str], out_path: Path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(max(6, len(label_names) * 1.5), max(5, len(label_names) * 1.2)))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Oranges",
        xticklabels=label_names, yticklabels=label_names,
        linewidths=0.5, linecolor="lightgray", ax=ax,
        cbar_kws={"label": "Normalised proportion"},
    )
    ax.set_title("ResNet50 — Test Confusion Matrix (Row-Normalised)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=30); ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_per_class_f1(report_dict: dict, label_names: list[str], out_path: Path):
    f1_scores = [report_dict.get(cls, {}).get("f1-score", 0.0) for cls in label_names]
    colors = ["#4CAF50", "#F44336", "#FF9800", "#9C27B0", "#2196F3", "#FF5722"][: len(label_names)]

    fig, ax = plt.subplots(figsize=(max(7, len(label_names) * 1.5), 5))
    bars = ax.bar(label_names, f1_scores, color=colors, edgecolor="black", width=0.55)
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("ResNet50 — Per-Class F1 Score (Test Set)", fontsize=13, fontweight="bold")
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.1)
    ax.axhline(y=np.mean(f1_scores), color="gray", linestyle="--",
               label=f"Macro Avg F1 = {np.mean(f1_scores):.3f}")
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args      = parse_args()
    binary    = not args.multiclass
    n_classes = 2 if binary else 6
    label_map = LABEL_MAP_BINARY if binary else LABEL_MAP_6
    label_names = [label_map[i] for i in range(n_classes)]
    task_tag  = "binary" if binary else "6class"

    DEVICE = torch.device(
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )

    run_dir  = RESULT_DIR / task_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  ResNet50 Image Classifier — MMHS150K")
    print(f"  Task        : {'Binary (Hate / NotHate)' if binary else '6-Class'}")
    print(f"  Device      : {DEVICE}")
    print(f"  Epochs      : {args.epochs}  (freeze_epochs={args.freeze_epochs})")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR (head)   : {args.lr}  |  LR (backbone): {args.backbone_lr}")
    print(f"  Results  →  : {run_dir}")
    print("=" * 65)

    # ── Load ground truth & splits ────────────────────────────────────────
    print("\nLoading dataset...")
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)

    train_ids = load_split_ids(SPLIT_DIR / "train_ids.txt")
    val_ids   = load_split_ids(SPLIT_DIR / "val_ids.txt")
    test_ids  = load_split_ids(SPLIT_DIR / "test_ids.txt")

    train_ds = MMHS150KImageDataset(train_ids, gt, IMG_DIR, TRAIN_TRANSFORMS, binary)
    val_ds   = MMHS150KImageDataset(val_ids,   gt, IMG_DIR, EVAL_TRANSFORMS,  binary)
    test_ds  = MMHS150KImageDataset(test_ids,  gt, IMG_DIR, EVAL_TRANSFORMS,  binary)

    if args.quick:
        print("  [QUICK MODE] Subsampling to 2000/500/1000")
        train_ds.samples = train_ds.samples[:2000]
        val_ds.samples   = val_ds.samples[:500]
        test_ds.samples  = test_ds.samples[:1000]

    print(f"  Train : {len(train_ds):,}  (skipped {train_ds.skipped})")
    print(f"  Val   : {len(val_ds):,}  (skipped {val_ds.skipped})")
    print(f"  Test  : {len(test_ds):,}  (skipped {test_ds.skipped})")

    train_counts = Counter(y for _, y in train_ds.samples)
    print(f"\n  Train class distribution:")
    for i in range(n_classes):
        cnt = train_counts.get(i, 0)
        print(f"    {label_map[i]:<12}: {cnt:>7,}  ({cnt/len(train_ds)*100:5.1f}%)")

    # ── DataLoaders ───────────────────────────────────────────────────────
    nw = 0 if args.quick else args.workers
    if not args.no_weighted_sampler:
        sampler = WeightedRandomSampler(
            train_ds.sample_weights(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=nw, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=nw, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    # Start with frozen backbone (only train head)
    model = ResNetImageClassifier(
        num_classes=n_classes,
        dropout=args.dropout,
        freeze_backbone=True,      # unfreeze after freeze_epochs
        pretrained=True,
    ).to(DEVICE)

    # Optimizer: only head parameters are active at first
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    cw      = train_ds.class_weights().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    print(f"\n  Class weights: {cw.cpu().tolist()}")

    best_val_f1 = -1.0
    best_ckpt   = CKPT_DIR / f"cnn_{task_tag}_best.pt"
    last_ckpt   = CKPT_DIR / f"cnn_{task_tag}_last.pt"
    history     = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    t0_total = time.time()

    for epoch in range(1, args.epochs + 1):

        # Unfreeze backbone after warm-up and rebuild optimizer with 2 param groups
        if epoch == args.freeze_epochs + 1:
            print(f"\n  [Epoch {epoch}] Unfreezing backbone — LR(backbone)={args.backbone_lr}")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": args.backbone_lr},
                {"params": model.head.parameters(),     "lr": args.lr},
            ], weight_decay=1e-4)

        model.train()
        running_loss, n_correct, n_total = 0.0, 0, 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=90)
        for step, (imgs, labels) in enumerate(pbar, 1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            preds     = torch.argmax(logits, dim=1)
            n_correct += (preds == labels).sum().item()
            n_total   += labels.size(0)

            pbar.set_postfix(
                loss=f"{running_loss/step:.4f}",
                acc=f"{n_correct/n_total:.4f}",
            )

        train_loss = running_loss / len(train_loader)
        train_acc  = n_correct / n_total

        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, loss_fn, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - t0
        print(
            f"\n  Epoch {epoch}/{args.epochs} [{elapsed:.0f}s]"
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
            f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ New best model saved  (val_f1={best_val_f1:.4f})")

    total_time = time.time() - t0_total
    torch.save(model.state_dict(), last_ckpt)
    print(f"\n  Total training time: {total_time/60:.1f} min")

    # ── Test ─────────────────────────────────────────────────────────────
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
        model, test_loader, loss_fn, DEVICE
    )

    report_str  = classification_report(y_true, y_pred, target_names=label_names,
                                        digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=label_names,
                                        digits=4, zero_division=0, output_dict=True)
    cm          = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    print("\n" + "=" * 65)
    print("  TEST RESULTS")
    print("=" * 65)
    print(f"  Loss     : {test_loss:.4f}")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  Macro F1 : {test_f1:.4f}")
    print("\n" + report_str)
    print("  Confusion matrix (raw):")
    print(cm)

    # ── Save artefacts ────────────────────────────────────────────────────
    print("\nSaving artefacts...")

    txt_path = run_dir / "classification_report.txt"
    with open(txt_path, "w") as f:
        f.write(f"ResNet50 Image Classifier — {task_tag}\n")
        f.write(f"Device: {DEVICE} | Epochs: {args.epochs} | LR: {args.lr}\n\n")
        f.write(report_str)
        f.write(f"\n\nConfusion Matrix:\n{cm}\n")
    print(f"  Saved → {txt_path}")

    metrics = {
        "task":        task_tag,
        "model":       "resnet50",
        "epochs_ran":  args.epochs,
        "best_val_f1": best_val_f1,
        "history":     history,
        "test": {
            "loss":     test_loss,
            "accuracy": test_acc,
            "macro_f1": test_f1,
            "per_class": {
                lbl: {
                    "precision": report_dict.get(lbl, {}).get("precision", 0),
                    "recall":    report_dict.get(lbl, {}).get("recall",    0),
                    "f1":        report_dict.get(lbl, {}).get("f1-score",  0),
                    "support":   report_dict.get(lbl, {}).get("support",   0),
                }
                for lbl in label_names
            },
        },
    }
    json_path = run_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → {json_path}")

    plot_training_curves(history, run_dir / "training_curves.png")
    plot_confusion_matrix(cm, label_names, run_dir / "confusion_matrix.png")
    plot_per_class_f1(report_dict, label_names, run_dir / "per_class_f1.png")

    print("\n" + "=" * 65)
    print(f"  All results saved to: {run_dir}")
    print(f"  Checkpoints saved to: {CKPT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
