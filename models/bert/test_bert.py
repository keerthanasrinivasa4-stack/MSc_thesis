"""
test_bert.py
============
Evaluate the best BERT text classifier checkpoint on the MMHS150K test set.
Loads the Colab-trained checkpoint from models/checkpoints/bert/.

Usage (from repo root):
    python -m models.bert.test_bert                    # binary (default)
    python -m models.bert.test_bert --multiclass       # 6-class
    python -m models.bert.test_bert --checkpoint path/to/custom.pt

Outputs saved to results/bert/<binary|6class>/:
    classification_report.txt
    metrics.json
    confusion_matrix.png
    per_class_f1.png
"""

from __future__ import annotations

import argparse
import json
import sys
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
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import BertTokenizerFast

from models.bert.bert_text import (
    MMHS150KTextDataset,
    BertTextClassifier,
    LABEL_MAP_6,
    LABEL_MAP_BINARY,
    load_split_ids,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT / "data"
GT_PATH    = DATA_DIR / "MMHS150K_GT.json"
SPLIT_DIR  = DATA_DIR / "splits"
CKPT_DIR   = ROOT / "models" / "checkpoints" / "bert"
RESULT_DIR = ROOT / "results" / "bert"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Test BERT text classifier on MMHS150K")
    p.add_argument("--multiclass",  action="store_true", help="6-class instead of binary")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Path to checkpoint file (default: models/checkpoints/bert/bert_best.pt)")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--max_len",     type=int, default=96)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--model_name",  type=str, default="bert-base-uncased")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loader — handles both Colab (dict) and local (raw state_dict) formats
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict:
    """Load model weights from checkpoint, returning any extra metadata."""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        # Try known state-dict keys in order of likelihood
        for key in ("model_state_dict", "model_state", "state_dict"):
            if key in ckpt:
                model.load_state_dict(ckpt[key])
                print(f"  Loaded weights from key '{key}'")
                return ckpt
        # Dict might itself be the state dict (local training format)
        model.load_state_dict(ckpt)
        print("  Loaded weights (raw state dict)")
        return {}
    else:
        model.load_state_dict(ckpt)
        print("  Loaded weights (raw state dict)")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)
        total_loss += loss_fn(logits, labels).item()
        all_preds.extend(torch.argmax(logits, 1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(1, len(loader))
    acc      = accuracy_score(all_labels, all_preds)
    f1_mac   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1_mac, all_labels, all_preds


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, label_names, out_path):
    import seaborn as sns
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(max(6, len(label_names) * 1.5), max(5, len(label_names) * 1.2)))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, linecolor="lightgray", ax=ax,
                cbar_kws={"label": "Normalised proportion"})
    ax.set_title("BERT — Test Confusion Matrix (Row-Normalised)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=30); ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_per_class_f1(report_dict, label_names, out_path):
    f1_scores = [report_dict.get(cls, {}).get("f1-score", 0.0) for cls in label_names]
    colors = ["#4CAF50", "#F44336", "#FF9800", "#9C27B0", "#2196F3", "#FF5722"][:len(label_names)]
    fig, ax = plt.subplots(figsize=(max(7, len(label_names) * 1.5), 5))
    bars = ax.bar(label_names, f1_scores, color=colors, edgecolor="black", width=0.55)
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(np.mean(f1_scores), color="gray", linestyle="--",
               label=f"Macro Avg F1 = {np.mean(f1_scores):.3f}")
    ax.set_title("BERT — Per-Class F1 Score (Test Set)", fontsize=13, fontweight="bold")
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.1)
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    binary      = not args.multiclass
    n_classes   = 2 if binary else 6
    label_map   = LABEL_MAP_BINARY if binary else LABEL_MAP_6
    label_names = [label_map[i] for i in range(n_classes)]
    task_tag    = "binary" if binary else "6class"

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    ckpt_path = Path(args.checkpoint) if args.checkpoint else CKPT_DIR / "bert_best.pt"
    run_dir   = RESULT_DIR / task_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  BERT Text Classifier — Test Evaluation")
    print(f"  Task       : {'Binary (Hate / NotHate)' if binary else '6-Class'}")
    print(f"  Device     : {DEVICE}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Results  → : {run_dir}")
    print("=" * 65)

    # ── Dataset ───────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)

    test_ids  = load_split_ids(SPLIT_DIR / "test_ids.txt")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    test_ds   = MMHS150KTextDataset(test_ids, gt, tokenizer, args.max_len, binary)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"  Test  : {len(test_ds):,}  (skipped {test_ds.skipped})")

    # Class distribution
    counts = Counter(y for _, y in test_ds.samples)
    print("\n  Test class distribution:")
    for i in range(n_classes):
        cnt = counts.get(i, 0)
        print(f"    {label_map[i]:<12}: {cnt:>7,}  ({cnt/len(test_ds)*100:5.1f}%)")

    # ── Model & checkpoint ─────────────────────────────────────────────────
    print(f"\nBuilding model ({args.model_name}) ...")
    model = BertTextClassifier(args.model_name, n_classes, args.dropout).to(DEVICE)

    print(f"Loading checkpoint: {ckpt_path}")
    meta = load_checkpoint(ckpt_path, model, DEVICE)
    if "val_f1" in meta:
        print(f"  Checkpoint val_f1 : {meta['val_f1']:.4f}  (epoch {meta.get('epoch', '?')})")

    # Loss fn (uniform weights for test — no sampling bias)
    loss_fn = nn.CrossEntropyLoss()

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\nRunning test evaluation...")
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, loss_fn, DEVICE)

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
        f.write(f"BERT Text Classifier — {task_tag}\n")
        f.write(f"Checkpoint: {ckpt_path.name}\n")
        if "val_f1" in meta:
            f.write(f"Best val_f1: {meta['val_f1']:.4f}  (epoch {meta.get('epoch', '?')})\n")
        f.write("\n" + report_str)
        f.write(f"\n\nConfusion Matrix:\n{cm}\n")
    print(f"  Saved → {txt_path}")

    metrics = {
        "task": task_tag,
        "model": args.model_name,
        "checkpoint": str(ckpt_path),
        "test": {
            "loss": test_loss,
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

    plot_confusion_matrix(cm, label_names, run_dir / "confusion_matrix.png")
    plot_per_class_f1(report_dict, label_names, run_dir / "per_class_f1.png")

    print("\n" + "=" * 65)
    print(f"  All results saved to: {run_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
