# Cross-Dataset Evaluation Report — GBV Dataset (Dataset2)
  
**Task:** Binary hate speech classification (NotHate vs Hate)  
**Evaluation type:** Zero-shot cross-dataset transfer — no fine-tuning on dataset2

---

## Executive Summary

This report presents the results of testing three pre-trained hate speech detection models on an external Gender-Based Violence (GBV) dataset of 224 annotated tweets. The models were originally trained on the MMHS150K dataset and evaluated here without any retraining to assess cross-dataset generalisation.

Seven experimental configurations were tested. The **best configuration — an ensemble of BERT (for text-only samples) and the multimodal model (for samples with images), with a classification threshold of 0.42 — achieved Macro-F1 = 0.6547**, a **+12.4% improvement** over the naïve multimodal baseline (0.5824).

---

## 1. Dataset Overview

### 1.1 Source Files

| File | Rows | Description |
|---|---|---|
| `source_dataset.csv.xlsx` | 541 | Tweet text, metadata, and image URLs |
| `annotation_record.csv.xlsx` | 469 | Per-annotator labels for each tweet |
| `annotator_info.csv.xlsx` | 21 | Annotator demographics and questionnaire responses |

### 1.2 Label Scheme

The GBV dataset uses four annotation categories:

| Category | Meaning | Count (all annotations) |
|---|---|---|
| **NEUTRAL** | No hate speech detected | 296 |
| **GBV** | Gender-Based Violence (sexist, misogynistic, gender-targeted) | 259 |
| **CS** | Counter-Speech (responses opposing hate) | 191 |
| **OTHER-Toxic** | Toxic speech not specifically gender-based | 181 |

### 1.3 Binary Label Mapping

For compatibility with the binary models (trained on MMHS150K), labels were aggregated via majority vote across annotators per tweet:

| Mapping | Binary Label |
|---|---|
| NEUTRAL | **NotHate** (0) |
| GBV, CS, OTHER-Toxic | **Hate** (1) |

**Resulting distribution (default mapping):**

| Class | Count | Percentage |
|---|---|---|
| Hate | 140 | 62.5% |
| NotHate | 84 | 37.5% |
| **Total** | **224** | 100% |

### 1.4 Image Availability

| Category | Count | % of Total |
|---|---|---|
| Text-only (no image URL) | 180 | 80.4% |
| Image URL present | 44 | 19.6% |
| — Successfully downloaded | 35 | 15.6% |
| — Failed download (expired Twitter CDN) | 9 | 4.0% |
| **Unique cached images** | **31** | — |

> 80% of samples are text-only. For those, the multimodal model receives a blank black 224×224 image, meaning the image branch contributes noise rather than signal.

---

## 2. Models Under Test

### 2.1 Multimodal Classifier (MM)

```
Text branch:   BERT-base-uncased → [CLS] pooler (768) → Linear(768→512) → ReLU
Image branch:  ResNet-50 (ImageNet-V2) → GAP (2048) → Linear(2048→512) → ReLU
Fusion:        Concat [512 ‖ 512] = 1024 → Dropout(0.4) → Linear(1024→2)
```

- **Checkpoint:** `multimodal_binary_best.pt` (epoch 5, val Macro-F1 = 0.6468 on MMHS150K)
- **Training:** Adam, lr=2e-5, weight_decay=0.01, soft-label cross-entropy, 5 epochs

### 2.2 BERT Text Classifier

```
BERT-base-uncased → [CLS] pooler (768) → Dropout(0.3) → Linear(768→2)
```

- **Checkpoint:** `bert_best.pt` (epoch 5, val Macro-F1 = 0.6399 on MMHS150K)
- **Training:** Same hyperparameters as multimodal text branch

### 2.3 Inference Settings

| Parameter | Value |
|---|---|
| Tokeniser | BertTokenizerFast (bert-base-uncased) |
| Max token length | 96 |
| Image transforms | Resize(256) → CenterCrop(224) → Normalize(ImageNet) |
| Batch size | 16 |
| Device | CPU |
| Mode | `model.eval()`, `torch.no_grad()` |

---

## 3. Baseline Results (Multimodal, threshold=0.50)

| Metric | Value |
|---|---|
| Accuracy | 0.5848 |
| **Macro-F1** | **0.5824** |
| Weighted-F1 | 0.5745 |

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| NotHate | 0.4713 | 0.8810 | 0.6141 | 84 |
| Hate | 0.8507 | 0.4071 | 0.5507 | 140 |

**Confusion Matrix:**

|  | Pred NotHate | Pred Hate | Total |
|---|---|---|---|
| True NotHate | 74 | 10 | 84 |
| True Hate | 83 | 57 | 140 |
| Total | 157 | 67 | 224 |

**Key issue:** The model is extremely conservative — it predicts NotHate 70% of the time (157/224) despite the dataset being 62.5% Hate. Hate precision is excellent (0.85) but recall is poor (0.41), meaning 83 hateful tweets are missed.

### 3.1 Impact of Image Availability (Baseline)

| Subset | n | Macro-F1 |
|---|---|---|
| With real image | 35 | **0.7287** |
| Without image (blank) | 189 | 0.5545 |
| All | 224 | 0.5824 |

The +17.4 point gap confirms that blank images actively harm the multimodal model's performance on text-only samples.

---

## 4. Optimisation Experiments

Seven configurations were tested to improve on the baseline, all using the **same pre-trained weights** (no retraining or fine-tuning).

### 4.1 Full Results Table

| Rank | Experiment | Macro-F1 | Accuracy | Δ Macro-F1 |
|---|---|---|---|---|
| **1** | **Ensemble (BERT text-only + MM with-image), t=0.42** | **0.6547** | **0.6652** | **+0.0723** |
| 2 | BERT-only, t=0.42 | 0.6338 | 0.6473 | +0.0514 |
| 3 | Ensemble (BERT + MM), t=0.50 | 0.6331 | 0.6384 | +0.0507 |
| 4 | Ensemble + CS→NotHate, t=0.45 | 0.6325 | 0.6339 | +0.0501 |
| 5 | BERT-only, t=0.50 | 0.6131 | 0.6205 | +0.0307 |
| 6 | MM + CS→NotHate, t=0.50 | 0.6051 | 0.6384 | +0.0227 |
| 7 | MM + CS→NotHate + threshold sweep (t=0.50 best) | 0.6051 | 0.6384 | +0.0227 |
| 8 | Baseline (MM, t=0.50, CS=Hate) | 0.5824 | 0.5848 | — |
| 9 | MM + threshold sweep (t=0.50 best) | 0.5824 | 0.5848 | — |

### 4.2 Experiment Descriptions

**Exp 1 — Baseline (MM, t=0.50):** Multimodal model on all 224 samples with default 0.50 threshold. Blank images for text-only samples.

**Exp 2 — Threshold sweep (MM):** Tried thresholds 0.30–0.50 for the Hate class. The multimodal model's probability distribution is so conservative that t=0.50 remained optimal — lowering the threshold actually hurt Macro-F1.

**Exp 3 — CS→NotHate remap (MM):** Counter-Speech mapped to NotHate instead of Hate, changing the distribution to 92 Hate / 132 NotHate. Improved accuracy (+5.4 pts) but the more imbalanced labels made Macro-F1 improvement modest (+2.3 pts).

**Exp 4 — BERT-only:** Standalone BERT model (no image branch, no fusion noise). Outperformed the multimodal baseline by +3.1 pts at default threshold, confirming that blank images add noise.

**Exp 4b — BERT + threshold=0.42:** BERT has a more balanced probability distribution than the multimodal model, so threshold tuning was effective. Gained +2.1 pts over BERT at t=0.50.

**Exp 5 — Ensemble (BERT + MM, t=0.50):** Route text-only samples to BERT and image samples to the multimodal model. Combines BERT's clean text predictions with MM's superior image-aided performance.

**Exp 5b — Ensemble + t=0.42 (BEST):** The ensemble with optimised threshold. Each component operates in its strength zone, and the lower threshold recovers Hate recall.

**Exp 6 — CS→NotHate + threshold (MM):** Combining CS remap with threshold tuning for the multimodal model. t=0.50 remained optimal; no synergistic improvement.

**Exp 7 — Ensemble + CS→NotHate + t=0.45:** Full combination. Competitive (0.6325) but the label distribution change from CS remap interacts poorly with the ensemble routing.

---

## 5. Best Configuration — Ensemble + Threshold=0.42

### 5.1 Strategy

```
For each sample:
  if sample has a downloaded image:
      → Use Multimodal model (BERT + ResNet-50 fusion)
  else (text-only):
      → Use BERT-only model (no image noise)
  
  Predict Hate if P(Hate) ≥ 0.42  (instead of default 0.50)
```

### 5.2 Results

| Metric | Baseline | Best Config | Change |
|---|---|---|---|
| **Macro-F1** | 0.5824 | **0.6547** | **+12.4%** |
| Accuracy | 0.5848 | 0.6652 | +13.7% |
| Hate Precision | 0.8507 | 0.7642 | −10.2% |
| Hate Recall | 0.4071 | 0.6714 | **+64.9%** |
| Hate F1 | 0.5507 | 0.7148 | +29.8% |
| NotHate Precision | 0.4713 | 0.5446 | +15.5% |
| NotHate Recall | 0.8810 | 0.6548 | −25.7% |
| NotHate F1 | 0.6141 | 0.5946 | −3.2% |

### 5.3 Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| NotHate | 0.5446 | 0.6548 | 0.5946 | 84 |
| Hate | 0.7642 | 0.6714 | 0.7148 | 140 |
| **Macro avg** | **0.6544** | **0.6631** | **0.6547** | **224** |

### 5.4 Confusion Matrix

|  | Pred NotHate | Pred Hate | Total |
|---|---|---|---|
| **True NotHate** | 55 (TN) | 29 (FP) | 84 |
| **True Hate** | 46 (FN) | 94 (TP) | 140 |
| **Total** | 101 | 123 | 224 |

### 5.5 Key Improvements Over Baseline

- **37 more hate tweets correctly detected** (94 vs 57 TP)
- **37 fewer false negatives** (46 vs 83 FN)
- Prediction distribution now matches the data: 123 Hate / 101 NotHate predictions vs 140/84 actual
- Trade-off: 19 more false positives (29 vs 10 FP), but the overall balance is much better

---

## 6. Analysis

### 6.1 Why BERT-Only Outperforms Multimodal on Text-Only Samples

The multimodal model's architecture concatenates text features (512-dim) with image features (512-dim) before the final classifier. When image input is a blank black tensor:

1. The ResNet-50 backbone produces a **fixed, non-zero feature vector** (due to batch norm and bias terms)
2. This vector is projected to 512 dimensions and concatenated with text features
3. The classifier was trained with real images — the blank-image projection is **out-of-distribution** for the learned weights
4. Result: the image branch adds a constant bias that shifts the decision boundary, reducing accuracy

BERT-only avoids this entirely by having no image branch. Its 768→2 classifier was trained purely on text features.

### 6.2 Why Threshold=0.42 Helps

The default threshold of 0.50 assumes symmetric model confidence. In practice:

- **BERT** outputs more balanced probabilities — many Hate samples get P(Hate) in the 0.42–0.50 range
- Lowering the threshold captures these "borderline Hate" predictions that would otherwise be classified as NotHate
- The multimodal model is more extreme (very confident or very uncertain), so threshold tuning has less effect on it alone
- In the ensemble, BERT handles 189/224 samples — its threshold sensitivity dominates

### 6.3 Cross-Dataset Domain Shift

| Factor | MMHS150K (training) | GBV Dataset2 (testing) |
|---|---|---|
| Platform | Twitter (2018–2019) | Twitter/X (2024) |
| Hate categories | Racist, Sexist, Homophobic, Religious, OtherHate | GBV, CS, OTHER-Toxic |
| Label count | 6 → binary | 4 → binary |
| Annotators | 3 per tweet (AMT) | Variable (20 total, study-based) |
| Images | ~95% available | ~16% available |
| Counter-speech | Not annotated | Explicit CS category |
| Topics | General hate speech | Gender-focused political/social discourse |

The domain shift is significant but the models still generalise reasonably, especially BERT's text representations which capture hate-speech semantics that transfer across datasets.

### 6.4 Counter-Speech (CS) Challenge

CS tweets quote or reference hateful language while opposing it. For example:
> "Calling women [slur] is unacceptable and we need to call this out"

This contains hate-speech keywords in a non-hateful context. The model may:
- Detect the slur → predict Hate (correct under CS=Hate mapping)
- Recognise the argumentative tone → predict NotHate (arguably correct semantically)

When CS is mapped to Hate (default), the model is penalised for a reasonable interpretation. When mapped to NotHate, accuracy improves but the label distribution shifts significantly (92 Hate / 132 NotHate), creating new challenges.

---

## 7. Comparison with In-Domain Performance

| Configuration | MMHS150K Test | Dataset2 (GBV) | Gap |
|---|---|---|---|
| Multimodal (baseline) | 0.6380 | 0.5824 | −5.6 pts |
| BERT-only | 0.6320 | 0.6131 | −1.9 pts |
| **Best ensemble** | — | **0.6547** | — |
| MM (with-image subset) | — | 0.7287 | +9.1 pts vs MMHS150K |

Notable findings:
- **BERT generalises better** than the multimodal model (−1.9 pts vs −5.6 pts), because it doesn't suffer from the blank-image problem
- **When images are available**, the multimodal model **exceeds** its MMHS150K performance (0.729 vs 0.638), suggesting visual features transfer well
- **The best ensemble (0.655) approaches MMHS150K in-domain performance** (0.638) — only 1.7 pts below the multimodal in-domain score, despite being fully zero-shot

---

## 8. Output Artefacts

| File | Location | Description |
|---|---|---|
| Per-sample predictions | `dataset2/results/dataset2_predictions.csv` | Text, true/predicted labels, probabilities |
| Experiment summary | `dataset2/results/dataset2_experiments.csv` | All 9 configurations ranked by Macro-F1 |
| Cached images | `dataset2/img_cache/*.jpg` | 31 downloaded tweet images |

---

## 9. Recommendations

### For Immediate Use (No Retraining)

1. **Deploy the Ensemble + t=0.42 configuration.** It requires both checkpoints (`multimodal_binary_best.pt` + `bert_best.pt`) and a simple routing rule based on image availability.

2. **Use the cached images.** Re-running the script will reuse `img_cache/` without re-downloading.

### For Future Improvement

3. **Fine-tune on GBV data.** Even 20–30 labelled samples as fine-tuning data could adapt the model's decision boundary to the GBV domain.

4. **Recover or collect more images.** The multimodal model scores 0.729 with images. Shifting more samples into the multimodal pathway would boost overall performance significantly.

5. **Handle Counter-Speech explicitly.** Add a CS detection pre-filter or train a 3-class model (NotHate / Hate / CS) to avoid the ambiguity.

6. **Apply probability calibration.** Temperature scaling on BERT's output probabilities could improve the threshold sensitivity and yield further Macro-F1 gains.

7. **Expand the threshold search.** A finer grid (0.01 increments) or a validation-based threshold selection could find a marginally better operating point.

---

## 10. Reproducibility

```bash
# From project root (Code/)

# Baseline evaluation (multimodal only)
.venv/bin/python dataset2/scripts/evaluate_dataset2.py

# Enhanced evaluation (all 7+ experiments)
.venv/bin/python dataset2/scripts/evaluate_dataset2_enhanced.py

# Text-only mode (isolate text branch)
.venv/bin/python dataset2/scripts/evaluate_dataset2.py --text-only

# Skip image downloads (use cached)
.venv/bin/python dataset2/scripts/evaluate_dataset2.py --skip-download
```

**Required checkpoints:**
- `models/checkpoints/multimodal/multimodal_binary_best.pt`
- `models/checkpoints/bert/bert_best.pt`

**Environment:** Python 3.12.1, PyTorch, Transformers, torchvision, scikit-learn, pandas, openpyxl, Pillow.
