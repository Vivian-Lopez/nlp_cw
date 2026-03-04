# BestModel – PCL Classification

This folder contains the final model used for Patronising and Condescending Language (PCL) detection (SemEval 2022 Task 4, Subtask 1).

## Model Overview

The model fine-tunes **RoBERTa-base** for binary classification (PCL vs No PCL) using imbalance-aware training.

### Final Dev F1 (Positive Class)
**0.597**

This outperforms the baseline RoBERTa-base model (≈0.48 F1).

---

## Key Improvements Over Baseline

- Class-weighted cross-entropy loss to address severe imbalance (~9.5:1 No PCL:PCL)
- Minority-class oversampling (≈2.5×)
- Threshold optimisation (sweeping 0.1–0.9 to maximise dev F1)
- Model checkpoint selection based on dev F1

---

## Training Configuration

| Setting | Value |
|----------|--------|
| Base Model | roberta-base |
| Max Length | 128 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Optimiser | AdamW |
| Oversampling | 2.5× (PCL class) |
| Loss | Weighted Cross-Entropy |

---

## Repository Contents

- `train.py` – Training and prediction script
- `dev.txt` – Dev set predictions (one label per line)
- `test.txt` – Test set predictions (one label per line)
- `requirements.txt` – Required Python packages

---

## How to Run

From inside this folder:

```bash
pip install -r requirements.txt
python train.py