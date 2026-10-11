# Material Integrity AI — Model Evaluation

## 1. Project Scope

Material Integrity AI is an AI-assisted weld integrity decision-support system.

The current implementation contains:

- Post-weld visual weld-defect detection using YOLOv8n.
- An experimental pre-weld crack-prediction model using welding-process parameters.
- A preliminary rule-based engineering severity/risk layer.

The pre-weld model is currently considered experimental and is not treated as a validated engineering failure-prediction system.

---

## 2. Pre-Weld Dataset

Dataset:

`WSteel_Dataset.csv`

Cleaned dataset:

`WSteel_Preweld_Clean.csv`

Dataset size:

- 360 samples
- 51 crack cases
- 309 non-crack cases

### Features used

The current model uses six input parameters:

1. Power (W)
2. Welding speed (m/min)
3. Gas flow rate (l/min)
4. Focal position (mm)
5. Angular position
6. Material-strength/thickness field

Post-weld outcome fields were excluded from the model to avoid data leakage.

---

## 3. Pre-Weld Model

Model:

**Random Forest Classifier**

Configuration:

- 300 trees
- `random_state = 42`
- `class_weight = "balanced"`

The target is:

`cracking in the weld metal (yes/no)`

---

## 4. Stratified 5-Fold Cross-Validation

The model was evaluated using 5-fold stratified cross-validation.

### Mean results

| Metric | Mean ± Standard Deviation |
|---|---:|
| Accuracy | 91.94% ± 3.45% |
| Precision | 65.19% ± 9.60% |
| Recall | 100.00% ± 0.00% |
| F1 Score | 78.50% ± 7.26% |

These results indicate good performance under conventional stratified cross-validation on the available dataset.

---

## 5. Random 80/20 Stratified Evaluation

A separate stratified 80/20 train/test split produced:

| Metric | Result |
|---|---:|
| Accuracy | 90.28% |
| Precision | 58.82% |
| Recall | 100.00% |
| F1 Score | 74.07% |

Confusion matrix:

```text
[[55, 7],
 [ 0,10]]