import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

FILE_PATH = "dataset/WSteel_Preweld_Clean.csv"

df = pd.read_csv(FILE_PATH)

features = [
    "power (W)",
    "welding speed (m/min)",
    "gas flow rate (l/min)",
    "focal position (mm)",
    "angular position (�)",
    "material strenght steel (mm)",
]

target = "cracking in the weld metal (yes/no)"

X = df[features]
y = df[target]

print("Dataset loaded.")
print("Samples:", len(df))
print("Cracks:", int(y.sum()))
print("No cracks:", int((y == 0).sum()))

# ---------------------------------------------------------
# STRATIFIED 5-FOLD CROSS-VALIDATION
# ---------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

print("\nStarting 5-fold stratified cross-validation...")

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)

# ---------------------------------------------------------
# DISPLAY FOLD RESULTS
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("5-FOLD STRATIFIED CROSS-VALIDATION RESULTS")
print("=" * 80)

for i in range(5):
    print(f"\nFold {i + 1}")
    print("Accuracy :", round(results["test_accuracy"][i], 4))
    print("Precision:", round(results["test_precision"][i], 4))
    print("Recall   :", round(results["test_recall"][i], 4))
    print("F1 Score :", round(results["test_f1"][i], 4))

# ---------------------------------------------------------
# MEAN AND STANDARD DEVIATION
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("MEAN CROSS-VALIDATION RESULTS")
print("=" * 80)

for metric in ["accuracy", "precision", "recall", "f1"]:
    values = results[f"test_{metric}"]

    print(
        f"{metric.capitalize():10s}: "
        f"{values.mean():.4f} ± {values.std():.4f}"
    )

# ---------------------------------------------------------
# FINAL MODEL FEATURE IMPORTANCE
# ---------------------------------------------------------

print("\nTraining final model on all samples...")

model.fit(X, y)

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n")
print("=" * 80)
print("FINAL MODEL FEATURE IMPORTANCE")
print("=" * 80)

print(importance.to_string(index=False))