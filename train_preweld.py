import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

FILE_PATH = "dataset/WSteel_Preweld_Clean.csv"

df = pd.read_csv(FILE_PATH)

print("Dataset loaded.")
print("Samples:", len(df))


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Create welding-condition groups
# --------------------------------------------------

df["welding_condition"] = (
    df[features]
    .astype(str)
    .agg("_".join, axis=1)
)

groups = df["welding_condition"]


# --------------------------------------------------
# 4. Leave-One-Condition-Out evaluation
# --------------------------------------------------

conditions = groups.unique()

results = []

print("\nStarting evaluation...")
print("Number of welding conditions:", len(conditions))


for i, condition in enumerate(conditions, start=1):

    print(f"\nTesting condition {i}/{len(conditions)}")
    print(condition)

    # Test = one complete welding condition
    test_mask = groups == condition
    train_mask = groups != condition

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    # Create model
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    # zero_division prevents errors when a test condition
    # contains only one class
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "condition": condition,
        "samples": len(y_test),
        "actual_cracks": int(y_test.sum()),
        "predicted_cracks": int(predictions.sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 100)
print("LEAVE-ONE-WELDING-CONDITION-OUT RESULTS")
print("=" * 100)

print(results_df.to_string(index=False))


# --------------------------------------------------
# 6. Overall metrics
# --------------------------------------------------

print("\n")
print("=" * 100)
print("AVERAGE RESULTS")
print("=" * 100)

print(
    "Average accuracy :",
    round(results_df["accuracy"].mean(), 4)
)

print(
    "Average precision:",
    round(results_df["precision"].mean(), 4)
)

print(
    "Average recall   :",
    round(results_df["recall"].mean(), 4)
)

print(
    "Average F1       :",
    round(results_df["f1"].mean(), 4)
)


# --------------------------------------------------
# 7. Train final model on all available data
# --------------------------------------------------

print("\nTraining final model on all 360 samples...")

final_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

final_model.fit(X, y)


# --------------------------------------------------
# 8. Feature importance
# --------------------------------------------------

print("\n")
print("=" * 100)
print("FEATURE IMPORTANCE")
print("=" * 100)

importance = pd.DataFrame({
    "feature": features,
    "importance": final_model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(importance.to_string(index=False))