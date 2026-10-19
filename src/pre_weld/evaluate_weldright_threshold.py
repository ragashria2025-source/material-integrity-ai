from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("dataset/WeldRight_Clean.csv")
MODEL_PATH = Path("models/pre_weld/weldright_defect_model.pkl")

FEATURES = [
    "Current",
    "Voltage",
    "Flow",
    "Humidity",
    "Temperature",
    "Job Temp",
]

RANDOM_STATE = 42


def main():
    print("Loading balanced Random Forest model...")

    package = joblib.load(MODEL_PATH)
    model = package["model"]

    print("Loading WeldRight dataset...")

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    df = df[
        FEATURES
        + [
            "Defect",
            "Production",
            "Order Operation No",
            "Date",
        ]
    ].copy()

    for column in FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=FEATURES + ["Defect"]
    )

    # Binary target:
    # 0 = No Defect
    # 1 = Defect
    df["Target"] = (
        df["Defect"] != "No Defect"
    ).astype(int)

    # Recreate the exact same grouping used during training.
    df["Group"] = (
        df["Production"].astype(str)
        + "_"
        + df["Order Operation No"].astype(str)
        + "_"
        + df["Date"].astype(str)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )

    _, test_idx = next(
        splitter.split(
            df,
            groups=df["Group"],
        )
    )

    test_df = df.iloc[test_idx].copy()

    X_test = test_df[FEATURES]
    y_test = test_df["Target"]

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print()
    print("=" * 60)
    print("BALANCED RANDOM FOREST THRESHOLD EVALUATION")
    print("=" * 60)

    print()
    print(
        f"Testing rows: {len(test_df):,}"
    )

    print(
        f"Actual defects: {int(y_test.sum()):,}"
    )

    print()
    print(
        f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}"
    )

    print(
        f"PR-AUC:  {average_precision_score(y_test, probabilities):.4f}"
    )

    print()
    print("Threshold results:")

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        tp = int(
            (
                (y_test == 1)
                & (predictions == 1)
            ).sum()
        )

        fp = int(
            (
                (y_test == 0)
                & (predictions == 1)
            ).sum()
        )

        fn = int(
            (
                (y_test == 1)
                & (predictions == 0)
            ).sum()
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"Precision {precision:.2f} | "
            f"Recall {recall:.2f} | "
            f"F1 {f1:.2f} | "
            f"TP {tp} | "
            f"FP {fp} | "
            f"FN {fn}"
        )


if __name__ == "__main__":
    main()