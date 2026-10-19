from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight


DATA_PATH = Path("dataset/WeldRight_Clean.csv")
MODEL_PATH = Path(
    "models/pre_weld/weldright_extratrees_model.pkl"
)

FEATURES = [
    "Current",
    "Voltage",
    "Flow",
    "Humidity",
    "Temperature",
    "Job Temp",
]

RANDOM_STATE = 42
MAJORITY_TO_MINORITY_RATIO = 10


def main():
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

    df["Target"] = (
        df["Defect"] != "No Defect"
    ).astype(int)

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

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=df["Group"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    print()
    print("Original training distribution:")
    print(
        train_df["Target"]
        .value_counts()
        .sort_index()
    )

    train_defect = train_df[
        train_df["Target"] == 1
    ].copy()

    train_no_defect = train_df[
        train_df["Target"] == 0
    ].copy()

    target_majority_size = min(
        len(train_no_defect),
        len(train_defect)
        * MAJORITY_TO_MINORITY_RATIO,
    )

    train_no_defect_sampled = (
        train_no_defect.sample(
            n=target_majority_size,
            random_state=RANDOM_STATE,
        )
    )

    balanced_train = pd.concat(
        [
            train_no_defect_sampled,
            train_defect,
        ],
        ignore_index=True,
    )

    balanced_train = balanced_train.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print()
    print("Balanced training distribution:")
    print(
        balanced_train["Target"]
        .value_counts()
        .sort_index()
    )

    X_train = balanced_train[FEATURES]
    y_train = balanced_train["Target"]

    X_test = test_df[FEATURES]
    y_test = test_df["Target"]

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    model = ExtraTreesClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )

    print()
    print("Training Extra Trees classifier...")

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print()
    print("=" * 60)
    print("WELDRIGHT EXTRA TREES")
    print("=" * 60)

    print()
    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Test rows:     {len(X_test):,}"
    )

    print(
        f"Test defects:  {int(y_test.sum()):,}"
    )

    print()
    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.4f}"
    )

    print()
    print("Classification report @ threshold 0.50:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Defect",
                "Defect",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()
    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    package = {
        "model": model,
        "features": FEATURES,
        "classes": [
            "No Defect",
            "Defect",
        ],
        "threshold": 0.50,
        "training_method": (
            "Extra Trees with 10:1 "
            "majority downsampling"
        ),
        "random_state": RANDOM_STATE,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        package,
        MODEL_PATH,
    )

    print()
    print(
        f"Experimental model saved to: "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()