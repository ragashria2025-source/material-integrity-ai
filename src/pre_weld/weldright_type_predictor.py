from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("dataset/WeldRight_Clean.csv")
MODEL_PATH = Path(
    "models/pre_weld/weldright_type_model.pkl"
)


FEATURES = [
    "Current",
    "Voltage",
    "Flow",
    "Humidity",
    "Temperature",
    "Job Temp",
]

TARGET = "Defect"

GROUP_COLUMNS = [
    "Production",
    "Order Operation No",
    "Date",
]


DEFECT_CLASSES = [
    "Porosity",
    "Tungsten Inclusion",
]


def load_dataset():
    print("Loading cleaned WeldRight dataset...")

    columns = (
        FEATURES
        + [TARGET]
        + GROUP_COLUMNS
    )

    df = pd.read_csv(
        DATA_PATH,
        usecols=columns,
        low_memory=False,
    )

    for column in FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=columns
    )

    # Keep only actual defect records.
    df = df[
        df[TARGET].isin(DEFECT_CLASSES)
    ].copy()

    return df


def train_model():
    df = load_dataset()

    print(
        f"Defect rows available: {len(df)}"
    )

    print()
    print("Defect type distribution:")
    print(
        df[TARGET].value_counts()
    )

    X = df[FEATURES]

    y = (
        df[TARGET]
        .map(
            {
                "Porosity": 0,
                "Tungsten Inclusion": 1,
            }
        )
        .values
    )

    groups = (
        df["Production"].astype(str)
        + "_"
        + df["Order Operation No"].astype(str)
        + "_"
        + df["Date"].astype(str)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_index, test_index = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y[train_index]
    y_test = y[test_index]

    print()
    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    print()
    print("Training defect-type Random Forest...")

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    print()
    print("Classification report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=DEFECT_CLASSES,
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

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    package = {
        "model": model,
        "features": FEATURES,
        "classes": DEFECT_CLASSES,
    }

    joblib.dump(
        package,
        MODEL_PATH,
    )

    print()
    print(
        f"Model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    train_model()