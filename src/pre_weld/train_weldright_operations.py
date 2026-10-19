from pathlib import Path
import re

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "WeldRight_Clean.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "pre_weld"
    / "weldright_operations"
)

FEATURES = [
    "Current",
    "Voltage",
    "Flow",
    "Humidity",
    "Temperature",
    "Job Temp",
]

VALID_OPERATIONS = [
    1,
    30,
    40,
    130,
    180,
    240,
]


def clean_binary_target(series):
    return (
        series
        .astype(str)
        .str.strip()
        .ne("No Defect")
        .astype(int)
    )


def normalize_operation(value):
    """
    Reproduce the important operation-number cleanup used
    in the public WeldRight implementation.

    Examples:
        "-"       -> 1
        "180-"    -> 180
        "-240"    -> 240
        "240"     -> 240
        240       -> 240

    Other malformed values containing a numeric operation
    identifier are normalized using the numeric part.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "-":
        return 1

    # Direct numeric conversion first.
    try:
        return abs(int(float(text)))
    except (ValueError, TypeError):
        pass

    # Handle values such as "180-" or "-240".
    match = re.search(
        r"\d+",
        text,
    )

    if match:
        return int(match.group())

    return None


def prepare_operation_column(df):
    """
    Normalize the raw Order Operation No column while
    preserving the six documented WeldRight operations.
    """

    raw_values = (
        df["Order Operation No"]
        .astype(str)
        .str.strip()
    )

    print()
    print(
        "Raw operation values before normalization:"
    )

    print(
        raw_values.value_counts()
        .head(20)
        .to_string()
    )

    df["Order Operation No"] = (
        df["Order Operation No"]
        .apply(normalize_operation)
    )

    df = df.dropna(
        subset=["Order Operation No"]
    ).copy()

    df["Order Operation No"] = (
        df["Order Operation No"]
        .astype(int)
    )

    return df


def prepare_data(df, operation):
    data = df[
        df["Order Operation No"] == operation
    ].copy()

    data = data.dropna(
        subset=FEATURES + ["Defect"]
    )

    data["Defect_Binary"] = clean_binary_target(
        data["Defect"]
    )

    return data


def make_groups(data):
    return (
        data["Production"].astype(str)
        + "_"
        + data["Date"].astype(str)
    )


def balance_training_data(
    train_data,
    ratio=10,
):
    defect_data = train_data[
        train_data["Defect_Binary"] == 1
    ]

    no_defect_data = train_data[
        train_data["Defect_Binary"] == 0
    ]

    if len(defect_data) == 0:
        return train_data

    target_no_defect_count = min(
        len(no_defect_data),
        len(defect_data) * ratio,
    )

    no_defect_sample = no_defect_data.sample(
        n=target_no_defect_count,
        random_state=42,
    )

    balanced = pd.concat(
        [
            no_defect_sample,
            defect_data,
        ],
        ignore_index=True,
    )

    return balanced.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(drop=True)


def train_operation(df, operation):
    print()
    print("=" * 70)
    print(f"OPERATION {operation}")
    print("=" * 70)

    data = prepare_data(
        df,
        operation,
    )

    if data.empty:
        print(
            f"No usable data found for operation {operation}."
        )
        return

    print(
        f"Rows: {len(data)}"
    )

    print(
        "Defect distribution:"
    )

    print(
        data["Defect_Binary"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if data["Defect_Binary"].nunique() < 2:
        print(
            f"Skipping operation {operation}: "
            "only one target class is present."
        )
        return

    groups = make_groups(data)

    unique_group_count = groups.nunique()

    print(
        f"Unique groups: {unique_group_count}"
    )

    if unique_group_count < 2:
        print(
            f"Skipping operation {operation}: "
            "fewer than 2 independent groups are available."
        )
        return

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            data,
            data["Defect_Binary"],
            groups=groups,
        )
    )

    train_data = data.iloc[
        train_idx
    ].copy()

    test_data = data.iloc[
        test_idx
    ].copy()

    if (
        train_data["Defect_Binary"].nunique() < 2
        or test_data["Defect_Binary"].nunique() < 2
    ):
        print(
            f"Skipping operation {operation}: "
            "train/test split does not contain both target classes."
        )
        return

    print(
        f"Training rows before balancing: "
        f"{len(train_data)}"
    )

    print(
        f"Testing rows: "
        f"{len(test_data)}"
    )

    balanced_train = balance_training_data(
        train_data,
        ratio=10,
    )

    print(
        f"Training rows after balancing: "
        f"{len(balanced_train)}"
    )

    X_train = balanced_train[
        FEATURES
    ]

    y_train = balanced_train[
        "Defect_Binary"
    ]

    X_test = test_data[
        FEATURES
    ]

    y_test = test_data[
        "Defect_Binary"
    ]

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    print()
    print(
        f"Operation {operation} F1: "
        f"{f1:.4f}"
    )

    print()

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

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / f"weldright_operation_{operation}.pkl"
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "operation": operation,
            "threshold": 0.50,
            "model_type": "RandomForestClassifier",
        },
        model_path,
    )

    print()
    print(
        f"Saved: {model_path}"
    )

    print()
    print(
        "Feature importance:"
    )

    importance = sorted(
        zip(
            FEATURES,
            model.feature_importances_,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    for feature, value in importance:
        print(
            f"  {feature}: {value:.4f}"
        )


def main():
    print(
        "Loading cleaned WeldRight dataset..."
    )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    print(
        f"Loaded {len(df)} rows."
    )

    print()
    print(
        "Normalizing Order Operation No..."
    )

    original_rows = len(df)

    df = prepare_operation_column(
        df
    )

    print()

    print(
        f"Rows after operation normalization: "
        f"{len(df)}"
    )

    print(
        f"Rows removed during operation normalization: "
        f"{original_rows - len(df)}"
    )

    available_operations = sorted(
        df["Order Operation No"]
        .unique()
        .tolist()
    )

    print()
    print(
        "Available operation numbers after normalization:"
    )

    print(
        available_operations
    )

    print()
    print(
        "Training documented operation numbers:"
    )

    print(
        VALID_OPERATIONS
    )

    for operation in VALID_OPERATIONS:

        if operation not in available_operations:
            print()
            print(
                f"Skipping operation {operation}: "
                "no rows found."
            )
            continue

        train_operation(
            df,
            operation,
        )

    print()
    print("=" * 70)
    print(
        "WeldRight operation-specific training complete."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()