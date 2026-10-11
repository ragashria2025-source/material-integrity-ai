from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


DATASET_PATH = Path("dataset/WSteel_Preweld_Clean.csv")
MODEL_PATH = Path("models/pre_weld/pre_weld_crack_model.pkl")


# Stable internal column names.
# These are intentionally independent of the original CSV headers,
# because the angular-position header contains an encoding artifact.
FEATURES = [
    "power",
    "welding_speed",
    "gas_flow_rate",
    "focal_position",
    "angular_position",
    "material_strength",
]

TARGET = "cracking"


def load_pre_weld_dataset():
    """
    Load the cleaned WSteel pre-weld dataset.

    The cleaned CSV has seven columns in this order:
    1. power
    2. welding speed
    3. gas flow rate
    4. focal position
    5. angular position
    6. material strength
    7. cracking target
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Pre-weld dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if len(df.columns) != 7:
        raise ValueError(
            "Unexpected number of columns in the pre-weld dataset. "
            f"Expected 7, found {len(df.columns)}."
        )

    df.columns = FEATURES + [TARGET]

    return df


def train_pre_weld_model():
    """
    Train the experimental pre-weld crack-risk model
    using the cleaned WSteel dataset.

    The model is trained on the full cleaned dataset after
    model evaluation has been performed separately.
    """

    df = load_pre_weld_dataset()

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X, y)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_data = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "model_type": "RandomForestClassifier",
        "dataset": str(DATASET_PATH),
        "training_rows": len(df),
        "positive_samples": int(y.sum()),
        "negative_samples": int((y == 0).sum()),
        "status": (
            "Experimental model. "
            "Random/stratified evaluation was promising, "
            "but generalization to unseen welding conditions "
            "was not validated."
        ),
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model_data, file)

    return model_data


def load_pre_weld_model():
    """
    Load the saved pre-weld model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pre-weld model not found: {MODEL_PATH}. "
            "Train the model first."
        )

    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)

    return model_data


def predict_crack_risk(
    power,
    welding_speed,
    gas_flow_rate,
    focal_position,
    angular_position,
    material_strength,
):
    """
    Predict experimental pre-weld crack risk.

    Returns
    -------
    dict
        Contains predicted class, estimated crack-risk score,
        and model metadata.
    """

    model_data = load_pre_weld_model()

    model = model_data["model"]

    input_data = pd.DataFrame(
        [
            {
                "power": power,
                "welding_speed": welding_speed,
                "gas_flow_rate": gas_flow_rate,
                "focal_position": focal_position,
                "angular_position": angular_position,
                "material_strength": material_strength,
            }
        ]
    )

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]

    class_probabilities = {
        int(class_id): float(probability)
        for class_id, probability in zip(
            model.classes_,
            probabilities,
        )
    }

    crack_risk_score = class_probabilities.get(1, 0.0)

    if prediction == 1:
        predicted_result = "Crack risk detected"
    else:
        predicted_result = "No crack risk detected"

    return {
        "prediction": prediction,
        "result": predicted_result,
        "crack_risk_score": crack_risk_score,
        "model_type": model_data["model_type"],
        "model_status": model_data["status"],
    }