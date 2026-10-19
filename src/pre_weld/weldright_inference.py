from pathlib import Path

import joblib
import pandas as pd


BINARY_MODEL_PATH = Path(
    "models/pre_weld/weldright_defect_model.pkl"
)

TYPE_MODEL_PATH = Path(
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

DEFECT_THRESHOLD = 0.25


_binary_package = joblib.load(
    BINARY_MODEL_PATH
)

_type_package = joblib.load(
    TYPE_MODEL_PATH
)

_binary_model = _binary_package["model"]
_type_model = _type_package["model"]


def predict_weldright_defect(
    current,
    voltage,
    flow,
    humidity,
    temperature,
    job_temp,
):
    values = {
        "Current": float(current),
        "Voltage": float(voltage),
        "Flow": float(flow),
        "Humidity": float(humidity),
        "Temperature": float(temperature),
        "Job Temp": float(job_temp),
    }

    X = pd.DataFrame(
        [values],
        columns=FEATURES,
    )

    binary_probability = float(
        _binary_model.predict_proba(X)[0][1]
    )

    if binary_probability < DEFECT_THRESHOLD:
        return {
            "defect": False,
            "defect_label": "No Defect",
            "defect_probability": binary_probability,
            "defect_threshold": DEFECT_THRESHOLD,
            "defect_type": None,
            "defect_type_probability": None,
            "defect_type_probabilities": {},
            "inputs": values,
        }

    type_probabilities = (
        _type_model.predict_proba(X)[0]
    )

    type_classes = _type_package["classes"]

    probability_map = {
        class_name: float(probability)
        for class_name, probability in zip(
            type_classes,
            type_probabilities,
        )
    }

    predicted_index = int(
        type_probabilities.argmax()
    )

    predicted_type = type_classes[
        predicted_index
    ]

    predicted_type_probability = float(
        type_probabilities[predicted_index]
    )

    return {
        "defect": True,
        "defect_label": "Defect",
        "defect_probability": binary_probability,
        "defect_threshold": DEFECT_THRESHOLD,
        "defect_type": predicted_type,
        "defect_type_probability": (
            predicted_type_probability
        ),
        "defect_type_probabilities": probability_map,
        "inputs": values,
    }