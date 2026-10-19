from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFECT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pre_weld"
    / "weldright_defect_model.pkl"
)

TYPE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pre_weld"
    / "weldright_type_model.pkl"
)

FEATURES = [
    "Current",
    "Voltage",
    "Flow",
    "Humidity",
    "Temperature",
    "Job Temp",
]

_FEATURE_LABELS = {
    "Current": "Current",
    "Voltage": "Voltage",
    "Flow": "Flow",
    "Humidity": "Humidity",
    "Temperature": "Temperature",
    "Job Temp": "Job Temp",
}

_defect_model = None
_type_model = None


def _load_defect_model():
    global _defect_model

    if _defect_model is None:
        _defect_model = joblib.load(
            DEFECT_MODEL_PATH
        )

    return _defect_model


def _load_type_model():
    global _type_model

    if _type_model is None:
        _type_model = joblib.load(
            TYPE_MODEL_PATH
        )

    return _type_model


def _make_input(
    current,
    voltage,
    flow,
    humidity,
    temperature,
    job_temp,
):
    return pd.DataFrame(
        [
            {
                "Current": float(current),
                "Voltage": float(voltage),
                "Flow": float(flow),
                "Humidity": float(humidity),
                "Temperature": float(temperature),
                "Job Temp": float(job_temp),
            }
        ],
        columns=FEATURES,
    )


def _find_defect_class_index(classes):
    classes = list(classes)

    if 1 in classes:
        return classes.index(1)

    if "Defect" in classes:
        return classes.index("Defect")

    if "defect" in classes:
        return classes.index("defect")

    raise ValueError(
        f"Could not identify the defect class in model classes: {classes}"
    )


def predict_weldright(
    current,
    voltage,
    flow,
    humidity,
    temperature,
    job_temp,
    threshold=0.50,
):
    model_input = _make_input(
        current=current,
        voltage=voltage,
        flow=flow,
        humidity=humidity,
        temperature=temperature,
        job_temp=job_temp,
    )

    defect_model = _load_defect_model()

    defect_probabilities = defect_model.predict_proba(
        model_input
    )[0]

    defect_class_index = _find_defect_class_index(
        defect_model.classes_
    )

    defect_score = float(
        defect_probabilities[
            defect_class_index
        ]
    )

    defect_flag = (
        defect_score >= float(threshold)
    )

    defect_type = None
    defect_type_score = None
    defect_type_probabilities = {}

    if defect_flag:

        type_model = _load_type_model()

        type_probabilities = type_model.predict_proba(
            model_input
        )[0]

        type_classes = list(
            type_model.classes_
        )

        best_index = int(
            np.argmax(type_probabilities)
        )

        defect_type = str(
            type_classes[best_index]
        )

        defect_type_score = float(
            type_probabilities[best_index]
        )

        defect_type_probabilities = {
            str(class_name): float(probability)
            for class_name, probability in zip(
                type_classes,
                type_probabilities,
            )
        }

    feature_importance = {}

    if hasattr(
        defect_model,
        "feature_importances_",
    ):

        feature_importance = {
            _FEATURE_LABELS.get(
                feature_name,
                feature_name,
            ): float(importance)
            for feature_name, importance in zip(
                FEATURES,
                defect_model.feature_importances_,
            )
        }

    return {
        "defect_score": defect_score,
        "defect_flag": defect_flag,
        "defect_type": defect_type,
        "defect_type_score": defect_type_score,
        "defect_type_probabilities": defect_type_probabilities,
        "feature_importance": feature_importance,
        "threshold": float(threshold),
        "model_type": type(
            defect_model
        ).__name__,
    }