from pathlib import Path

import pandas as pd

from src.pre_weld.crack_predictor import load_pre_weld_model


DATASET_PATH = Path("dataset/WSteel_Preweld_Clean.csv")


DATASET_FEATURE_COLUMNS = [
    "power (W)",
    "welding speed (m/min)",
    "gas flow rate (l/min)",
    "focal position (mm)",
    "angular position (�)",
    "material strenght steel (mm)",
]


MODEL_FEATURE_COLUMNS = [
    "power",
    "welding_speed",
    "gas_flow_rate",
    "focal_position",
    "angular_position",
    "material_strength",
]


TARGET_COLUMN = "cracking in the weld metal (yes/no)"


# These are treated as process-control parameters.
# Material thickness/strength is intentionally excluded
# from recommended changes.
CONTROLLABLE_PARAMETERS = [
    "power (W)",
    "welding speed (m/min)",
    "gas flow rate (l/min)",
    "focal position (mm)",
    "angular position (�)",
]


DISPLAY_NAMES = {
    "power (W)": "Power",
    "welding speed (m/min)": "Welding Speed",
    "gas flow rate (l/min)": "Gas Flow Rate",
    "focal position (mm)": "Focal Position",
    "angular position (�)": "Angular Position",
    "material strenght steel (mm)": (
        "Material Thickness / Strength"
    ),
}


UNITS = {
    "power (W)": "W",
    "welding speed (m/min)": "m/min",
    "gas flow rate (l/min)": "l/min",
    "focal position (mm)": "mm",
    "angular position (�)": "°",
    "material strenght steel (mm)": "mm",
}


DATASET_TO_MODEL = dict(
    zip(
        DATASET_FEATURE_COLUMNS,
        MODEL_FEATURE_COLUMNS,
    )
)


def load_observed_conditions():
    """
    Load the actual welding-condition combinations present
    in the WSteel pre-weld dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Pre-weld dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = (
        DATASET_FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required pre-weld columns are missing: "
            + ", ".join(missing_columns)
        )

    observed = (
        df[DATASET_FEATURE_COLUMNS]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return observed


def load_dataset():
    """
    Load the complete WSteel pre-weld dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Pre-weld dataset not found: {DATASET_PATH}"
        )

    return pd.read_csv(DATASET_PATH)


def _convert_to_model_features(values):
    """
    Convert WSteel CSV feature names into the simplified
    feature names used by the trained Random Forest.
    """

    return {
        DATASET_TO_MODEL[dataset_feature]: float(
            values[dataset_feature]
        )
        for dataset_feature in DATASET_FEATURE_COLUMNS
    }


def _predict_risk(model, values):
    """
    Return the Random Forest class-1 score.

    This is a model score from predict_proba.
    It is NOT claimed to be a calibrated probability.
    """

    model_values = _convert_to_model_features(values)

    input_df = pd.DataFrame(
        [model_values],
        columns=MODEL_FEATURE_COLUMNS,
    )

    probabilities = model.predict_proba(input_df)[0]

    class_probabilities = {
        int(class_id): float(probability)
        for class_id, probability in zip(
            model.classes_,
            probabilities,
        )
    }

    return class_probabilities.get(1, 0.0)


def _observed_condition_statistics(
    dataset,
    candidate_values,
):
    """
    Calculate the experimentally observed crack rate
    and sample count for a candidate condition.
    """

    mask = pd.Series(
        True,
        index=dataset.index,
    )

    for feature in DATASET_FEATURE_COLUMNS:
        mask = mask & (
            dataset[feature].astype(float)
            == float(candidate_values[feature])
        )

    matching_rows = dataset.loc[mask]

    if matching_rows.empty:
        return {
            "observed_crack_rate": None,
            "observed_samples": 0,
        }

    crack_rate = (
        matching_rows[TARGET_COLUMN]
        .astype(int)
        .mean()
    )

    return {
        "observed_crack_rate": float(
            crack_rate
        ),
        "observed_samples": int(
            len(matching_rows)
        ),
    }


def _count_changed_parameters(
    current_values,
    candidate_values,
):
    """
    Count only controllable process parameters that differ.
    """

    changed = 0

    for feature in CONTROLLABLE_PARAMETERS:
        if float(current_values[feature]) != float(
            candidate_values[feature]
        ):
            changed += 1

    return changed


def _get_parameter_changes(
    current_values,
    candidate_values,
):
    """
    Return human-readable process parameter changes.
    """

    changes = []

    for feature in CONTROLLABLE_PARAMETERS:

        current_value = float(
            current_values[feature]
        )

        candidate_value = float(
            candidate_values[feature]
        )

        if current_value != candidate_value:
            changes.append(
                {
                    "parameter": feature,
                    "label": DISPLAY_NAMES[feature],
                    "current": current_value,
                    "recommended": candidate_value,
                    "unit": UNITS[feature],
                }
            )

    return changes


def recommend_parameter_changes(
    power,
    welding_speed,
    gas_flow_rate,
    focal_position,
    angular_position,
    material_strength,
    top_n=3,
):
    """
    Find lower-risk, experimentally observed welding
    conditions.

    Recommendations are restricted to conditions that
    actually occur in the WSteel dataset.

    Material thickness/strength is held constant and is
    never presented as a process adjustment.

    Ranking prioritizes:

    1. Lower experimentally observed crack rate.
    2. Lower model-estimated crack-risk score.
    3. Fewer process-parameter changes.
    """

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    model_data = load_pre_weld_model()
    model = model_data["model"]

    dataset = load_dataset()

    current_values = {
        "power (W)": float(power),
        "welding speed (m/min)": float(
            welding_speed
        ),
        "gas flow rate (l/min)": float(
            gas_flow_rate
        ),
        "focal position (mm)": float(
            focal_position
        ),
        "angular position (�)": float(
            angular_position
        ),
        "material strenght steel (mm)": float(
            material_strength
        ),
    }

    current_model_risk = _predict_risk(
        model,
        current_values,
    )

    current_observed = (
        _observed_condition_statistics(
            dataset,
            current_values,
        )
    )

    observed_conditions = load_observed_conditions()

    candidates = []

    for _, row in observed_conditions.iterrows():

        candidate_values = {
            feature: float(row[feature])
            for feature in DATASET_FEATURE_COLUMNS
        }

        # Keep the same material condition.
        if (
            candidate_values[
                "material strenght steel (mm)"
            ]
            != current_values[
                "material strenght steel (mm)"
            ]
        ):
            continue

        changed_parameters = (
            _count_changed_parameters(
                current_values,
                candidate_values,
            )
        )

        if changed_parameters == 0:
            continue

        model_risk = _predict_risk(
            model,
            candidate_values,
        )

        observed_stats = (
            _observed_condition_statistics(
                dataset,
                candidate_values,
            )
        )

        if (
            observed_stats["observed_crack_rate"]
            is None
        ):
            continue

        # Candidate must have a lower observed crack
        # rate than the current condition.
        current_observed_rate = (
            current_observed[
                "observed_crack_rate"
            ]
        )

        candidate_observed_rate = (
            observed_stats[
                "observed_crack_rate"
            ]
        )

        if current_observed_rate is not None:
            if (
                candidate_observed_rate
                >= current_observed_rate
            ):
                continue

        risk_reduction = (
            current_model_risk - model_risk
        )

        parameter_changes = (
            _get_parameter_changes(
                current_values,
                candidate_values,
            )
        )

        candidates.append(
            {
                "parameters": candidate_values,
                "predicted_risk": model_risk,
                "observed_crack_rate": (
                    candidate_observed_rate
                ),
                "observed_samples": (
                    observed_stats[
                        "observed_samples"
                    ]
                ),
                "risk_reduction": risk_reduction,
                "risk_reduction_percent": (
                    risk_reduction * 100.0
                ),
                "changed_parameter_count": (
                    changed_parameters
                ),
                "change_type": (
                    "Single-parameter change"
                    if changed_parameters == 1
                    else "Multi-parameter observed alternative"
                ),
                "changes": parameter_changes,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["observed_crack_rate"],
            item["predicted_risk"],
            item["changed_parameter_count"],
        )
    )

    recommendations = candidates[:top_n]

    return {
        "current_model_risk": current_model_risk,
        "current_observed_crack_rate": (
            current_observed[
                "observed_crack_rate"
            ]
        ),
        "current_observed_samples": (
            current_observed[
                "observed_samples"
            ]
        ),
        "recommendations": recommendations,
        "candidate_count": len(candidates),
        "observed_condition_count": len(
            observed_conditions
        ),
        "model_status": model_data.get(
            "status",
            "",
        ),
    }