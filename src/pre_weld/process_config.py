"""
Welding-process configuration for the Material Integrity AI pre-weld module.

This file defines:
- Supported welding-process options
- Process-specific input parameters
- Whether an actual trained ML model is currently available
- Expected prediction type

Important:
Only processes with a trained and validated model should produce an AI
prediction. Other processes are currently UI/configuration placeholders.
"""

PROCESS_CONFIGS = {
    "Laser Welding": {
        "model_available": True,
        "model_type": "RandomForestClassifier",
        "prediction_type": "Crack Risk",
        "description": (
            "Laser welding crack-risk prediction using the WSteel "
            "experimental dataset."
        ),
        "parameters": [
            {
                "key": "power",
                "label": "Laser Power",
                "unit": "W",
                "min": 0.0,
                "max": 5000.0,
                "default": 1200.0,
                "step": 10.0,
            },
            {
                "key": "welding_speed",
                "label": "Welding Speed",
                "unit": "m/min",
                "min": 0.01,
                "max": 10.0,
                "default": 0.8,
                "step": 0.01,
            },
            {
                "key": "gas_flow_rate",
                "label": "Gas Flow Rate",
                "unit": "l/min",
                "min": 0.0,
                "max": 50.0,
                "default": 10.0,
                "step": 0.5,
            },
            {
                "key": "focal_position",
                "label": "Focal Position",
                "unit": "mm",
                "min": -10.0,
                "max": 10.0,
                "default": 0.0,
                "step": 0.1,
            },
            {
                "key": "angular_position",
                "label": "Angular Position",
                "unit": "°",
                "min": -30.0,
                "max": 30.0,
                "default": 15.0,
                "step": 1.0,
            },
            {
                "key": "material_strength",
                "label": "Material Thickness / Strength Field",
                "unit": "mm",
                "min": 0.1,
                "max": 10.0,
                "default": 0.7,
                "step": 0.1,
            },
        ],
    },

    "MIG / GMAW": {
        "model_available": False,
        "model_type": None,
        "prediction_type": "Weld Quality / Defect Risk",
        "description": (
            "MIG/GMAW process configuration. A dedicated trained model "
            "will be connected after suitable data is prepared and evaluated."
        ),
        "parameters": [
            {
                "key": "current",
                "label": "Welding Current",
                "unit": "A",
                "min": 0.0,
                "max": 1000.0,
                "default": 180.0,
                "step": 1.0,
            },
            {
                "key": "voltage",
                "label": "Arc Voltage",
                "unit": "V",
                "min": 0.0,
                "max": 100.0,
                "default": 24.0,
                "step": 0.5,
            },
            {
                "key": "travel_speed",
                "label": "Travel Speed",
                "unit": "mm/s",
                "min": 0.1,
                "max": 100.0,
                "default": 6.0,
                "step": 0.1,
            },
            {
                "key": "wire_feed_speed",
                "label": "Wire Feed Speed",
                "unit": "m/min",
                "min": 0.0,
                "max": 30.0,
                "default": 5.0,
                "step": 0.1,
            },
            {
                "key": "gas_flow",
                "label": "Shielding Gas Flow",
                "unit": "l/min",
                "min": 0.0,
                "max": 50.0,
                "default": 15.0,
                "step": 0.5,
            },
            {
                "key": "torch_angle",
                "label": "Torch Angle",
                "unit": "°",
                "min": 0.0,
                "max": 90.0,
                "default": 45.0,
                "step": 1.0,
            },
        ],
    },

    "TIG / GTAW": {
        "model_available": False,
        "model_type": None,
        "prediction_type": "Weld Quality / Defect Risk",
        "description": (
            "TIG/GTAW process configuration. A dedicated trained model "
            "will be connected after suitable data is prepared and evaluated."
        ),
        "parameters": [
            {
                "key": "current",
                "label": "Welding Current",
                "unit": "A",
                "min": 0.0,
                "max": 500.0,
                "default": 120.0,
                "step": 1.0,
            },
            {
                "key": "voltage",
                "label": "Arc Voltage",
                "unit": "V",
                "min": 0.0,
                "max": 50.0,
                "default": 14.0,
                "step": 0.5,
            },
            {
                "key": "travel_speed",
                "label": "Travel Speed",
                "unit": "mm/s",
                "min": 0.1,
                "max": 50.0,
                "default": 3.0,
                "step": 0.1,
            },
            {
                "key": "gas_flow",
                "label": "Shielding Gas Flow",
                "unit": "l/min",
                "min": 0.0,
                "max": 40.0,
                "default": 10.0,
                "step": 0.5,
            },
            {
                "key": "torch_angle",
                "label": "Torch Angle",
                "unit": "°",
                "min": 0.0,
                "max": 90.0,
                "default": 45.0,
                "step": 1.0,
            },
            {
                "key": "material_thickness",
                "label": "Material Thickness",
                "unit": "mm",
                "min": 0.1,
                "max": 50.0,
                "default": 3.0,
                "step": 0.1,
            },
        ],
    },

    "SMAW": {
        "model_available": False,
        "model_type": None,
        "prediction_type": "Weld Quality / Defect Risk",
        "description": (
            "SMAW process configuration. A dedicated trained model "
            "will be connected after suitable data is prepared and evaluated."
        ),
        "parameters": [
            {
                "key": "current",
                "label": "Welding Current",
                "unit": "A",
                "min": 0.0,
                "max": 500.0,
                "default": 120.0,
                "step": 1.0,
            },
            {
                "key": "voltage",
                "label": "Arc Voltage",
                "unit": "V",
                "min": 0.0,
                "max": 50.0,
                "default": 24.0,
                "step": 0.5,
            },
            {
                "key": "travel_speed",
                "label": "Travel Speed",
                "unit": "mm/s",
                "min": 0.1,
                "max": 50.0,
                "default": 4.0,
                "step": 0.1,
            },
            {
                "key": "electrode_diameter",
                "label": "Electrode Diameter",
                "unit": "mm",
                "min": 1.0,
                "max": 10.0,
                "default": 3.2,
                "step": 0.1,
            },
            {
                "key": "electrode_angle",
                "label": "Electrode Angle",
                "unit": "°",
                "min": 0.0,
                "max": 90.0,
                "default": 45.0,
                "step": 1.0,
            },
            {
                "key": "material_thickness",
                "label": "Plate Thickness",
                "unit": "mm",
                "min": 0.5,
                "max": 100.0,
                "default": 6.0,
                "step": 0.1,
            },
        ],
    },

    "Submerged Arc Welding": {
        "model_available": False,
        "model_type": None,
        "prediction_type": "Weld Quality / Defect Risk",
        "description": (
            "Submerged Arc Welding process configuration. A dedicated "
            "trained model will be connected after suitable data is "
            "prepared and evaluated."
        ),
        "parameters": [
            {
                "key": "current",
                "label": "Welding Current",
                "unit": "A",
                "min": 0.0,
                "max": 2000.0,
                "default": 600.0,
                "step": 10.0,
            },
            {
                "key": "voltage",
                "label": "Arc Voltage",
                "unit": "V",
                "min": 0.0,
                "max": 100.0,
                "default": 32.0,
                "step": 0.5,
            },
            {
                "key": "travel_speed",
                "label": "Travel Speed",
                "unit": "mm/s",
                "min": 0.1,
                "max": 100.0,
                "default": 10.0,
                "step": 0.1,
            },
            {
                "key": "wire_feed_speed",
                "label": "Wire Feed Speed",
                "unit": "m/min",
                "min": 0.0,
                "max": 30.0,
                "default": 8.0,
                "step": 0.1,
            },
            {
                "key": "flux_parameter",
                "label": "Flux Parameter",
                "unit": "relative",
                "min": 0.0,
                "max": 100.0,
                "default": 50.0,
                "step": 1.0,
            },
            {
                "key": "material_thickness",
                "label": "Plate Thickness",
                "unit": "mm",
                "min": 1.0,
                "max": 100.0,
                "default": 12.0,
                "step": 0.5,
            },
        ],
    },

    "Resistance Spot Welding": {
        "model_available": False,
        "model_type": None,
        "prediction_type": "Spot-Weld Quality / Fault Risk",
        "description": (
            "Resistance spot welding process configuration. A dedicated "
            "trained model will be connected after suitable data is "
            "prepared and evaluated."
        ),
        "parameters": [
            {
                "key": "current",
                "label": "Welding Current",
                "unit": "A",
                "min": 0.0,
                "max": 50000.0,
                "default": 10000.0,
                "step": 100.0,
            },
            {
                "key": "weld_time",
                "label": "Weld Time",
                "unit": "ms",
                "min": 1.0,
                "max": 1000.0,
                "default": 200.0,
                "step": 1.0,
            },
            {
                "key": "electrode_force",
                "label": "Electrode Force",
                "unit": "N",
                "min": 0.0,
                "max": 10000.0,
                "default": 2000.0,
                "step": 50.0,
            },
            {
                "key": "voltage",
                "label": "Welding Voltage",
                "unit": "V",
                "min": 0.0,
                "max": 100.0,
                "default": 5.0,
                "step": 0.1,
            },
            {
                "key": "material_thickness",
                "label": "Sheet Thickness",
                "unit": "mm",
                "min": 0.1,
                "max": 10.0,
                "default": 1.0,
                "step": 0.1,
            },
        ],
    },
}


def get_process_names():
    """Return the welding processes available in the interface."""
    return list(PROCESS_CONFIGS.keys())


def get_process_config(process_name):
    """Return configuration for a selected welding process."""
    if process_name not in PROCESS_CONFIGS:
        raise ValueError(
            f"Unsupported welding process: {process_name}"
        )

    return PROCESS_CONFIGS[process_name]