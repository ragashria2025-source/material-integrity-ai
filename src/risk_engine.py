def calculate_severity(base, defect, loading, stress, temperature):
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    index = levels.index(base)

    if loading == "Cyclic / Fatigue" and defect in [
        "Surface Crack-SC",
        "Lack of Fusion-LOF",
        "Undercut-UC",
    ]:
        index += 1

    if stress >= 300:
        index += 1

    if temperature >= 400:
        index += 1

    return levels[min(index, 3)]


def preliminary_risk(severity, confidence):
    if confidence < 0.30:
        return "UNCERTAIN"

    return severity