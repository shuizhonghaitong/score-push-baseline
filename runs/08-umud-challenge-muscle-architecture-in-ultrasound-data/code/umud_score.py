import numpy as np
import pandas as pd

def umud_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id="image_id") -> float:
    """Official UMUD Score (lower is better)."""
    tolerances = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
    weights = {"pa_deg": 1.0, "fl_mm": 1.0, "mt_mm": 1.0}
    merged = solution.merge(submission, on=row_id, how="inner", suffixes=("_true", "_pred"))
    assert len(merged) == len(solution), f"missing rows: {len(solution)-len(merged)}"
    primary = secondary = tertiary = 0.0
    for c in ("pa_deg", "fl_mm", "mt_mm"):
        y = merged[f"{c}_true"].to_numpy(float)
        p = merged[f"{c}_pred"].to_numpy(float)
        tau = tolerances[c]
        primary += weights[c] * np.mean(np.abs(p - y)) / tau
        secondary += weights[c] * np.median(np.abs(p - y)) / tau
        tertiary += weights[c] * np.sqrt(np.mean((p - y) ** 2)) / tau
    n = 3.0
    return (primary + 1e-6 * secondary + 1e-9 * tertiary) / n

if __name__ == "__main__":
    import sys
    gt = pd.read_csv(sys.argv[1])
    pred = pd.read_csv(sys.argv[2])
    print("UMUD score:", umud_score(gt, pred))
