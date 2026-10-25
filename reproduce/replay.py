"""Submission reconstruction; no network, submission, or base-model training."""
import csv
import json
import math
from pathlib import Path
from .common import now, sha256


def rebuild_housing(run, output):
    """Recover the saved ridge and seen-ZIP correction; never train base models."""
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge

    run, output = Path(run).resolve(), Path(output)
    names = ["lgb63", "lgb31", "lgb127", "lgb63nc", "xgb7", "xgb5"]
    relative = ["data/train.csv", "data/test.csv", "models/oof_stack/oof.json",
                "models/oof_stack/summary.json", "models/oof_stack/test_3seed.npz.npy",
                "models/oof_stack/blend_3seed_fixed.npy", "subs/v4b_stack.csv"]
    paths = [run / name for name in relative]
    train, test = [pd.read_csv(path, dtype={"Zip": str}) for path in paths[:2]]
    if test["Zip"].duplicated().any() or test["Zip"].isna().any():
        raise ValueError("Missing or duplicate test ZIP identity")
    month = pd.to_datetime(train.Month).dt.month
    may = train.loc[month == 5].set_index("Zip").AffordabilityPercentageTrue
    june = train.loc[month == 6].set_index("Zip").AffordabilityPercentageTrue
    if may.index.has_duplicates or june.index.has_duplicates:
        raise ValueError("Duplicate training ZIP/month identity")
    oof, summary = [json.loads(path.read_text()) for path in paths[2:4]]
    ids = np.sort(june.index.unique())
    matrix = np.column_stack([[oof[name].get(z, np.nan) for z in ids] for name in names])
    matrix = SimpleImputer().fit_transform(matrix)
    target = june.loc[ids].to_numpy()
    if matrix.shape != (len(ids), 6) or not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("Invalid OOF matrix or labels")
    ridge = Ridge(alpha=summary["alpha"], fit_intercept=True).fit(matrix, target)
    saved_sum = np.load(paths[4], allow_pickle=False)
    if saved_sum.shape != (len(test), 6) or not np.isfinite(saved_sum).all():
        raise ValueError("Test predictions do not match complete test row count")
    blend = saved_sum / 3 @ ridge.coef_ + ridge.intercept_
    saved_blend = np.load(paths[5], allow_pickle=False)
    if saved_blend.shape != blend.shape or not np.isfinite(saved_blend).all():
        raise ValueError("Invalid saved corrected blend")
    delta = float(np.max(np.abs(blend - saved_blend)))
    if delta > 1e-10:
        raise ValueError(f"Recovered stack differs from saved corrected blend: {delta}")
    shared = may.index.intersection(june.index)
    a, b = may.loc[shared].to_numpy(), june.loc[shared].to_numpy()
    # Preserve the original covariance/variance degrees of freedom exactly.
    slope = float(np.cov(a, b)[0, 1] / np.var(a))
    intercept = float(b.mean() - slope * a.mean())
    seen = test.Zip.isin(june.index).to_numpy()
    blend[seen] = slope * june.loc[test.loc[seen, "Zip"]].to_numpy() + intercept
    result = test[["Zip", "State"]].copy()
    result["AffordabilityPercentageTrue"] = np.clip(blend, 0, 100)
    result["Zip"] = result.Zip.str.zfill(5)
    expected = pd.read_csv(paths[6], dtype={"Zip": str})
    if result.shape != expected.shape or not result[["Zip", "State"]].equals(expected[["Zip", "State"]]):
        raise ValueError("Saved submission IDs/states do not match test order")
    if not np.isfinite(blend).all() or not np.allclose(result.iloc[:, 2], expected.iloc[:, 2], rtol=0, atol=1e-10):
        raise ValueError("Rebuilt housing predictions differ from saved submission")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    matched = sha256(output) == sha256(paths[6])
    return {"checked_at": now(), "kind": "submission_reconstruction", "passed": matched,
            "run": str(run), "output": str(output), "rows": len(result),
            "inputs": [{"path": str(path), "sha256": sha256(path)} for path in paths[:-1]],
            "saved_best_submission": str(paths[6]), "sha256": sha256(output),
            "saved_sha256": sha256(paths[6]), "byte_identical": matched,
            "variants": names, "recovered_ridge_coefficients": ridge.coef_.tolist(),
            "recovered_ridge_intercept": float(ridge.intercept_), "oof_rows": len(ids),
            "oof_rmse": float(np.sqrt(np.mean((ridge.predict(matrix) - target) ** 2))),
            "saved_blend_max_absolute_error": delta,
            "seen_zip_rows": int(seen.sum()), "drift_slope": slope, "drift_intercept": intercept,
            "historical_public_score": 6.98, "historical_submission_id": 57478451,
            "training_rerun": False, "stack_refit_from_saved_oof": True,
            "official_score_reproduced": False, "independent_holdout_proven": False,
            "limitations": ["Reconstructs submission from retained base-model predictions; does not retrain base models",
                            "OOF labels were used to tune the stack; this is not an independent final test",
                            "Test prediction row identities are positional in the legacy NPY; full base-model provenance still needs checking"]}


def rebuild_eeg(run, output):
    """Reconstruct the recorded v5 submission without invoking its submit script."""
    run, output = Path(run).resolve(), Path(output)
    paths = [run / "submissions/nb_v3f_out/test_probs.csv",
             run / "submissions/nb_dlcpu_out/test_probs.csv",
             run / "submissions/blend_v5_cpu.csv"]
    tables = []
    for path in paths:
        with path.open(newline="") as stream:
            tables.append(list(csv.DictReader(stream)))
    ml, dl, expected = tables
    if not len(ml) == len(dl) == len(expected) == 360:
        raise ValueError("The saved v5 candidate requires 360 predictions")
    ml_ids = [int(row["ID"]) for row in ml]
    dl_ids = [int(row["trial"]) for row in dl]
    if ml_ids != list(range(360)) or dl_ids != ml_ids:
        raise ValueError("Saved ML/DL row IDs do not align; refusing positional truncation")
    groups = {}
    for index, (m, d) in enumerate(zip(ml, dl)):
        probability = .45 * float(m["P"]) + .55 * float(d["P"])
        if not math.isfinite(probability):
            raise ValueError("Non-finite prediction")
        groups.setdefault(m["file"], []).append((index, probability))
    targets = ["rest"] * 360
    for group in groups.values():
        ranked = sorted(group, key=lambda row: -row[1])
        half = len(ranked) // 2
        if not half or ranked[half - 1][1] == ranked[half][1]:
            raise ValueError("Empty group or ambiguous threshold tie")
        for index, _ in ranked[:half]:
            targets[index] = "move"
    rebuilt = [{"ID": str(index), "TARGET": target} for index, target in enumerate(targets)]
    if rebuilt != expected:
        raise ValueError("Rebuilt v5 predictions differ from the saved best submission")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ID", "TARGET"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rebuilt)
    matched = sha256(output) == sha256(paths[2])
    return {"checked_at": now(), "kind": "submission_reconstruction", "passed": matched,
            "run": str(run), "output": str(output), "rows": 360,
            "saved_best_submission": str(paths[2]), "sha256": sha256(output),
            "saved_sha256": sha256(paths[2]), "byte_identical": matched,
            "inputs": [{"path": str(path), "sha256": sha256(path)} for path in paths[:2]],
            "historical_public_score": .76, "historical_submission_id": 57498703,
            "training_rerun": False, "official_score_reproduced": False,
            "limitations": ["Reconstructs best submitted predictions from retained ML/DL probabilities; does not retrain upstream models",
                            "DL IDs use saved global trial order; hidden test labels are unavailable"]}
