"""Run the historical models in a new work directory, then apply the recorded fix."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .common import now, sha256, write_json


def completed_fits(log_directory):
    names = {"lgb63", "lgb31", "lgb127", "lgb63nc", "xgb7", "xgb5"}
    first = re.findall(r"^(lgb63|lgb31|lgb127|lgb63nc|xgb7|xgb5) fold ([0-4]) done$",
                       (log_directory / "oof_stack.log").read_text(), re.M)
    rest = re.findall(r"^(lgb63|lgb31|lgb127|lgb63nc|xgb7|xgb5) ([12]) ([0-4])$",
                      (log_directory / "stack_full.log").read_text(), re.M)
    expected_first = {(name, str(fold)) for name in names for fold in range(5)}
    expected_rest = {(name, str(seed), str(fold)) for name in names for seed in (1, 2) for fold in range(5)}
    if len(first) != 30 or set(first) != expected_first or len(rest) != 60 or set(rest) != expected_rest:
        raise ValueError("Training logs do not establish all 90 completed variant/seed/fold fits")
    return len(first) + len(rest)


def train(root, output):
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from .cli import verify

    root, output = Path(root).resolve(), Path(output).resolve()
    if not verify(root)["passed"]:
        raise ValueError("Snapshot identity check failed")
    if output.exists():
        raise ValueError("Training requires a fresh output directory")
    if output.is_relative_to(root / "runs") or root.is_relative_to(output):
        raise ValueError("Training output must not replace the preserved source tree")
    run = root / "runs/01-predicting-housing-affordability-in-the-u-s"
    inputs = json.loads((root / "reproduce/required-inputs.json").read_text())["inputs"]
    for row in inputs:
        path = root / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Restore exact input before training: {row['path']}")
    output.mkdir(parents=True)
    for directory in ["data", "external/us-zip-codes-demographics", "models", "subs", "logs"]:
        (output / directory).mkdir(parents=True, exist_ok=True)
    shutil.copytree(run / "scripts", output / "scripts")
    for row in inputs:
        relative = Path(row["path"]).relative_to(run.relative_to(root))
        shutil.copyfile(root / row["path"], output / relative)
    status = {"started_at": now(), "kind": "housing_base_model_retraining",
              "status": "running", "training_rerun": True, "official_submissions": 0,
              "cpu_threads": 16, "expected_model_fits": 90,
              "input_sha256": {r["path"]: r["sha256"] for r in inputs},
              "source_scripts": {name: sha256(run / "scripts" / name)
                                 for name in ["oof_stack.py", "stack_full.py", "pipeline_v2.py"]}}
    report = output / "training-report.json"
    write_json(report, status)
    started = time.monotonic()
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        for module in ["oof_stack", "stack_full"]:
            print(f"Running original scripts.{module}; log: {output / 'logs' / (module + '.log')}", flush=True)
            with (output / "logs" / f"{module}.log").open("w") as stream:
                subprocess.run([sys.executable, "-u", "-m", f"scripts.{module}"], cwd=output,
                               env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        names = ["lgb63", "lgb31", "lgb127", "lgb63nc", "xgb7", "xgb5"]
        status["completed_model_fits"] = completed_fits(output / "logs")
        checkpoint = output / "models/oof_stack"
        tr = pd.read_csv(output / "data/train.csv", dtype={"Zip": str})
        te = pd.read_csv(output / "data/test.csv", dtype={"Zip": str})
        month = pd.to_datetime(tr.Month).dt.month
        june = tr.loc[month == 6].set_index("Zip").AffordabilityPercentageTrue
        may = tr.loc[month == 5].set_index("Zip").AffordabilityPercentageTrue
        oof = json.loads((checkpoint / "oof.json").read_text())
        summary = json.loads((checkpoint / "summary.json").read_text())
        ids = np.sort(june.index.unique())
        matrix = SimpleImputer().fit_transform(np.column_stack([[oof[n].get(z, np.nan) for z in ids] for n in names]))
        ridge = Ridge(alpha=summary["alpha"], fit_intercept=True).fit(matrix, june.loc[ids])
        # The historical v4b correction: original stack_full saves a sum of 3 seeds.
        blend = np.load(checkpoint / "test_3seed.npz.npy", allow_pickle=False) / 3 @ ridge.coef_ + ridge.intercept_
        np.save(checkpoint / "blend_3seed_fixed.npy", blend)
        saved_blend = np.load(run / "models/oof_stack/blend_3seed_fixed.npy", allow_pickle=False)
        shared = may.index.intersection(june.index)
        a, b = may.loc[shared].to_numpy(), june.loc[shared].to_numpy()
        slope = np.cov(a, b)[0, 1] / np.var(a)
        intercept = b.mean() - slope * a.mean()
        seen = te.Zip.isin(june.index).to_numpy()
        prediction = blend.copy()
        prediction[seen] = slope * june.loc[te.loc[seen, "Zip"]].to_numpy() + intercept
        result = te[["Zip", "State"]].copy()
        result["AffordabilityPercentageTrue"] = np.clip(prediction, 0, 100)
        result["Zip"] = result.Zip.str.zfill(5)
        submission = output / "subs/v4b_stack.csv"
        result.to_csv(submission, index=False)
        expected_path = run / "subs/v4b_stack.csv"
        expected = pd.read_csv(expected_path, dtype={"Zip": str})
        if not result[["Zip", "State"]].equals(expected[["Zip", "State"]]) or not np.isfinite(prediction).all():
            raise ValueError("Invalid final row identity or predictions")
        differences = result.iloc[:, 2].to_numpy() - expected.iloc[:, 2].to_numpy()
        same = sha256(submission) == sha256(expected_path)
        status.update({"status": "completed", "passed": same, "rows": len(result),
                       "submission_sha256": sha256(submission), "reference_sha256": sha256(expected_path),
                       "byte_identical_to_historical_best": same,
                       "max_absolute_submission_difference": float(np.abs(differences).max()),
                       "max_absolute_base_blend_difference": float(np.abs(blend - saved_blend).max()),
                       "retrained_stack_oof_rmse": float(np.sqrt(np.mean((ridge.predict(matrix) - june.loc[ids].to_numpy()) ** 2))),
                       "official_score_reproduced": False,
                       "interpretation": "Full CPU training completed; exact-file equality is reported independently from execution success"})
    except BaseException as error:
        status.update({"status": "failed", "passed": False, "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        status.update({"finished_at": now(), "elapsed_seconds": time.monotonic() - started})
        write_json(report, status)
    return status
