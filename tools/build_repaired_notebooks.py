"""Build compact package-only notebooks for the five benchmark datasets."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOKS = {
    "m4": "s3forecaster-s3fastsketch-m4.ipynb",
    "m3": "s3forecaster-s3fastsketch-m3.ipynb",
    "tourism": "s3forecaster-s3fastsketch-tourism.ipynb",
    "cif": "s3forecaster-s3fastsketch-cif.ipynb",
    "icmd": "s3forecaster-s3fastsketch-icmd.ipynb",
}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def notebook(dataset: str) -> dict:
    title = dataset.upper() if dataset != "tourism" else "Tourism"
    cells = [
        markdown(
            f"""# S3-Forecaster / S3-FastSketch {title}

Package-only repaired notebook. It performs environment setup, deterministic configuration, dataset loading, package calls, result inspection, and export. It intentionally does not redeclare models, metrics, HPO, conformal logic, or plotting utilities.
"""
        ),
        code(
            """
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RUN_MODE = os.environ.get("RUN_MODE", "smoke")  # smoke, standard, full
RUN_FOUNDATION_PRIORS = False
RUN_ARKAN_BASELINE = False
DATASET = "__DATASET__"
REPOSITORY_URL = os.environ.get("S3_REPOSITORY_URL", "https://github.com/alejopatio/S3Forecaster-S3FastSketch.git")
REPOSITORY_REF = os.environ.get("S3_REPOSITORY_REF", "HEAD")
RUN_ID = os.environ.get("RUN_ID", f"{DATASET}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}")
OUTPUT_DIR = Path("/kaggle/working/outputs") / DATASET / RUN_ID if Path("/kaggle/working").exists() else Path("outputs") / DATASET / RUN_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
""".replace("__DATASET__", dataset)
        ),
        code(
            """
repo_root = Path.cwd()
if Path("/kaggle/working").exists() and not (repo_root / "s3paper").exists():
    repo_root = Path("/kaggle/working/S3Forecaster-S3FastSketch")
    if not repo_root.exists():
        subprocess.run(["git", "clone", REPOSITORY_URL, str(repo_root)], check=True)
    subprocess.run(["git", "checkout", REPOSITORY_REF], cwd=repo_root, check=True)

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repo_root)], check=False)
try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
except Exception:
    git_commit = "unknown"

(OUTPUT_DIR / "git_commit.txt").write_text(git_commit + "\\n", encoding="utf-8")
print({"dataset": DATASET, "run_id": RUN_ID, "git_commit": git_commit, "output_dir": str(OUTPUT_DIR)})
"""
        ),
        code(
            """
from s3paper.baselines import evaluate_baseline
from s3paper.diagnostics import diagnostics_frame
from s3paper.result_store import (
    per_series_metrics_from_store,
    result_rows_from_forecast,
    save_result_store,
)
from s3paper.s3_fastsketch_experiment import evaluate_fastsketch, optimize_fastsketch_point
from s3paper.s3_forecaster_experiment import evaluate_s3_forecaster, optimize_s3_forecaster
from s3paper.shock_analysis import inject_additive_spike
from s3paper.statistical_analysis import friedman_test, holm_posthoc, summarize_metric_by_model

TRIALS_BY_MODE = {"smoke": 1, "standard": 25, "full": 100}
MAX_SERIES_BY_MODE = {"smoke": 1, "standard": 20, "full": None}
N_TRIALS = TRIALS_BY_MODE.get(RUN_MODE, 1)
MAX_SERIES = MAX_SERIES_BY_MODE.get(RUN_MODE)
SEASONAL_PERIOD = 12
ALPHA = 0.10
"""
        ),
        code(
            """
def synthetic_monthly_series(series_id: str, n: int = 96) -> pd.Series:
    rng = np.random.default_rng(abs(hash((DATASET, series_id, SEED))) % (2**32))
    t = np.arange(n)
    values = 20 + 0.05 * t + 2.0 * np.sin(2 * np.pi * t / SEASONAL_PERIOD) + rng.normal(0, 0.25, n)
    return pd.Series(values, index=pd.date_range("2015-01-31", periods=n, freq="ME"), name=series_id)


def load_dataset_series() -> dict[str, pd.Series]:
    data_path = os.environ.get(f"{DATASET.upper()}_CSV") or os.environ.get("DATASET_CSV")
    if data_path and Path(data_path).exists():
        frame = pd.read_csv(data_path)
        id_col = next((c for c in ["series_id", "unique_id", "id"] if c in frame.columns), None)
        date_col = next((c for c in ["ds", "date", "timestamp"] if c in frame.columns), None)
        value_col = next((c for c in ["y", "value", "target"] if c in frame.columns), None)
        if id_col and date_col and value_col:
            out = {}
            for sid, group in frame.groupby(id_col):
                s = pd.Series(group[value_col].to_numpy(dtype=float), index=pd.to_datetime(group[date_col]), name=str(sid)).sort_index()
                if len(s) >= 36:
                    out[str(sid)] = s
            if out:
                return out
        numeric = frame.select_dtypes(include=[np.number])
        if not numeric.empty:
            return {str(col): pd.Series(numeric[col].dropna().to_numpy(dtype=float), name=str(col)) for col in numeric.columns}
    return {f"{DATASET}_smoke_1": synthetic_monthly_series(f"{DATASET}_smoke_1")}


series_map = load_dataset_series()
if MAX_SERIES is not None:
    series_map = dict(list(series_map.items())[:MAX_SERIES])

splits = {}
for sid, series in series_map.items():
    horizon = min(12, max(3, len(series) // 5))
    splits[sid] = (series.iloc[:-horizon], series.iloc[-horizon:])

pd.DataFrame({"series_id": list(series_map), "n": [len(s) for s in series_map.values()]}).to_csv(OUTPUT_DIR / "development_series.csv", index=False)
pd.DataFrame({"series_id": list(splits), "horizon": [len(test) for _, test in splits.values()]}).to_csv(OUTPUT_DIR / "evaluation_series.csv", index=False)
diagnostics_frame({sid: train for sid, (train, _) in splits.items()}, dataset=DATASET, seasonal_period=SEASONAL_PERIOD).to_csv(OUTPUT_DIR / "diagnostics.csv", index=False)
"""
        ),
        code(
            """
config = {
    "dataset": DATASET,
    "run_id": RUN_ID,
    "run_mode": RUN_MODE,
    "run_foundation_priors": RUN_FOUNDATION_PRIORS,
    "run_arkan_baseline": RUN_ARKAN_BASELINE,
    "seed": SEED,
    "seasonal_period": SEASONAL_PERIOD,
    "alpha": ALPHA,
    "n_trials": N_TRIALS,
    "repository_ref": REPOSITORY_REF,
    "git_commit": git_commit,
}
environment = {
    "python": sys.version,
    "platform": platform.platform(),
    "cwd": str(Path.cwd()),
}
(OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
(OUTPUT_DIR / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
"""
        ),
        code(
            """
stores = []
metric_rows = []
runtime_rows = []
best_s3_params = {}
best_fast_params = {}

for sid, (train, test) in splits.items():
    print(f"Optimizing and evaluating {sid}...")
    s3_study = optimize_s3_forecaster(train, val_size=min(6, max(3, len(train) // 5)), n_trials=N_TRIALS, seasonal_period=SEASONAL_PERIOD)
    fast_study = optimize_fastsketch_point(train, val_size=min(6, max(3, len(train) // 5)), n_trials=N_TRIALS, seasonal_period=SEASONAL_PERIOD)
    best_s3_params[sid] = dict(s3_study.best_params)
    best_fast_params[sid] = dict(fast_study.best_params)

    s3 = evaluate_s3_forecaster(train, test, best_s3_params[sid], seasonal_period=SEASONAL_PERIOD, alpha=ALPHA)
    fast = evaluate_fastsketch(train, test, best_fast_params[sid], seasonal_period=SEASONAL_PERIOD, alpha=ALPHA)
    naive = evaluate_baseline("Naive", train, test, {}, seasonal_period=SEASONAL_PERIOD, alpha=ALPHA)
    theta = evaluate_baseline("Theta", train, test, {}, seasonal_period=SEASONAL_PERIOD, alpha=ALPHA)

    for model_name, result, params in [
        ("S3-Forecaster", s3, best_s3_params[sid]),
        ("S3-FastSketch", fast, best_fast_params[sid]),
        ("Naive", naive, {}),
        ("Theta", theta, {}),
    ]:
        stores.append(
            result_rows_from_forecast(
                result["forecast"],
                train_series=train,
                test_series=test,
                dataset=DATASET,
                series_id=sid,
                model=model_name,
                run_id=RUN_ID,
                git_commit=git_commit,
                prior_name=params.get("prior_name", ""),
                model_params=params,
                seed=SEED,
                runtime_fit=result["metrics"].get("elapsed_seconds", np.nan),
            )
        )
        metric_rows.append({"dataset": DATASET, "series_id": sid, "model": model_name, **result["metrics"]})
        runtime_rows.append({"dataset": DATASET, "series_id": sid, "model": model_name, "rolling_evaluation_time": result["metrics"].get("elapsed_seconds", np.nan), "trainable_params": result["metrics"].get("trainable_params", np.nan)})

pd.DataFrame(metric_rows).to_parquet(OUTPUT_DIR / "per_series_metrics.parquet", index=False)
pd.DataFrame(runtime_rows).to_csv(OUTPUT_DIR / "runtime_summary.csv", index=False)
(OUTPUT_DIR / "best_s3_params.json").write_text(json.dumps(best_s3_params, indent=2), encoding="utf-8")
(OUTPUT_DIR / "best_fastsketch_params.json").write_text(json.dumps(best_fast_params, indent=2), encoding="utf-8")
"""
        ),
        code(
            """
store = pd.concat(stores, ignore_index=True)
save_result_store(store, OUTPUT_DIR / "per_origin_forecasts.parquet")
per_series = per_series_metrics_from_store(store, seasonal_period=SEASONAL_PERIOD, alpha=ALPHA)
aggregate = summarize_metric_by_model(per_series, metric="mase", n_boot=200)
aggregate.to_csv(OUTPUT_DIR / "aggregate_metrics.csv", index=False)
aggregate[["dataset", "model", "average_rank"]].to_csv(OUTPUT_DIR / "average_ranks.csv", index=False)
pd.DataFrame([friedman_test(per_series, metric="mase")]).to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)
holm_posthoc(per_series, metric="mase").to_csv(OUTPUT_DIR / "holm_posthoc.csv", index=False)
pd.DataFrame({"model": sorted(store["model"].unique()), "status": "evaluated"}).to_csv(OUTPUT_DIR / "model_audit.csv", index=False)

shock_rows = []
for sid, (train, test) in splits.items():
    shock = inject_additive_spike(test, magnitude=float(test.std() or 1.0), start=max(0, len(test) // 3))
    shock_rows.append({"dataset": DATASET, "series_id": sid, "event_start": shock["event_start"], "duration": shock["duration"], "magnitude": shock["magnitude"]})
pd.DataFrame(shock_rows).to_parquet(OUTPUT_DIR / "shock_results.parquet", index=False)

print("Saved artifacts to", OUTPUT_DIR)
display(aggregate)
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    target_dir = Path("notebooks")
    target_dir.mkdir(exist_ok=True)
    for dataset, filename in NOTEBOOKS.items():
        path = target_dir / filename
        path.write_text(json.dumps(notebook(dataset), indent=1) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
