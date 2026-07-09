"""Canonical long-format result store utilities."""

from __future__ import annotations

import json
import hashlib
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import evaluate_forecast, seasonal_naive_scale, seasonal_naive_squared_scale
from .utils import ensure_series


REQUIRED_RESULT_COLUMNS = (
    "run_id",
    "git_commit",
    "dataset",
    "series_id",
    "history_budget",
    "split_id",
    "model",
    "prior_name",
    "prior_params_json",
    "model_params_json",
    "uq_params_json",
    "seed",
    "seasonal_period",
    "requested_coverage",
    "effective_target_coverage",
    "interval_method",
    "mase_scale",
    "rmsse_scale",
    "msis_scale",
    "n_train",
    "train_start",
    "train_end",
    "n_readout",
    "n_adapter_calibration",
    "n_conformal_calibration",
    "prior_status",
    "model_status",
    "config_hash",
    "data_hash",
    "forecast_origin",
    "information_cutoff",
    "target_timestamp",
    "y_true",
    "y_pred",
    "lower",
    "upper",
    "adapter_active",
    "predictability_score",
    "runtime_fit",
    "runtime_predict",
    "status",
    "error",
)


def current_git_commit(cwd: str | Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def json_dumps_stable(value: Any) -> str:
    def default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (set, tuple)):
            return list(obj)
        return str(obj)

    return json.dumps(value or {}, sort_keys=True, default=default)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json_dumps_stable(value).encode("utf-8")).hexdigest()


def result_rows_from_forecast(
    forecast: pd.DataFrame,
    *,
    train_series: Any,
    test_series: Any,
    dataset: str,
    series_id: str,
    model: str,
    run_id: str | None = None,
    git_commit: str | None = None,
    history_budget: int | str | None = None,
    split_id: str = "default",
    prior_name: str | None = None,
    prior_params: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
    uq_params: dict[str, Any] | None = None,
    seed: int | None = None,
    seasonal_period: int = 12,
    requested_coverage: float | None = None,
    effective_target_coverage: float | None = None,
    interval_method: str = "aci",
    prior_status: dict[str, Any] | None = None,
    model_status: dict[str, Any] | None = None,
    runtime_fit: float = np.nan,
    runtime_predict: float | None = None,
    status: str = "ok",
    error: str | None = None,
) -> pd.DataFrame:
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    m = int(seasonal_period)
    mase_scale = seasonal_naive_scale(train, seasonal_period=m)
    rmsse_scale = seasonal_naive_squared_scale(train, seasonal_period=m)
    msis_scale = mase_scale
    model_params_json = json_dumps_stable(model_params)
    prior_params_json = json_dumps_stable(prior_params)
    uq_params_json = json_dumps_stable(uq_params)
    config_hash = stable_hash(
        {
            "model": model,
            "prior_name": prior_name,
            "prior_params": prior_params,
            "model_params": model_params,
            "uq_params": uq_params,
            "seed": seed,
            "seasonal_period": m,
        }
    )
    data_hash = stable_hash(
        {
            "train_index": list(map(str, train.index)),
            "train_values": train.to_numpy(dtype=float),
            "test_index": list(map(str, test.index)),
            "test_values": test.to_numpy(dtype=float),
        }
    )
    frame = forecast.copy()
    if "target" not in frame:
        frame["target"] = test.reindex(frame.index).to_numpy(dtype=float)
    if "target_timestamp" not in frame:
        frame["target_timestamp"] = frame.index
    runtime_predict_values = (
        frame["runtime_predict"].to_numpy(dtype=float)
        if "runtime_predict" in frame
        else np.repeat(np.nan if runtime_predict is None else float(runtime_predict), len(frame))
    )
    adapter_active = frame.get("adapter_active", pd.Series(np.nan, index=frame.index))
    predictability = frame.get("predictability_score", pd.Series(np.nan, index=frame.index))

    rows = pd.DataFrame(
        {
            "run_id": run_id or str(uuid.uuid4()),
            "git_commit": git_commit or current_git_commit(),
            "dataset": dataset,
            "series_id": series_id,
            "history_budget": len(train) if history_budget is None else history_budget,
            "split_id": split_id,
            "model": model,
            "prior_name": prior_name or "",
            "prior_params_json": prior_params_json,
            "model_params_json": model_params_json,
            "uq_params_json": uq_params_json,
            "seed": np.nan if seed is None else int(seed),
            "seasonal_period": m,
            "requested_coverage": np.nan if requested_coverage is None else float(requested_coverage),
            "effective_target_coverage": np.nan if effective_target_coverage is None else float(effective_target_coverage),
            "interval_method": interval_method,
            "mase_scale": mase_scale,
            "rmsse_scale": rmsse_scale,
            "msis_scale": msis_scale,
            "n_train": int(len(train)),
            "train_start": train.index[0] if len(train) else pd.NaT,
            "train_end": train.index[-1] if len(train) else pd.NaT,
            "n_readout": (model_status or {}).get("fit_report", model_status or {}).get("n_readout_samples", np.nan),
            "n_adapter_calibration": (model_status or {}).get("fit_report", model_status or {}).get("n_adapter_calibration", np.nan),
            "n_conformal_calibration": (model_status or {}).get("fit_report", model_status or {}).get("n_conformal_calibration", np.nan),
            "prior_status": json_dumps_stable(prior_status),
            "model_status": json_dumps_stable(model_status),
            "config_hash": config_hash,
            "data_hash": data_hash,
            "forecast_origin": frame.get("forecast_origin", pd.Series(pd.NaT, index=frame.index)).to_numpy(),
            "information_cutoff": frame.get("information_cutoff", pd.Series(pd.NaT, index=frame.index)).to_numpy(),
            "target_timestamp": frame["target_timestamp"].to_numpy(),
            "y_true": frame["target"].to_numpy(dtype=float),
            "y_pred": frame["pred"].to_numpy(dtype=float),
            "lower": frame["lower"].to_numpy(dtype=float) if "lower" in frame else np.nan,
            "upper": frame["upper"].to_numpy(dtype=float) if "upper" in frame else np.nan,
            "adapter_active": adapter_active.to_numpy() if hasattr(adapter_active, "to_numpy") else adapter_active,
            "predictability_score": predictability.to_numpy()
            if hasattr(predictability, "to_numpy")
            else predictability,
            "runtime_fit": float(runtime_fit),
            "runtime_predict": runtime_predict_values,
            "status": status,
            "error": "" if error is None else str(error),
        }
    )
    return validate_result_store(rows)


def validate_result_store(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_RESULT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Canonical result store is missing required columns: {missing}")
    invalid = frame["target_timestamp"] <= frame["information_cutoff"]
    if invalid.any():
        raise AssertionError("Every target_timestamp must be greater than information_cutoff.")
    return frame.loc[:, list(REQUIRED_RESULT_COLUMNS) + [c for c in frame.columns if c not in REQUIRED_RESULT_COLUMNS]]


def save_result_store(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = validate_result_store(frame)
    if target.suffix == ".parquet":
        frame.to_parquet(target, index=False)
    elif target.suffix == ".csv":
        frame.to_csv(target, index=False)
    else:
        raise ValueError("Result store path must end with .parquet or .csv.")
    return target


def per_series_metrics_from_store(
    store: pd.DataFrame,
    *,
    seasonal_period: int = 12,
    alpha: float = 0.10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in store[store["status"] == "ok"].groupby(["dataset", "series_id", "model"], dropna=False):
        dataset, series_id, model = keys
        m = int(group["seasonal_period"].iloc[0]) if "seasonal_period" in group else int(seasonal_period)
        metrics = evaluate_forecast(
            group["y_true"],
            group["y_pred"],
            lower=group["lower"],
            upper=group["upper"],
            seasonal_period=m,
            alpha=alpha,
            mase_scale_value=float(group["mase_scale"].iloc[0]),
            rmsse_scale_value=float(group["rmsse_scale"].iloc[0]),
            msis_scale_value=float(group["msis_scale"].iloc[0]),
        )
        rows.append({"dataset": dataset, "series_id": series_id, "model": model, **metrics})
    return pd.DataFrame(rows)


def assert_unique_metric_source(metrics: pd.DataFrame, *, tolerance: float = 1e-10) -> None:
    required = {"run_id", "dataset", "series_id", "model", "split_id", "metric", "value"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Metric source table missing columns: {sorted(missing)}")
    for keys, group in metrics.groupby(["run_id", "dataset", "series_id", "model", "split_id", "metric"], dropna=False):
        values = group["value"].astype(float).to_numpy()
        values = values[np.isfinite(values)]
        if len(values) > 1 and float(np.max(values) - np.min(values)) > tolerance:
            raise AssertionError(f"Incompatible duplicate metric values for {keys}.")


def export_paper_tables(result_store: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = per_series_metrics_from_store(result_store)
    tables = {
        "paper_table_point_metrics": metrics[
            [c for c in ["dataset", "series_id", "model", "mase", "rmsse", "smape_percent", "rmse", "mae", "wape_percent", "mape_percent", "r2"] if c in metrics]
        ],
        "paper_table_interval_metrics": metrics[
            [c for c in ["dataset", "series_id", "model", "ecp", "coverage_error", "mean_width", "interval_score", "msis"] if c in metrics]
        ],
        "paper_table_runtime": result_store[
            [c for c in ["dataset", "series_id", "model", "runtime_fit", "runtime_predict", "n_readout", "n_adapter_calibration", "n_conformal_calibration"] if c in result_store]
        ],
        "paper_table_diagnostics": pd.DataFrame(),
        "paper_table_ablation": pd.DataFrame(),
        "paper_table_shocks": pd.DataFrame(),
        "paper_table_prior_selection": result_store[
            [c for c in ["dataset", "series_id", "model", "prior_name", "prior_params_json", "prior_status"] if c in result_store]
        ].drop_duplicates(),
        "paper_table_statistical_tests": pd.DataFrame(),
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        csv_path = output / f"{name}.csv"
        tex_path = output / f"{name}.tex"
        table.to_csv(csv_path, index=False)
        table.to_latex(tex_path, index=False)
        paths[name] = csv_path
        paths[f"{name}_latex"] = tex_path
    return paths


def time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, time.perf_counter() - start
