"""Canonical long-format result store utilities."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import evaluate_forecast
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
    runtime_fit: float = np.nan,
    runtime_predict: float | None = None,
    status: str = "ok",
    error: str | None = None,
) -> pd.DataFrame:
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
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
            "prior_params_json": json_dumps_stable(prior_params),
            "model_params_json": json_dumps_stable(model_params),
            "uq_params_json": json_dumps_stable(uq_params),
            "seed": np.nan if seed is None else int(seed),
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
        history = group["y_true"].to_numpy(dtype=float)
        metrics = evaluate_forecast(
            group["y_true"],
            group["y_pred"],
            y_train=history,
            lower=group["lower"],
            upper=group["upper"],
            seasonal_period=seasonal_period,
            alpha=alpha,
        )
        rows.append({"dataset": dataset, "series_id": series_id, "model": model, **metrics})
    return pd.DataFrame(rows)


def time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, time.perf_counter() - start
