"""Data-efficiency experiments under progressively restricted training history."""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .ablation_study import run_s3_ablation_study
from .baselines import BASELINE_CLASSES, BASELINE_ALIASES, evaluate_baseline
from .metrics import evaluate_forecast
from .s3_fastsketch_experiment import evaluate_fastsketch
from .s3_forecaster_experiment import evaluate_s3_forecaster
from .utils import count_trainable_parameters, ensure_series, parse_forecast_output


def make_params_feasible(
    model_name: str,
    params: Optional[dict],
    n_history: int,
    *,
    seasonal_period: int = 12,
) -> dict:
    """Clip history-dependent hyperparameters without re-optimizing them."""

    p = copy.deepcopy(params or {})
    max_window = max(2, n_history - 2)
    for key in ("input_window", "window_size", "lags", "ar_lags", "foundation_window"):
        if key in p and p[key] is not None:
            p[key] = int(np.clip(int(p[key]), 1, max_window))

    if model_name == "ARIMA":
        p["p"] = int(min(max(0, int(p.get("p", 1))), max(0, n_history // 4)))
        p["q"] = int(min(max(0, int(p.get("q", 0))), max(0, n_history // 4)))
        p["d"] = int(np.clip(int(p.get("d", 0)), 0, 2))
        if p["p"] == p["q"] == p["d"] == 0:
            p["p"] = 1

    if model_name == "AR":
        p["lags"] = int(min(max(1, int(p.get("lags", 1))), max(1, n_history // 3)))
        if p.get("seasonal", False) and n_history <= p["lags"] + seasonal_period + 2:
            p["seasonal"] = False

    if model_name == "ETS" and p.get("seasonal_periods"):
        if n_history < 2 * int(p["seasonal_periods"]) + 2:
            p["seasonal"] = None
            p["seasonal_periods"] = None

    if model_name in {"S3-Forecaster", "S3Forecaster", "S3-FastSketch", "S3FastSketch"}:
        if "ar_lags" in p:
            p["ar_lags"] = int(min(max(1, p["ar_lags"]), max(1, n_history // 4)))
        if "foundation_window" in p:
            p["foundation_window"] = int(
                min(max(2, p["foundation_window"]), max(2, n_history // 4))
            )
        if "oob_split_ratio" in p:
            p["oob_split_ratio"] = float(np.clip(p["oob_split_ratio"], 0.55, 0.85))
    return p


def _evaluate_external(
    factory: Callable,
    train: pd.Series,
    test: pd.Series,
    params: dict,
    *,
    seasonal_period: int,
):
    start = time.perf_counter()
    model = factory(**params) if isinstance(params, dict) else factory()
    if hasattr(model, "fit"):
        try:
            model.fit(train)
        except TypeError:
            pass
    if hasattr(model, "predict"):
        try:
            forecast = model.predict(len(test))
        except TypeError:
            try:
                forecast = model.predict(steps=len(test))
            except TypeError:
                forecast = model.predict(train, horizon=len(test))
    else:
        forecast = model(train, len(test))
    elapsed = time.perf_counter() - start
    out = parse_forecast_output(forecast)
    metrics = evaluate_forecast(
        test,
        out.pred,
        y_train=train,
        lower=out.lower,
        upper=out.upper,
        seasonal_period=seasonal_period,
        elapsed_seconds=elapsed,
        trainable_params=count_trainable_parameters(model),
    )
    return {"model": model, "forecast": forecast, "metrics": metrics}


def evaluate_named_model(
    model_name: str,
    train_series: Any,
    test_series: Any,
    params: dict,
    *,
    uq_params: Optional[dict] = None,
    external_factories: Optional[dict[str, Callable]] = None,
    seasonal_period: int = 12,
):
    if model_name in {"S3-Forecaster", "S3Forecaster"}:
        return evaluate_s3_forecaster(
            train_series,
            test_series,
            params,
            uq_params=uq_params,
            seasonal_period=seasonal_period,
        )
    if model_name in {"S3-FastSketch", "S3FastSketch"}:
        return evaluate_fastsketch(
            train_series,
            test_series,
            params,
            uq_params=uq_params,
            seasonal_period=seasonal_period,
        )
    if model_name in {"base_only", "base_residual", "base_residual_gate", "full_aci", "full_static"}:
        output = run_s3_ablation_study(
            train_series,
            test_series,
            params,
            modes=[model_name],
            seasonal_period=seasonal_period,
        )
        row = output["results"].iloc[0].to_dict()
        return {
            "model": output["models"][model_name],
            "forecast": output["forecasts"][model_name],
            "metrics": {k: v for k, v in row.items() if isinstance(v, (int, float, np.number))},
        }
    if model_name in BASELINE_CLASSES or model_name in BASELINE_ALIASES:
        return evaluate_baseline(
            model_name,
            train_series,
            test_series,
            params,
            seasonal_period=seasonal_period,
        )
    if external_factories and model_name in external_factories:
        return _evaluate_external(
            external_factories[model_name],
            ensure_series(train_series),
            ensure_series(test_series),
            params,
            seasonal_period=seasonal_period,
        )
    raise ValueError(f"No dispatcher is registered for {model_name!r}.")


def run_data_efficiency_analysis(
    train_series: Any,
    test_series: Any,
    best_params_by_model: dict[str, dict],
    *,
    uq_params_by_model: Optional[dict[str, dict]] = None,
    history_sizes: tuple[int, ...] = (36, 60, 84, 120, 160),
    model_names: Optional[list[str]] = None,
    auto_clip_windows: bool = True,
    external_factories: Optional[dict[str, Callable]] = None,
    seasonal_period: int = 12,
    min_history: int = 18,
):
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    uq_params_by_model = uq_params_by_model or {}
    model_names = model_names or list(best_params_by_model)

    rows, forecasts, models = [], {}, {}
    for model_name in model_names:
        forecasts[model_name], models[model_name] = {}, {}
        for history_size in history_sizes:
            if history_size > len(train) or history_size < min_history:
                continue
            train_subset = train.iloc[-int(history_size) :]
            params = copy.deepcopy(best_params_by_model.get(model_name, {}))
            if auto_clip_windows:
                params = make_params_feasible(
                    model_name,
                    params,
                    len(train_subset),
                    seasonal_period=seasonal_period,
                )
            try:
                output = evaluate_named_model(
                    model_name,
                    train_subset,
                    test,
                    params,
                    uq_params=uq_params_by_model.get(model_name),
                    external_factories=external_factories,
                    seasonal_period=seasonal_period,
                )
                row = {
                    "model_name": model_name,
                    "history_size": int(history_size),
                    "status": "ok",
                    "error": None,
                    "effective_params": params,
                    **output["metrics"],
                }
                forecasts[model_name][history_size] = output["forecast"]
                models[model_name][history_size] = output["model"]
            except Exception as exc:
                row = {
                    "model_name": model_name,
                    "history_size": int(history_size),
                    "status": "failed",
                    "error": str(exc),
                    "effective_params": params,
                }
                forecasts[model_name][history_size] = None
                models[model_name][history_size] = None
            rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["model_name", "history_size"]).reset_index(drop=True)
    return {"summary": summary, "forecasts": forecasts, "models": models}


def make_data_efficiency_pivot(summary: pd.DataFrame, metric: str = "rmse"):
    valid = summary[summary["status"] == "ok"]
    return valid.pivot_table(
        index="model_name", columns="history_size", values=metric, aggfunc="first"
    )


def plot_data_efficiency_curves(
    summary: pd.DataFrame,
    *,
    metric: str = "rmse",
    model_order: Optional[list[str]] = None,
    figsize=(10, 6),
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    valid = summary[summary["status"] == "ok"]
    model_order = model_order or list(valid["model_name"].dropna().unique())
    fig, ax = plt.subplots(figsize=figsize)
    for model_name in model_order:
        subset = valid[valid["model_name"] == model_name].sort_values("history_size")
        if len(subset):
            ax.plot(subset["history_size"], subset[metric], marker="o", label=model_name)
    ax.set_xlabel("Available history (N)")
    ax.set_ylabel(metric.upper())
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax
