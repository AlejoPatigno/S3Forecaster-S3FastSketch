"""Shock versus non-shock evaluation for both proposed forecasters."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .metrics import evaluate_forecast
from .s3_fastsketch_experiment import evaluate_fastsketch
from .s3_forecaster_experiment import evaluate_s3_forecaster
from .utils import ensure_series, parse_forecast_output


def seasonal_abs_innovations_train(train_series: Any, seasonal_period: int = 12):
    train = ensure_series(train_series)
    y = train.to_numpy(dtype=float)
    m = int(seasonal_period)
    if len(y) <= m:
        m = 1
    if len(y) <= m:
        raise ValueError("The training series is too short for shock detection.")
    return np.abs(y[m:] - y[:-m]), m


def compute_pretest_shock_threshold(
    train_series: Any,
    *,
    seasonal_period: int = 12,
    method: str = "iqr",
    iqr_k: float = 1.5,
    quantile: float = 0.90,
):
    innovations, m = seasonal_abs_innovations_train(train_series, seasonal_period)
    if method == "iqr":
        q1, q3 = np.quantile(innovations, [0.25, 0.75])
        iqr = q3 - q1
        threshold = q3 + iqr_k * iqr
        metadata = {
            "method": method,
            "seasonal_period": m,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "iqr_k": float(iqr_k),
            "threshold": float(threshold),
        }
    elif method == "quantile":
        threshold = np.quantile(innovations, quantile)
        metadata = {
            "method": method,
            "seasonal_period": m,
            "quantile": float(quantile),
            "threshold": float(threshold),
        }
    else:
        raise ValueError("method must be 'iqr' or 'quantile'.")
    return float(threshold), metadata


def label_test_shocks(
    train_series: Any,
    test_series: Any,
    *,
    threshold: float,
    seasonal_period: int = 12,
):
    train = ensure_series(train_series)
    test = ensure_series(test_series)
    y_all = np.concatenate([train.to_numpy(dtype=float), test.to_numpy(dtype=float)])
    m = int(seasonal_period)
    if len(train) <= m:
        m = 1

    innovations = []
    for t in range(len(train), len(y_all)):
        reference = max(0, t - m)
        innovations.append(abs(y_all[t] - y_all[reference]))
    innovations = np.asarray(innovations, dtype=float)
    shock = innovations > float(threshold)
    details = pd.DataFrame(
        {
            "y_true": test.to_numpy(dtype=float),
            "seasonal_innovation": innovations,
            "is_shock": shock.astype(int),
        },
        index=test.index,
    )
    return shock, ~shock, details


def _subset_metrics(
    subset_name: str,
    mask: np.ndarray,
    test: pd.Series,
    forecast: Any,
    train: pd.Series,
    *,
    alpha: float,
    seasonal_period: int,
):
    out = parse_forecast_output(forecast)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return {
            "subset": subset_name,
            "n_points": 0,
            "fraction": float(mask.mean()),
            "status": "empty",
        }
    metrics = evaluate_forecast(
        test.to_numpy()[mask],
        out.pred[mask],
        y_train=train,
        lower=out.lower[mask] if out.lower is not None else None,
        upper=out.upper[mask] if out.upper is not None else None,
        alpha=alpha,
        seasonal_period=seasonal_period,
    )
    return {
        "subset": subset_name,
        "n_points": int(mask.sum()),
        "fraction": float(mask.mean()),
        "status": "ok",
        **metrics,
    }


def analyze_forecast_shocks(
    train_series: Any,
    test_series: Any,
    forecast: Any,
    *,
    model_name: str,
    alpha: float = 0.10,
    seasonal_period: int = 12,
    threshold_method: str = "iqr",
    iqr_k: float = 1.5,
    quantile: float = 0.90,
):
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    out = parse_forecast_output(forecast)
    if len(out.pred) != len(test):
        raise ValueError("Forecast horizon does not match the test horizon.")

    threshold, metadata = compute_pretest_shock_threshold(
        train,
        seasonal_period=seasonal_period,
        method=threshold_method,
        iqr_k=iqr_k,
        quantile=quantile,
    )
    shock, non_shock, details = label_test_shocks(
        train,
        test,
        threshold=threshold,
        seasonal_period=metadata["seasonal_period"],
    )
    rows = []
    for name, mask in (
        ("overall", np.ones(len(test), dtype=bool)),
        ("shock", shock),
        ("non_shock", non_shock),
    ):
        row = _subset_metrics(
            name,
            mask,
            test,
            forecast,
            train,
            alpha=alpha,
            seasonal_period=metadata["seasonal_period"],
        )
        row.update(
            {
                "model": model_name,
                "threshold": threshold,
                "threshold_method": threshold_method,
            }
        )
        rows.append(row)

    details["pred"] = out.pred
    if out.lower is not None:
        details["lower"] = out.lower
        details["upper"] = out.upper
    return {
        "summary": pd.DataFrame(rows),
        "details": details,
        "shock_mask": shock,
        "non_shock_mask": non_shock,
        "threshold_metadata": metadata,
    }


def run_s3_shock_analysis(
    train_series: Any,
    test_series: Any,
    point_params: dict,
    *,
    uq_params: Optional[dict] = None,
    **analysis_kwargs,
):
    evaluation = evaluate_s3_forecaster(
        train_series, test_series, point_params, uq_params=uq_params
    )
    analysis = analyze_forecast_shocks(
        train_series,
        test_series,
        evaluation["forecast"],
        model_name="S3-Forecaster",
        **analysis_kwargs,
    )
    analysis.update(evaluation)
    return analysis


def run_fastsketch_shock_analysis(
    train_series: Any,
    test_series: Any,
    point_params: dict,
    *,
    uq_params: Optional[dict] = None,
    **analysis_kwargs,
):
    evaluation = evaluate_fastsketch(
        train_series, test_series, point_params, uq_params=uq_params
    )
    analysis = analyze_forecast_shocks(
        train_series,
        test_series,
        evaluation["forecast"],
        model_name="S3-FastSketch",
        **analysis_kwargs,
    )
    analysis.update(evaluation)
    return analysis


def compare_s3_models_on_shocks(
    train_series: Any,
    test_series: Any,
    s3_params: dict,
    fastsketch_params: dict,
    *,
    s3_uq: Optional[dict] = None,
    fastsketch_uq: Optional[dict] = None,
    **analysis_kwargs,
):
    s3 = run_s3_shock_analysis(
        train_series,
        test_series,
        s3_params,
        uq_params=s3_uq,
        **analysis_kwargs,
    )
    fast = run_fastsketch_shock_analysis(
        train_series,
        test_series,
        fastsketch_params,
        uq_params=fastsketch_uq,
        **analysis_kwargs,
    )
    return {
        "summary": pd.concat([s3["summary"], fast["summary"]], ignore_index=True),
        "S3-Forecaster": s3,
        "S3-FastSketch": fast,
    }
