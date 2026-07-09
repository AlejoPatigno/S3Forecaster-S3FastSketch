"""Shock versus non-shock evaluation for both proposed forecasters."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .metrics import evaluate_forecast, mase
from .s3_fastsketch_experiment import evaluate_fastsketch
from .s3_forecaster_experiment import evaluate_s3_forecaster
from .utils import ensure_series, parse_forecast_output


def _event_start(series: pd.Series, start: int | None = None) -> int:
    if start is not None:
        return int(np.clip(start, 0, len(series) - 1))
    return int(max(1, len(series) // 2))


def _shock_result(clean: pd.Series, modified: pd.Series, start: int, duration: int, magnitude: float) -> dict[str, Any]:
    return {
        "series": modified,
        "event_start": clean.index[int(start)],
        "event_start_position": int(start),
        "duration": int(duration),
        "magnitude": float(magnitude),
        "counterfactual": clean,
    }


def inject_additive_spike(series: Any, magnitude: float, *, start: int | None = None):
    clean = ensure_series(series)
    modified = clean.copy()
    pos = _event_start(clean, start)
    modified.iloc[pos] = float(modified.iloc[pos] + magnitude)
    return _shock_result(clean, modified, pos, 1, magnitude)


def inject_temporary_pulse(series: Any, magnitude: float, duration: int = 3, *, start: int | None = None):
    clean = ensure_series(series)
    modified = clean.copy()
    pos = _event_start(clean, start)
    end = min(len(modified), pos + int(duration))
    modified.iloc[pos:end] = modified.iloc[pos:end] + float(magnitude)
    return _shock_result(clean, modified, pos, end - pos, magnitude)


def inject_level_shift(series: Any, magnitude: float, *, start: int | None = None):
    clean = ensure_series(series)
    modified = clean.copy()
    pos = _event_start(clean, start)
    modified.iloc[pos:] = modified.iloc[pos:] + float(magnitude)
    return _shock_result(clean, modified, pos, len(modified) - pos, magnitude)


def inject_variance_shift(series: Any, magnitude: float, *, start: int | None = None, seed: int = 42):
    clean = ensure_series(series)
    modified = clean.copy()
    pos = _event_start(clean, start)
    rng = np.random.default_rng(seed)
    scale = float(abs(magnitude))
    noise = rng.normal(0.0, scale, len(modified) - pos)
    modified.iloc[pos:] = modified.iloc[pos:].to_numpy(dtype=float) + noise
    return _shock_result(clean, modified, pos, len(modified) - pos, magnitude)


def inject_trend_shift(series: Any, magnitude: float, *, start: int | None = None):
    clean = ensure_series(series)
    modified = clean.copy()
    pos = _event_start(clean, start)
    ramp = np.arange(len(modified) - pos, dtype=float)
    modified.iloc[pos:] = modified.iloc[pos:].to_numpy(dtype=float) + float(magnitude) * ramp
    return _shock_result(clean, modified, pos, len(modified) - pos, magnitude)


def inject_seasonal_amplitude_shift(
    series: Any,
    magnitude: float,
    *,
    seasonal_period: int = 12,
    start: int | None = None,
):
    clean = ensure_series(series)
    modified = clean.copy()
    pos = _event_start(clean, start)
    m = max(2, int(seasonal_period))
    t = np.arange(len(modified) - pos, dtype=float)
    seasonal = np.sin(2.0 * np.pi * t / m)
    modified.iloc[pos:] = modified.iloc[pos:].to_numpy(dtype=float) + float(magnitude) * seasonal
    return _shock_result(clean, modified, pos, len(modified) - pos, magnitude)


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


def controlled_shock_metrics(
    train_series: Any,
    test_series: Any,
    forecast: Any,
    *,
    event_start_position: int,
    duration: int,
    seasonal_period: int = 12,
    alpha: float = 0.10,
    recovery_tolerance: float | None = None,
) -> dict[str, Any]:
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    out = parse_forecast_output(forecast)
    y = test.to_numpy(dtype=float)
    pred = out.pred
    errors = np.abs(y - pred)
    start = int(np.clip(event_start_position, 0, len(test)))
    end = int(np.clip(start + int(duration), start, len(test)))
    pre_mask = np.arange(len(test)) < start
    shock_mask = (np.arange(len(test)) >= start) & (np.arange(len(test)) < end)
    post_mask = np.arange(len(test)) >= end

    def subset_mase(mask: np.ndarray) -> float:
        return mase(y[mask], pred[mask], train, seasonal_period=seasonal_period) if mask.any() else np.nan

    tolerance = recovery_tolerance
    if tolerance is None:
        tolerance = float(np.nanmedian(errors[pre_mask])) if pre_mask.any() else float(np.nanmedian(errors))
        if not np.isfinite(tolerance) or tolerance <= 1e-12:
            tolerance = float(np.nanmean(errors) + 1e-12)
    recovery_time = np.nan
    for offset, value in enumerate(errors[end:], start=0):
        if value <= tolerance:
            recovery_time = float(offset)
            break

    lower = out.lower if out.lower is not None else np.repeat(np.nan, len(test))
    upper = out.upper if out.upper is not None else np.repeat(np.nan, len(test))
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    baseline_width = float(np.nanmedian(width[pre_mask])) if pre_mask.any() else float(np.nanmedian(width))
    shock_width = float(np.nanmedian(width[shock_mask])) if shock_mask.any() else np.nan
    frame = forecast.copy() if isinstance(forecast, pd.DataFrame) else pd.DataFrame(index=test.index)
    return {
        "pre_shock_mase": subset_mase(pre_mask),
        "shock_window_mase": subset_mase(shock_mask),
        "post_shock_mase": subset_mase(post_mask),
        "peak_error": float(np.nanmax(errors)) if len(errors) else np.nan,
        "recovery_time": recovery_time,
        "interval_coverage": float(np.nanmean(covered.astype(float))) if len(covered) else np.nan,
        "interval_width_inflation": float(shock_width / baseline_width) if np.isfinite(baseline_width) and baseline_width > 1e-12 else np.nan,
        "gate_trajectory": frame["gate"].tolist() if "gate" in frame else [],
        "adapter_activation": frame["adapter_active"].astype(bool).tolist() if "adapter_active" in frame else [],
        "alpha": float(alpha),
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
