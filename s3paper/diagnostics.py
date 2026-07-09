"""Training-only diagnostics for benchmark time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

from .utils import ensure_series


def _safe_stat(fn, *args, **kwargs) -> tuple[float, float]:
    try:
        result = fn(*args, **kwargs)
        return float(result[0]), float(result[1])
    except Exception:
        return np.nan, np.nan


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.nanmean(values)) if len(values) else np.nan
    std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return float(std / abs(mean)) if np.isfinite(mean) and abs(mean) > 1e-12 else np.nan


def spectral_entropy(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 3:
        return np.nan
    centered = clean - np.mean(clean)
    power = np.abs(np.fft.rfft(centered)) ** 2
    power = power[1:]
    total = float(np.sum(power))
    if total <= 1e-12:
        return 0.0
    probabilities = power / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return float(entropy / np.log(len(probabilities))) if len(probabilities) > 1 else 0.0


def trend_seasonal_strength(series: pd.Series, seasonal_period: int) -> tuple[float, float]:
    if len(series) < max(8, 2 * seasonal_period):
        return np.nan, np.nan
    try:
        period = max(2, int(seasonal_period))
        fit = STL(series.astype(float), period=period, robust=True).fit()
        remainder = np.asarray(fit.resid, dtype=float)
        trend = np.asarray(fit.trend, dtype=float)
        seasonal = np.asarray(fit.seasonal, dtype=float)
        var_remainder = float(np.nanvar(remainder))
        trend_strength = max(0.0, 1.0 - var_remainder / max(float(np.nanvar(trend + remainder)), 1e-12))
        seasonal_strength = max(0.0, 1.0 - var_remainder / max(float(np.nanvar(seasonal + remainder)), 1e-12))
        return float(trend_strength), float(seasonal_strength)
    except Exception:
        return np.nan, np.nan


def structural_break_count(values: np.ndarray, *, min_segment: int = 8, z_threshold: float = 3.0) -> int:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 2 * min_segment:
        return 0
    breaks = 0
    global_std = float(np.std(clean, ddof=1))
    if global_std <= 1e-12:
        return 0
    last_break = -min_segment
    for idx in range(min_segment, len(clean) - min_segment + 1):
        if idx - last_break < min_segment:
            continue
        left = clean[idx - min_segment : idx]
        right = clean[idx : idx + min_segment]
        z = abs(float(np.mean(right) - np.mean(left))) / (global_std / np.sqrt(min_segment))
        if z >= z_threshold:
            breaks += 1
            last_break = idx
    return int(breaks)


def series_diagnostics(
    series: Any,
    *,
    dataset: str = "",
    series_id: str = "",
    seasonal_period: int = 12,
) -> dict[str, float | int | str]:
    y = ensure_series(series)
    values = y.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    adf_stat, adf_p = _safe_stat(adfuller, finite, autolag="AIC") if len(finite) >= 8 else (np.nan, np.nan)
    kpss_stat, kpss_p = _safe_stat(kpss, finite, regression="c", nlags="auto") if len(finite) >= 8 else (np.nan, np.nan)
    max_lag = max(1, min(int(seasonal_period), len(finite) - 1)) if len(finite) > 1 else 1
    acf_values = acf(finite, nlags=max_lag, fft=False, missing="drop") if len(finite) > 1 else np.asarray([np.nan, np.nan])
    lb_lag = max(1, min(max_lag, len(finite) // 2 - 1)) if len(finite) >= 6 else 1
    try:
        lb = acorr_ljungbox(finite, lags=[lb_lag], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[0])
        lb_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        lb_stat, lb_p = np.nan, np.nan
    trend_strength, seasonal_strength = trend_seasonal_strength(y.dropna(), seasonal_period)
    return {
        "dataset": dataset,
        "series_id": series_id,
        "observations": int(len(values)),
        "missing_fraction": float(np.mean(~np.isfinite(values))) if len(values) else np.nan,
        "zero_fraction": float(np.mean(values == 0.0)) if len(values) else np.nan,
        "coefficient_of_variation": coefficient_of_variation(finite),
        "adf_statistic": adf_stat,
        "adf_p_value": adf_p,
        "kpss_statistic": kpss_stat,
        "kpss_p_value": kpss_p,
        "acf_lag1": float(acf_values[1]) if len(acf_values) > 1 else np.nan,
        "acf_seasonal_lag": float(acf_values[max_lag]) if len(acf_values) > max_lag - 1 else np.nan,
        "ljung_box_statistic": lb_stat,
        "ljung_box_p_value": lb_p,
        "trend_strength": trend_strength,
        "seasonal_strength": seasonal_strength,
        "spectral_entropy": spectral_entropy(finite),
        "structural_break_count": structural_break_count(finite),
    }


def diagnostics_frame(series_map: dict[str, Any], *, dataset: str, seasonal_period: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        [
            series_diagnostics(series, dataset=dataset, series_id=str(series_id), seasonal_period=seasonal_period)
            for series_id, series in series_map.items()
        ]
    )
