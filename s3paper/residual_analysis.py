"""Residual diagnostics used to motivate and validate the residual adapters."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from .metrics import point_metrics
from .utils import ensure_series, parse_forecast_output


def residual_diagnostics(
    residuals: Any,
    *,
    fitted: Optional[Any] = None,
    seasonal_period: int = 12,
    alpha: float = 0.05,
    ljungbox_lags: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Compute descriptive, normality, and serial-dependence diagnostics."""

    r = np.asarray(residuals, dtype=float).reshape(-1)
    finite = np.isfinite(r)
    r = r[finite]
    if len(r) < 8:
        raise ValueError("At least eight finite residuals are required.")

    m = max(1, int(seasonal_period))
    max_lag = min(max(2 * m, 10), len(r) - 1)
    acf_values = acf(r, nlags=max_lag, fft=True)
    seasonal_acf = float(acf_values[m]) if m < len(acf_values) else np.nan

    if ljungbox_lags is None:
        candidates = [min(m, len(r) // 4), min(2 * m, len(r) // 3)]
        ljungbox_lags = sorted({lag for lag in candidates if lag >= 1})
    lb = acorr_ljungbox(r, lags=ljungbox_lags, return_df=True)

    shapiro_stat, shapiro_p = stats.shapiro(r)
    dag_stat, dag_p = stats.normaltest(r)
    jb_stat, jb_p = stats.jarque_bera(r)

    result = {
        "n": int(len(r)),
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)),
        "median": float(np.median(r)),
        "mad": float(stats.median_abs_deviation(r, scale="normal")),
        "skewness": float(stats.skew(r, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)),
        "seasonal_acf": seasonal_acf,
        "acf": acf_values,
        "ljung_box": lb,
        "normality": {
            "shapiro": {"statistic": float(shapiro_stat), "p_value": float(shapiro_p)},
            "dagostino_k2": {"statistic": float(dag_stat), "p_value": float(dag_p)},
            "jarque_bera": {"statistic": float(jb_stat), "p_value": float(jb_p)},
        },
        "normality_not_rejected_all": bool(
            shapiro_p >= alpha and dag_p >= alpha and jb_p >= alpha
        ),
        "serial_independence_not_rejected_all": bool((lb["lb_pvalue"] >= alpha).all()),
        "explicit_seasonality_recommended": bool(
            np.isfinite(seasonal_acf) and abs(seasonal_acf) > 0.20
        ),
    }

    if fitted is not None:
        fitted_array = np.asarray(fitted, dtype=float).reshape(-1)
        if len(fitted_array) != len(finite):
            raise ValueError("fitted and residuals must have the same original length.")
        result["fitted"] = fitted_array[finite]
    return result


def analyze_prior_residuals(
    series: Any,
    prior: Any,
    *,
    horizon: int = 1,
    seasonal_period: int = 12,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate the in-sample prior fit and diagnose the unexplained residual."""

    y = ensure_series(series)
    if not hasattr(prior, "fit_predict"):
        raise TypeError("prior must implement fit_predict(series, horizon).")
    fitted, future = prior.fit_predict(y, horizon)
    fitted = ensure_series(fitted, name="fitted").reindex(y.index).ffill().bfill()
    residuals = y.to_numpy() - fitted.to_numpy()

    return {
        "point_metrics": point_metrics(y, fitted),
        "residuals": pd.Series(residuals, index=y.index, name="residual"),
        "diagnostics": residual_diagnostics(
            residuals,
            fitted=fitted,
            seasonal_period=seasonal_period,
            alpha=alpha,
        ),
        "fitted": fitted,
        "future": future,
    }


def analyze_forecast_residuals(
    y_true: Any,
    forecast: Any,
    *,
    seasonal_period: int = 12,
    alpha: float = 0.05,
) -> dict[str, Any]:
    out = parse_forecast_output(forecast)
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    if len(yt) != len(out.pred):
        raise ValueError("Forecast and target lengths differ.")
    residuals = yt - out.pred
    return {
        "point_metrics": point_metrics(yt, out.pred),
        "residuals": residuals,
        "diagnostics": residual_diagnostics(
            residuals,
            fitted=out.pred,
            seasonal_period=seasonal_period,
            alpha=alpha,
        ),
    }


def plot_residual_diagnostics(
    residuals: Any,
    *,
    fitted: Optional[Any] = None,
    seasonal_period: int = 12,
):
    """Create separate publication-friendly diagnostic figures."""

    import matplotlib.pyplot as plt

    r = np.asarray(residuals, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]

    fig_series, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual")
    fig_series.tight_layout()

    fig_hist, ax = plt.subplots(figsize=(7, 4))
    ax.hist(r, bins="auto", density=True, alpha=0.7)
    x = np.linspace(np.min(r), np.max(r), 300)
    ax.plot(x, stats.norm.pdf(x, loc=np.mean(r), scale=np.std(r, ddof=1)))
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    fig_hist.tight_layout()

    fig_qq = plt.figure(figsize=(5, 5))
    stats.probplot(r, dist="norm", plot=plt)
    fig_qq.tight_layout()

    max_lag = min(max(24, 2 * seasonal_period), len(r) - 1)
    acf_values = acf(r, nlags=max_lag, fft=True)
    fig_acf, ax = plt.subplots(figsize=(8, 4))
    ax.stem(np.arange(len(acf_values)), acf_values)
    ax.axvline(seasonal_period, linestyle="--", linewidth=1)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    fig_acf.tight_layout()

    figures = {
        "residual_series": fig_series,
        "histogram": fig_hist,
        "qq_plot": fig_qq,
        "acf": fig_acf,
    }

    if fitted is not None:
        yhat = np.asarray(fitted, dtype=float).reshape(-1)
        if len(yhat) != len(r):
            raise ValueError("fitted must have the same finite length as residuals.")
        fig_scatter, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(yhat, r, s=16)
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("Fitted value")
        ax.set_ylabel("Residual")
        fig_scatter.tight_layout()
        figures["residuals_vs_fitted"] = fig_scatter

    return figures
