"""Canonical point and probabilistic metrics used throughout the paper."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .utils import to_1d_array


def mape(y_true: Any, y_pred: Any, eps: float = 1e-5, percentage: bool = True) -> float:
    yt = to_1d_array(y_true)
    yp = to_1d_array(y_pred)
    value = np.mean(np.abs(yt - yp) / np.maximum(np.abs(yt), eps))
    return float(100.0 * value if percentage else value)


def smape(y_true: Any, y_pred: Any, eps: float = 1e-5, percentage: bool = True) -> float:
    yt = to_1d_array(y_true)
    yp = to_1d_array(y_pred)
    value = np.mean(2.0 * np.abs(yt - yp) / np.maximum(np.abs(yt) + np.abs(yp), eps))
    return float(100.0 * value if percentage else value)


def point_metrics(y_true: Any, y_pred: Any, eps: float = 1e-5) -> dict[str, float]:
    yt = to_1d_array(y_true)
    yp = to_1d_array(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true={len(yt)}, y_pred={len(yp)}.")

    mse_value = float(mean_squared_error(yt, yp))
    try:
        r2_value = float(r2_score(yt, yp))
    except Exception:
        r2_value = np.nan

    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "mse": mse_value,
        "rmse": float(np.sqrt(mse_value)),
        "r2": r2_value,
        "mape_percent": mape(yt, yp, eps=eps, percentage=True),
        "mape": mape(yt, yp, eps=eps, percentage=False),
        "smape_percent": smape(yt, yp, eps=eps, percentage=True),
        "smape": smape(yt, yp, eps=eps, percentage=False),
    }


def mase(y_true: Any, y_pred: Any, y_train: Any, seasonal_period: int = 12) -> float:
    yt = to_1d_array(y_true)
    yp = to_1d_array(y_pred)
    scale = seasonal_naive_scale(y_train, seasonal_period=seasonal_period)
    return float(np.mean(np.abs(yt - yp)) / scale)


def rmsse(y_true: Any, y_pred: Any, y_train: Any, seasonal_period: int = 12) -> float:
    yt = to_1d_array(y_true)
    yp = to_1d_array(y_pred)
    scale = seasonal_naive_squared_scale(y_train, seasonal_period=seasonal_period)
    return float(np.sqrt(np.mean((yt - yp) ** 2) / scale))


def wape(y_true: Any, y_pred: Any, eps: float = 1e-8, percentage: bool = True) -> float:
    yt = to_1d_array(y_true)
    yp = to_1d_array(y_pred)
    value = float(np.sum(np.abs(yt - yp)) / max(float(np.sum(np.abs(yt))), float(eps)))
    return float(100.0 * value if percentage else value)


def seasonal_naive_scale(y_train: Any, seasonal_period: int = 12) -> float:
    y = to_1d_array(y_train)
    m = max(1, int(seasonal_period))
    if len(y) > m:
        scale = np.mean(np.abs(y[m:] - y[:-m]))
    elif len(y) > 1:
        scale = np.mean(np.abs(np.diff(y)))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 1e-8:
        scale = np.nanstd(y)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return float(scale)


def seasonal_naive_squared_scale(y_train: Any, seasonal_period: int = 12) -> float:
    y = to_1d_array(y_train)
    m = max(1, int(seasonal_period))
    if len(y) > m:
        scale = np.mean((y[m:] - y[:-m]) ** 2)
    elif len(y) > 1:
        scale = np.mean(np.diff(y) ** 2)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.nanvar(y)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return float(scale)


def interval_score_per_t(
    y_true: Any,
    lower: Any,
    upper: Any,
    alpha: float = 0.10,
) -> np.ndarray:
    yt = to_1d_array(y_true)
    lo = to_1d_array(lower)
    up = to_1d_array(upper)
    if not (len(yt) == len(lo) == len(up)):
        raise ValueError("y_true, lower, and upper must have the same length.")
    lo, up = np.minimum(lo, up), np.maximum(lo, up)
    width = up - lo
    below = yt < lo
    above = yt > up
    return (
        width
        + (2.0 / alpha) * (lo - yt) * below
        + (2.0 / alpha) * (yt - up) * above
    )


def interval_metrics(
    y_true: Any,
    lower: Any,
    upper: Any,
    y_train: Any,
    *,
    alpha: float = 0.10,
    seasonal_period: int = 12,
) -> dict[str, float]:
    yt = to_1d_array(y_true)
    lo = to_1d_array(lower)
    up = to_1d_array(upper)
    lo, up = np.minimum(lo, up), np.maximum(lo, up)

    covered = (yt >= lo) & (yt <= up)
    scores = interval_score_per_t(yt, lo, up, alpha=alpha)
    scale = seasonal_naive_scale(y_train, seasonal_period=seasonal_period)
    ecp = float(np.mean(covered))

    return {
        "ecp": ecp,
        "coverage_error": float(abs(ecp - (1.0 - alpha))),
        "iw": float(np.mean(up - lo)),
        "mean_width": float(np.mean(up - lo)),
        "mis": float(np.mean(scores)),
        "interval_score": float(np.mean(scores)),
        "msis": float(np.mean(scores) / scale),
        "scale": scale,
    }


def evaluate_forecast(
    y_true: Any,
    y_pred: Any,
    *,
    y_train: Optional[Any] = None,
    lower: Optional[Any] = None,
    upper: Optional[Any] = None,
    alpha: float = 0.10,
    seasonal_period: int = 12,
    eps: float = 1e-5,
    elapsed_seconds: Optional[float] = None,
    trainable_params: Optional[int] = None,
) -> dict[str, float]:
    result = point_metrics(y_true, y_pred, eps=eps)
    result["wape_percent"] = wape(y_true, y_pred, eps=eps, percentage=True)
    result["wape"] = wape(y_true, y_pred, eps=eps, percentage=False)
    if y_train is not None:
        result["mase"] = mase(
            y_true,
            y_pred,
            y_train,
            seasonal_period=seasonal_period,
        )
        result["rmsse"] = rmsse(
            y_true,
            y_pred,
            y_train,
            seasonal_period=seasonal_period,
        )
    else:
        result["mase"] = np.nan
        result["rmsse"] = np.nan
    if lower is not None and upper is not None:
        if y_train is None:
            raise ValueError("y_train is required to calculate MSIS.")
        result.update(
            interval_metrics(
                y_true,
                lower,
                upper,
                y_train,
                alpha=alpha,
                seasonal_period=seasonal_period,
            )
        )
    else:
        result.update(
            {
                "ecp": np.nan,
                "coverage_error": np.nan,
                "iw": np.nan,
                "mean_width": np.nan,
                "mis": np.nan,
                "interval_score": np.nan,
                "msis": np.nan,
                "scale": np.nan,
            }
        )

    result["elapsed_seconds"] = float(elapsed_seconds) if elapsed_seconds is not None else np.nan
    result["trainable_params"] = int(trainable_params) if trainable_params is not None else np.nan
    return result


def paper_metric_row(model_name: str, metrics: dict[str, float]) -> dict[str, float]:
    """Return publication-oriented column names."""

    return {
        "Model": model_name,
        "MAE": metrics.get("mae", np.nan),
        "MSE": metrics.get("mse", np.nan),
        "RMSE": metrics.get("rmse", np.nan),
        "R^2": metrics.get("r2", np.nan),
        "MAPE (%)": metrics.get("mape_percent", np.nan),
        "sMAPE (%)": metrics.get("smape_percent", np.nan),
        "WAPE (%)": metrics.get("wape_percent", np.nan),
        "MASE": metrics.get("mase", np.nan),
        "RMSSE": metrics.get("rmsse", np.nan),
        "Time (s)": metrics.get("elapsed_seconds", np.nan),
        "trainable_params": metrics.get("trainable_params", np.nan),
        "ECP": metrics.get("ecp", np.nan),
        "IW": metrics.get("iw", np.nan),
        "MIS": metrics.get("mis", np.nan),
        "MSIS": metrics.get("msis", np.nan),
    }


def metrics_frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
