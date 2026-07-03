"""Shared utilities for the S3-Forecaster paper experiments."""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class ForecastOutput:
    """Normalized representation of a point or interval forecast."""

    pred: np.ndarray
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    raw: Any = None

    @property
    def has_interval(self) -> bool:
        return self.lower is not None and self.upper is not None


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def to_1d_array(x: Any, dtype=float) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("A DataFrame input must contain exactly one column.")
        x = x.iloc[:, 0]
    if isinstance(x, (pd.Series, pd.Index)):
        return np.asarray(x, dtype=dtype).reshape(-1)
    return np.asarray(x, dtype=dtype).reshape(-1)


def ensure_series(
    x: Any,
    *,
    name: str = "y",
    start: str = "2000-01-31",
    freq: str = "ME",
    drop_nonfinite: bool = True,
) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} must have exactly one column.")
        x = x.iloc[:, 0]

    if isinstance(x, pd.Series):
        out = x.copy().astype(float)
    else:
        values = np.asarray(x, dtype=float).reshape(-1)
        index = pd.date_range(start=start, periods=len(values), freq=freq)
        out = pd.Series(values, index=index, name=name)

    out.name = name
    out = out.replace([np.inf, -np.inf], np.nan)
    if drop_nonfinite:
        out = out.dropna()
    return out


def infer_frequency(index: pd.Index, default: str = "ME") -> str:
    if isinstance(index, pd.PeriodIndex):
        return index.freqstr or "M"
    if isinstance(index, pd.DatetimeIndex):
        return index.freqstr or pd.infer_freq(index) or default
    return default


def make_future_index(series: pd.Series, horizon: int, default_freq: str = "ME") -> pd.Index:
    series = ensure_series(series)
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    idx = series.index
    if isinstance(idx, pd.DatetimeIndex):
        freq = infer_frequency(idx, default=default_freq)
        offset = pd.tseries.frequencies.to_offset(freq)
        return pd.date_range(start=idx[-1] + offset, periods=horizon, freq=freq)
    if isinstance(idx, pd.PeriodIndex):
        freq = infer_frequency(idx, default="M")
        return pd.period_range(start=idx[-1] + 1, periods=horizon, freq=freq)
    if isinstance(idx, pd.RangeIndex):
        step = idx.step or 1
        start = (idx[-1] + step) if len(idx) else 0
        return pd.RangeIndex(start=start, stop=start + horizon * step, step=step)
    return pd.RangeIndex(start=len(series), stop=len(series) + horizon)


def append_series_value(series: pd.Series, value: float) -> pd.Series:
    series = ensure_series(series)
    next_index = make_future_index(series, 1)[0]
    return pd.concat([series, pd.Series([float(value)], index=[next_index], name=series.name)])


def parse_forecast_output(pred_obj: Any) -> ForecastOutput:
    """Normalize arrays, Series, or forecast DataFrames.

    Recognized point columns: pred, forecast, mean, median, yhat,
    predictions, prediction, and common median quantile names.
    Recognized interval columns: lower/q0.05/p05/0.05/0.1 and
    upper/q0.95/p95/0.95/0.9.
    """

    if isinstance(pred_obj, pd.DataFrame):
        colmap = {str(c).lower(): c for c in pred_obj.columns}
        point_names = (
            "pred",
            "forecast",
            "mean",
            "median",
            "yhat",
            "predictions",
            "prediction",
            "0.5",
            "q0.5",
            "q50",
            "p50",
        )
        lower_names = ("lower", "q0.05", "p05", "0.05", "q0.1", "p10", "0.1")
        upper_names = ("upper", "q0.95", "p95", "0.95", "q0.9", "p90", "0.9")
        pcol = next((colmap[n] for n in point_names if n in colmap), None)
        lcol = next((colmap[n] for n in lower_names if n in colmap), None)
        ucol = next((colmap[n] for n in upper_names if n in colmap), None)

        if lcol is not None and ucol is not None:
            lower = pred_obj[lcol].to_numpy(dtype=float).reshape(-1)
            upper = pred_obj[ucol].to_numpy(dtype=float).reshape(-1)
            pred = (
                pred_obj[pcol].to_numpy(dtype=float).reshape(-1)
                if pcol is not None
                else 0.5 * (lower + upper)
            )
            return ForecastOutput(pred=pred, lower=lower, upper=upper, raw=pred_obj)
        if pcol is not None:
            return ForecastOutput(
                pred=pred_obj[pcol].to_numpy(dtype=float).reshape(-1), raw=pred_obj
            )
        if pred_obj.shape[1] == 1:
            return ForecastOutput(
                pred=pred_obj.iloc[:, 0].to_numpy(dtype=float).reshape(-1), raw=pred_obj
            )
        raise ValueError(f"Cannot identify forecast columns in {list(pred_obj.columns)}.")

    if isinstance(pred_obj, pd.Series):
        return ForecastOutput(pred=pred_obj.to_numpy(dtype=float).reshape(-1), raw=pred_obj)

    return ForecastOutput(pred=np.asarray(pred_obj, dtype=float).reshape(-1), raw=pred_obj)


def call_predict_with_horizon(model: Any, history: pd.Series, horizon: int) -> Any:
    """Call heterogeneous predictor APIs without imposing a specific signature."""

    predict = getattr(model, "predict")
    try:
        signature = inspect.signature(predict)
        if "horizon" in signature.parameters:
            return predict(history, horizon=horizon)
        if "steps" in signature.parameters:
            return predict(steps=horizon)
    except (TypeError, ValueError):
        pass

    for call in (
        lambda: predict(history, horizon=horizon),
        lambda: predict(history, horizon),
        lambda: predict(history),
        lambda: predict(steps=horizon),
        lambda: predict(),
    ):
        try:
            return call()
        except TypeError:
            continue
    raise TypeError("The predictor API could not be called with the supported signatures.")


def recursive_point_forecast(model: Any, history: pd.Series, horizon: int) -> pd.Series:
    """Obtain a fixed-horizon forecast from direct or one-step predictors."""

    history = ensure_series(history)
    first = parse_forecast_output(call_predict_with_horizon(model, history, horizon)).pred
    if len(first) == horizon:
        return pd.Series(first, index=make_future_index(history, horizon), name="pred")

    current = history.copy()
    values = []
    for _ in range(horizon):
        out = parse_forecast_output(call_predict_with_horizon(model, current, 1)).pred
        if len(out) == 0:
            raise ValueError("The predictor returned an empty forecast.")
        value = float(out[-1])
        values.append(value)
        current = append_series_value(current, value)
    return pd.Series(values, index=make_future_index(history, horizon), name="pred")


def count_trainable_parameters(model: Any) -> int:
    """Count learned scalar parameters for sklearn, PyTorch, TensorFlow, and S3 models."""

    if model is None:
        return 0

    if hasattr(model, "trainable_parameter_count"):
        try:
            return int(model.trainable_parameter_count())
        except Exception:
            pass

    try:
        import torch

        if isinstance(model, torch.nn.Module):
            return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception:
        pass

    if hasattr(model, "count_params"):
        try:
            return int(model.count_params())
        except Exception:
            pass

    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        total = 0
        for estimator in estimators:
            total += np.asarray(getattr(estimator, "coef_", [])).size
            total += np.asarray(getattr(estimator, "intercept_", [])).size
        return int(total)

    total = 0
    for attr in ("coef_", "intercept_", "dual_coef_", "alpha_"):
        if hasattr(model, attr):
            total += np.asarray(getattr(model, attr)).size
    return int(total)


def filter_kwargs(mapping: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key in allowed}
