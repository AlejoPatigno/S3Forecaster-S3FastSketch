"""Chronos adapters for baselines and foundation-prior robustness.

The project keeps Chronos optional. Tests and reproducible CPU runs can pass a
mock or callable predictor, while full Chronos runs may use chronos-forecasting.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .utils import (
    call_predict_with_horizon,
    ensure_series,
    make_future_index,
    parse_forecast_output,
)


def _to_numpy(obj: Any) -> np.ndarray:
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "numpy"):
        obj = obj.numpy()
    return np.asarray(obj, dtype=float)


def extract_chronos_point(pred_obj: Any, horizon: Optional[int] = None) -> np.ndarray:
    """Extract a point forecast from common Chronos output shapes.

    Supported shapes include pandas outputs, dicts with point or sample keys,
    Chronos sample tensors, and arrays shaped as samples x horizon or
    batch x samples x horizon.
    """

    if isinstance(pred_obj, dict):
        for key in ("pred", "forecast", "mean", "median", "yhat", "predictions"):
            if key in pred_obj:
                return extract_chronos_point(pred_obj[key], horizon=horizon)
        if "samples" in pred_obj:
            return extract_chronos_point(pred_obj["samples"], horizon=horizon)

    if hasattr(pred_obj, "samples"):
        return extract_chronos_point(pred_obj.samples, horizon=horizon)

    if isinstance(pred_obj, (pd.Series, pd.DataFrame)):
        return parse_forecast_output(pred_obj).pred

    arr = _to_numpy(pred_obj)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr.reshape(-1)

    if horizon is not None:
        horizon = int(horizon)
        if arr.shape[-1] == horizon:
            return np.nanmedian(arr.reshape(-1, horizon), axis=0)
        if arr.shape[0] == horizon:
            return np.nanmedian(arr.reshape(horizon, -1), axis=1)

    return np.nanmedian(arr.reshape(-1, arr.shape[-1]), axis=0)


def _call_external_predictor(predictor: Any, history: pd.Series, horizon: int) -> Any:
    if hasattr(predictor, "predict"):
        try:
            return call_predict_with_horizon(predictor, history, horizon)
        except TypeError:
            predict = predictor.predict
            for call in (
                lambda: predict(history, prediction_length=horizon),
                lambda: predict(history.to_numpy(), prediction_length=horizon),
                lambda: predict(history.to_numpy(), horizon),
            ):
                try:
                    return call()
                except TypeError:
                    continue
            raise
    return predictor(history, horizon)


def chronos_predict_fixed_horizon(predictor: Any, history: Any, horizon: int) -> pd.Series:
    """Return exactly ``horizon`` point forecasts from direct or one-step Chronos APIs."""

    history = ensure_series(history)
    horizon = int(horizon)
    first = extract_chronos_point(
        _call_external_predictor(predictor, history, horizon),
        horizon=horizon,
    )
    if len(first) == horizon:
        return pd.Series(first, index=make_future_index(history, horizon), name="pred")

    current = history.copy()
    values = []
    for _ in range(horizon):
        out = extract_chronos_point(
            _call_external_predictor(predictor, current, 1),
            horizon=1,
        )
        if len(out) == 0:
            raise ValueError("Chronos returned an empty forecast.")
        value = float(out[-1])
        values.append(value)
        current.loc[make_future_index(current, 1)[0]] = value
    return pd.Series(values, index=make_future_index(history, horizon), name="pred")


class ChronosZeroShotModel:
    """Lazy wrapper around chronos-forecasting pipelines.

    Parameters are intentionally light so the baseline can also work with the
    newer Chronos-2 ``predict_df`` API and the original sample-based API.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-small",
        *,
        device_map: str = "cpu",
        torch_dtype: Any = None,
        quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
        prediction_kwargs: Optional[dict[str, Any]] = None,
        pipeline: Any = None,
    ):
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.quantile_levels = quantile_levels
        self.prediction_kwargs = dict(prediction_kwargs or {})
        self.pipeline = pipeline

    def _load_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        try:
            from chronos import (
                BaseChronosPipeline,
                Chronos2Pipeline,
            )
        except ModuleNotFoundError as exc:
            raise ImportError(
                "The chronos module was not found. Install "
                "chronos-forecasting before using Chronos."
            ) from exc
        except Exception as exc:
            raise ImportError(
                "Chronos is installed, but one of its dependencies "
                "failed during import. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        kwargs = {"device_map": self.device_map}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype

        pipeline_class = Chronos2Pipeline if "chronos-2" in self.model_id else BaseChronosPipeline
        self.pipeline = pipeline_class.from_pretrained(self.model_id, **kwargs)
        return self.pipeline

    def _predict_df(self, pipeline: Any, history: pd.Series, horizon: int):
        context = history.copy()
        if not isinstance(context.index, pd.DatetimeIndex):
            context.index = pd.date_range("2000-01-31", periods=len(context), freq="ME")
        frame = pd.DataFrame(
            {
                "id": "series",
                "timestamp": context.index,
                "target": context.to_numpy(),
            }
        )
        output = pipeline.predict_df(
            frame,
            prediction_length=horizon,
            quantile_levels=list(self.quantile_levels),
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            **self.prediction_kwargs,
        )
        output = output[output["id"] == "series"] if "id" in output.columns else output
        return output

    def predict(self, history: Any, horizon: int):
        history = ensure_series(history)
        pipeline = self._load_pipeline()
        if hasattr(pipeline, "predict_df"):
            return self._predict_df(pipeline, history, int(horizon))

        try:
            import torch

            context = torch.tensor(history.to_numpy(dtype=float))
        except Exception:
            context = history.to_numpy(dtype=float)

        return pipeline.predict(
            context,
            prediction_length=int(horizon),
            **self.prediction_kwargs,
        )


def make_chronos_predictor(
    *,
    model_or_factory: Any = None,
    predict_fn: Optional[Callable] = None,
    **chronos_kwargs: Any,
):
    if predict_fn is not None:
        return predict_fn
    if model_or_factory is not None:
        return model_or_factory() if callable(model_or_factory) else model_or_factory
    return ChronosZeroShotModel(**chronos_kwargs)
