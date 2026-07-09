"""Canonical rolling-origin evaluation."""

from __future__ import annotations

from typing import Any
import time

import pandas as pd

from .metrics import evaluate_forecast
from .utils import count_trainable_parameters, ensure_series


def rolling_one_step_forecast(model: Any, train_series: Any, test_series: Any) -> pd.DataFrame:
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    model.fit(train)
    rows = []
    information_cutoff = train.index[-1]
    for target_timestamp, observation in test.items():
        start = time.perf_counter()
        forecast = model.predict_one().copy()
        runtime_predict = time.perf_counter() - start
        forecast.index = pd.Index([target_timestamp])
        forecast["forecast_origin"] = information_cutoff
        forecast["information_cutoff"] = information_cutoff
        forecast["target_timestamp"] = target_timestamp
        forecast["target"] = float(observation)
        forecast["runtime_predict"] = runtime_predict
        selector = getattr(model, "selector_report_", {}) or {}
        forecast["adapter_active"] = bool(selector.get("activated", False))
        forecast["predictability_score"] = selector.get("r2_res_cal", selector.get("predictability_score", float("nan")))
        rows.append(forecast)
        model.update(float(observation))
        information_cutoff = target_timestamp
    return pd.concat(rows) if rows else pd.DataFrame()


def evaluate_rolling_model(
    model: Any,
    train_series: Any,
    test_series: Any,
    *,
    seasonal_period: int = 12,
    alpha: float = 0.10,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    forecast = rolling_one_step_forecast(model, train, test)
    if len(forecast):
        invalid = forecast["target_timestamp"] <= forecast["information_cutoff"]
        if invalid.any():
            raise AssertionError("Every target timestamp must be after its information cutoff.")
    metrics = evaluate_forecast(
        test,
        forecast["pred"],
        y_train=train,
        lower=forecast["lower"],
        upper=forecast["upper"],
        alpha=alpha,
        seasonal_period=seasonal_period,
        elapsed_seconds=elapsed_seconds,
        trainable_params=count_trainable_parameters(model),
    )
    return {"model": model, "forecast": forecast, "metrics": metrics}
