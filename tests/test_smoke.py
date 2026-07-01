from __future__ import annotations

import numpy as np
import pandas as pd

from s3paper.metrics import evaluate_forecast
from s3paper.s3_fastsketch_experiment import evaluate_fastsketch
from s3paper.s3_forecaster_experiment import evaluate_s3_forecaster


def _monthly_data():
    rng = np.random.default_rng(7)
    n = 108
    t = np.arange(n)
    index = pd.date_range("2015-01-31", periods=n, freq="ME")
    y = 20 + 0.04 * t + 2 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.35, n)
    series = pd.Series(y, index=index)
    return series.iloc[:-12], series.iloc[-12:]


def test_metrics_percentage_and_fraction_are_consistent():
    metrics = evaluate_forecast([10.0, 20.0], [9.0, 18.0])
    assert np.isclose(metrics["mape_percent"], 100.0 * metrics["mape"])
    assert np.isfinite(metrics["rmse"])


def test_s3_forecaster_end_to_end():
    train, test = _monthly_data()
    output = evaluate_s3_forecaster(
        train,
        test,
        {
            "reservoir_size": 8,
            "spectral_radius": 0.8,
            "ar_lags": 3,
            "foundation_window": 6,
            "regressor_type": "Ridge",
            "reg_alpha": 1.0,
            "oob_split_ratio": 0.70,
            "aci_step_size": 0.05,
        },
    )
    assert output["model"].fitted
    assert len(output["forecast"]) == len(test)
    assert output["metrics"]["trainable_params"] > 0
    assert np.isfinite(output["metrics"]["mape_percent"])


def test_fastsketch_end_to_end():
    train, test = _monthly_data()
    output = evaluate_fastsketch(
        train,
        test,
        {
            "foundation_window": 6,
            "ar_lags": 3,
            "ema_spans": (2, 4),
            "conv_scales": (2, 3),
            "use_calendar": True,
            "ridge_alpha": 1.0,
            "shrinkage_max": 1.5,
            "oob_split_ratio": 0.65,
        },
    )
    assert output["model"].fitted
    assert len(output["forecast"]) == len(test)
    assert output["metrics"]["trainable_params"] > 0
    assert np.isfinite(output["metrics"]["mape_percent"])
