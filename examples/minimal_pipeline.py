"""Minimal executable example for the two proposed forecasting approaches."""

from __future__ import annotations

import numpy as np
import pandas as pd

from s3paper.paper_pipeline import evaluate_proposed_models


def make_example_series(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = 108
    time = np.arange(n)
    index = pd.date_range("2015-01-31", periods=n, freq="ME")
    values = (
        20.0
        + 0.04 * time
        + 2.0 * np.sin(2.0 * np.pi * time / 12.0)
        + rng.normal(0.0, 0.35, n)
    )
    series = pd.Series(values, index=index, name="synthetic_monthly")
    return series.iloc[:-12], series.iloc[-12:]


def main():
    train, test = make_example_series()

    s3_params = {
        "reservoir_size": 8,
        "spectral_radius": 0.8,
        "ar_lags": 3,
        "foundation_window": 6,
        "regressor_type": "Ridge",
        "reg_alpha": 1.0,
        "oob_split_ratio": 0.70,
        "aci_step_size": 0.05,
    }
    fastsketch_params = {
        "foundation_window": 6,
        "ar_lags": 3,
        "ema_spans": (2, 4),
        "conv_scales": (2, 3),
        "use_calendar": True,
        "ridge_alpha": 1.0,
        "shrinkage_max": 1.0,
        "oob_split_ratio": 0.65,
    }

    output = evaluate_proposed_models(
        train,
        test,
        s3_params=s3_params,
        fastsketch_params=fastsketch_params,
    )
    print(output["table"].to_string(index=False))


if __name__ == "__main__":
    main()
