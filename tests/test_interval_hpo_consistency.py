from __future__ import annotations

import numpy as np
import pandas as pd

from s3paper.metrics import evaluate_forecast
from s3paper.multiseries_hpo import optimize_s3_collection
from s3paper.s3_fastsketch import S3FastSketchForecaster
from s3paper.s3_fastsketch_experiment import evaluate_fastsketch
from s3paper.s3_forecaster import S3Forecaster
from s3paper.s3_forecaster_experiment import evaluate_s3_forecaster


def _series(n: int = 96) -> pd.Series:
    rng = np.random.default_rng(11)
    t = np.arange(n)
    values = 20 + 0.05 * t + np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.2, n)
    return pd.Series(values, index=pd.date_range("2010-01-31", periods=n, freq="ME"))


def _s3_params() -> dict:
    return {
        "reservoir_size": 8,
        "spectral_radius": 0.8,
        "leak_rate": 0.5,
        "ar_lags": 2,
        "prior_name": "causal_rolling_mean",
        "prior__window": 6,
        "regressor_type": "Ridge",
        "reg_alpha": 1.0,
        "calibration_split_ratio": 0.70,
        "aci_step_size": 0.05,
    }


def _s3_model_params() -> dict:
    params = dict(_s3_params())
    params.pop("prior__window")
    params["prior_params"] = {"window_size": 6}
    return params


def _fast_params() -> dict:
    return {
        "foundation_window": 6,
        "ar_lags": 2,
        "ema_spans": (2, 4),
        "conv_scales": (2, 3),
        "use_calendar": True,
        "ridge_alpha": 1.0,
        "shrinkage_max": 1.5,
        "calibration_split_ratio": 0.65,
    }


def test_s3_update_uses_reported_interval():
    model = S3Forecaster(**_s3_model_params(), interval_scale=0.5).fit(_series().iloc[:-6])
    row = model.predict_one()
    pred = float(row["pred"].iloc[0])
    lower = float(row["lower"].iloc[0])
    upper = float(row["upper"].iloc[0])
    alpha_before = model.current_alpha_t
    miss_target = upper + max(1.0, upper - lower)

    model.update(miss_target, is_observed=True)

    expected = np.clip(alpha_before + model.aci_step_size * (model.target_miscoverage - 1), 0.01, 0.50)
    assert lower <= pred <= upper
    assert np.isclose(model.current_alpha_t, expected)


def test_fastsketch_update_uses_reported_interval():
    model = S3FastSketchForecaster(**_fast_params(), interval_scale=0.5, min_train_samples=6).fit(_series().iloc[:-6])
    row = model.predict_one()
    upper = float(row["upper"].iloc[0])
    alpha_before = model.current_alpha_t

    model.update(upper + 1.0, is_observed=True)

    expected = np.clip(alpha_before + model.aci_step_size * (model.aci_target - 1), 0.01, 0.50)
    assert np.isclose(model.current_alpha_t, expected)


def test_recursive_prediction_does_not_update_aci():
    s3 = S3Forecaster(**_s3_model_params()).fit(_series().iloc[:-6])
    s3_alpha = s3.current_alpha_t
    s3_scores = len(s3.conformity_scores)
    s3.predict(steps=3)
    assert s3.current_alpha_t == s3_alpha
    assert len(s3.conformity_scores) == s3_scores

    fast = S3FastSketchForecaster(**_fast_params(), min_train_samples=6).fit(_series().iloc[:-6])
    fast_alpha = fast.current_alpha_t
    fast_scores = len(fast.conformity_scores)
    fast.predict(steps=3)
    assert fast.current_alpha_t == fast_alpha
    assert len(fast.conformity_scores) == fast_scores


def test_returned_metrics_match_returned_forecast():
    y = _series()
    train, test = y.iloc[:-6], y.iloc[-6:]

    s3 = evaluate_s3_forecaster(train, test, _s3_params(), seasonal_period=12)
    s3_recomputed = evaluate_forecast(
        test,
        s3["forecast"]["pred"],
        y_train=train,
        lower=s3["forecast"]["lower"],
        upper=s3["forecast"]["upper"],
        seasonal_period=12,
    )
    assert np.isclose(s3["metrics"]["ecp"], s3_recomputed["ecp"])
    assert np.isclose(s3["metrics"]["mis"], s3_recomputed["mis"])
    assert np.isclose(s3["metrics"]["msis"], s3_recomputed["msis"])

    fast = evaluate_fastsketch(train, test, _fast_params(), seasonal_period=12)
    fast_recomputed = evaluate_forecast(
        test,
        fast["forecast"]["pred"],
        y_train=train,
        lower=fast["forecast"]["lower"],
        upper=fast["forecast"]["upper"],
        seasonal_period=12,
    )
    assert np.isclose(fast["metrics"]["ecp"], fast_recomputed["ecp"])
    assert np.isclose(fast["metrics"]["mis"], fast_recomputed["mis"])
    assert np.isclose(fast["metrics"]["msis"], fast_recomputed["msis"])


def test_multiseries_hpo_has_no_nested_study(monkeypatch):
    import optuna

    original_create_study = optuna.create_study
    calls = []

    def counting_create_study(*args, **kwargs):
        calls.append((args, kwargs))
        return original_create_study(*args, **kwargs)

    monkeypatch.setattr(optuna, "create_study", counting_create_study)
    y = _series(84)
    optimize_s3_collection(
        {"a": y, "b": y + 1.0},
        n_trials=1,
        n_folds=1,
        validation_size=3,
        prior_names=("causal_rolling_mean",),
    )

    assert len(calls) == 1
