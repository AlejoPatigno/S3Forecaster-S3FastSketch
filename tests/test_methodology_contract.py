from __future__ import annotations

import numpy as np
import pandas as pd

from s3paper.conformal import SequentialACI, finite_sample_quantile
from s3paper.preprocessing import FrozenRobustScaler, PositiveLogTransformer
from s3paper.priors import (
    MANDATORY_PRIOR_NAMES,
    PRIOR_REGISTRY,
    RollingMeanPrior,
    build_prior,
    split_model_prior_params,
)
from s3paper.residual_features import EchoStateResidualTransformer
from s3paper.rolling_protocol import rolling_one_step_forecast
from s3paper.ablation_study import run_fastsketch_ablation, run_s3_ablation_study
from s3paper.s3_forecaster_experiment import evaluate_s3_forecaster
from s3paper.s3_fastsketch_experiment import evaluate_fastsketch
from s3paper.s3_fastsketch import S3FastSketchForecaster
from s3paper.s3_fastsketch_experiment import optimize_fastsketch_point
from s3paper.s3_forecaster_experiment import optimize_s3_forecaster


def _series(n: int = 72) -> pd.Series:
    t = np.arange(n)
    y = 10 + 0.1 * t + np.sin(2 * np.pi * t / 12)
    return pd.Series(y, index=pd.date_range("2020-01-31", periods=n, freq="ME"))


def test_prior_fitted_values_do_not_use_future_observations():
    y = _series(48)
    changed_future = y.copy()
    changed_future.iloc[30:] += 1000.0

    base = RollingMeanPrior(window_size=6).fit(y).fitted_values()
    perturbed = RollingMeanPrior(window_size=6).fit(changed_future).fitted_values()

    np.testing.assert_allclose(base.iloc[:30], perturbed.iloc[:30])


def test_frozen_scaler_does_not_refit_on_transform():
    scaler = FrozenRobustScaler().fit([1.0, 2.0, 3.0, 4.0])
    center = scaler.center_.copy()
    scale = scaler.scale_.copy()

    _ = scaler.transform([1.0, 10000.0])

    np.testing.assert_allclose(scaler.center_, center)
    np.testing.assert_allclose(scaler.scale_, scale)


def test_reservoir_radius_is_subunit_and_deterministic():
    a = EchoStateResidualTransformer(reservoir_size=12, spectral_radius=1.5, seed=11)
    b = EchoStateResidualTransformer(reservoir_size=12, spectral_radius=1.5, seed=11)

    assert a.achieved_spectral_radius_ < 1.0
    np.testing.assert_allclose(a.W_in_, b.W_in_)
    np.testing.assert_allclose(a.W_res_, b.W_res_)


def test_aci_uses_finite_sample_order_and_updates_after_interval():
    scores = [1.0, 2.0, 3.0, 4.0]
    assert finite_sample_quantile(scores, alpha=0.25) == 4.0

    aci = SequentialACI(target_miscoverage=0.1, step_size=0.05)
    aci.scores = [1.0, 2.0]
    before = list(aci.scores)
    interval = aci.interval(10.0)

    assert aci.scores == before
    aci.update(20.0, 10.0, interval=interval)
    assert len(aci.scores) == len(before) + 1


def test_fastsketch_canonical_gamma_is_nonnegative():
    model = S3FastSketchForecaster(shrinkage_max=1.5)
    gamma = model._estimate_gamma(np.asarray([1.0, 2.0, 3.0]), np.asarray([-3.0, -2.0, -1.0]))

    assert gamma == 0.0


def test_rolling_protocol_appends_observation_after_forecast():
    y = _series(72)
    train, test = y.iloc[:-6], y.iloc[-6:]
    model = S3FastSketchForecaster(
        ar_lags=2,
        ema_spans=(2, 4),
        conv_scales=(2,),
        min_train_samples=6,
        min_calib_samples=2,
    )

    forecast = rolling_one_step_forecast(model, train, test)

    assert len(forecast) == len(test)
    assert list(forecast.index) == list(test.index)
    assert np.isfinite(forecast["pred"]).all()


def test_positive_log_transform_roundtrip():
    y = np.asarray([1.0, 2.0, 5.0])
    transformer = PositiveLogTransformer().fit(y)

    np.testing.assert_allclose(transformer.inverse_transform(transformer.transform(y)), y)


def test_prior_registry_contains_required_names():
    assert set(MANDATORY_PRIOR_NAMES).issubset(PRIOR_REGISTRY)
    assert {"chronos", "timesfm"}.issubset(PRIOR_REGISTRY)


def test_mandatory_priors_fit_predict_and_update():
    y = _series(36)
    for prior_name in MANDATORY_PRIOR_NAMES:
        prior = build_prior(prior_name)
        prior.fit(y)
        fitted = prior.fitted_values()
        one = prior.predict_one()
        forecast = prior.predict(2)
        prior.update(y.index[-1] + pd.offsets.MonthEnd(1), float(y.iloc[-1]))

        assert len(fitted) == len(y)
        assert np.isfinite(one)
        assert len(forecast) == 2
        assert np.isfinite(fitted.to_numpy()).all()
        assert np.isfinite(forecast.to_numpy()).all()


def test_namespaced_prior_params_are_split_for_model_constructor():
    params, prior_name, prior_params = split_model_prior_params(
        {
            "prior_name": "causal_rolling_mean",
            "prior__window": 9,
            "ar_lags": 2,
        }
    )

    assert prior_name == "causal_rolling_mean"
    assert prior_params == {"window_size": 9}
    assert "prior__window" not in params
    assert params["prior_params"] == {"window_size": 9}


def test_optional_timesfm_failure_is_explicit():
    prior = build_prior("timesfm", model_id="missing-timesfm")

    try:
        prior.fit(_series(24))
    except ImportError as exc:
        assert "TimesFM prior requested" in str(exc)
    else:
        raise AssertionError("TimesFM prior should fail explicitly when unavailable.")


def test_s3_objective_records_namespaced_prior_params():
    y = _series(72)
    study = optimize_s3_forecaster(
        y,
        val_size=6,
        n_trials=1,
        prior_names=("causal_rolling_mean",),
    )
    trial = study.trials[0]

    assert trial.params["prior_name"] == "causal_rolling_mean"
    assert "prior__window" in trial.params
    assert trial.user_attrs["prior_name"] == "causal_rolling_mean"


def test_fastsketch_objective_records_namespaced_prior_params():
    y = _series(72)
    study = optimize_fastsketch_point(
        y,
        val_size=6,
        n_trials=1,
        prior_names=("causal_rolling_mean",),
    )
    trial = study.trials[0]

    assert trial.params["prior_name"] == "causal_rolling_mean"
    assert "prior__window" in trial.params
    assert trial.user_attrs["prior_name"] == "causal_rolling_mean"


def test_full_ablation_matches_production_forecasts():
    y = _series(84)
    train, test = y.iloc[:-6], y.iloc[-6:]
    s3_params = {
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
    fast_params = {
        "prior_name": "causal_rolling_mean",
        "prior__window": 6,
        "ar_lags": 2,
        "ema_spans": (2, 4),
        "conv_scales": (2,),
        "use_calendar": True,
        "ridge_alpha": 1.0,
        "shrinkage_max": 1.5,
        "calibration_split_ratio": 0.65,
    }

    production_s3 = evaluate_s3_forecaster(train, test, s3_params)["forecast"]["pred"].to_numpy()
    ablation_s3 = run_s3_ablation_study(train, test, s3_params, modes=["full_aci"])["forecasts"]["full_aci"]["pred"].to_numpy()
    production_fast = evaluate_fastsketch(train, test, fast_params)["forecast"]["pred"].to_numpy()
    ablation_fast = run_fastsketch_ablation(train, test, fast_params, variants={"Full": {}})["forecasts"]["Full"]["pred"].to_numpy()

    np.testing.assert_allclose(ablation_s3, production_s3)
    np.testing.assert_allclose(ablation_fast, production_fast)
