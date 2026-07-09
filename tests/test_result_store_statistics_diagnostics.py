from __future__ import annotations

import numpy as np
import pandas as pd

from s3paper.ablation_study import run_fastsketch_ablation, run_s3_ablation_study
from s3paper.diagnostics import diagnostics_frame, series_diagnostics
from s3paper.metrics import evaluate_forecast
from s3paper.result_store import (
    REQUIRED_RESULT_COLUMNS,
    per_series_metrics_from_store,
    result_rows_from_forecast,
    validate_result_store,
)
from s3paper.rolling_evaluation import evaluate_rolling_model
from s3paper.s3_fastsketch import S3FastSketchForecaster
from s3paper.s3_forecaster import S3Forecaster
from s3paper.shock_analysis import (
    controlled_shock_metrics,
    inject_additive_spike,
    inject_level_shift,
    inject_seasonal_amplitude_shift,
    inject_temporary_pulse,
    inject_trend_shift,
    inject_variance_shift,
)
from s3paper.statistical_analysis import (
    coverage_breakdowns,
    diebold_mariano_hln,
    friedman_test,
    holm_posthoc,
    paired_bootstrap_difference,
    summarize_metric_by_model,
)


def _series(n: int = 72) -> pd.Series:
    t = np.arange(n)
    y = 20 + 0.05 * t + 2 * np.sin(2 * np.pi * t / 12)
    return pd.Series(y, index=pd.date_range("2018-01-31", periods=n, freq="ME"))


def test_original_scale_metrics_include_required_point_and_interval_metrics():
    train = _series(48)
    true = pd.Series([10.0, 20.0, 30.0])
    pred = pd.Series([11.0, 18.0, 33.0])
    metrics = evaluate_forecast(
        true,
        pred,
        y_train=train,
        lower=pred - 2,
        upper=pred + 2,
        seasonal_period=12,
    )

    for key in ("mase", "smape_percent", "rmse", "mae", "rmsse", "wape_percent", "mape_percent"):
        assert key in metrics
        assert np.isfinite(metrics[key])
    for key in ("ecp", "coverage_error", "mean_width", "interval_score", "msis"):
        assert key in metrics
        assert np.isfinite(metrics[key])
    scaled = evaluate_forecast(true * 10, pred * 10, y_train=train * 10, seasonal_period=12)
    assert np.isclose(metrics["mase"], scaled["mase"])
    assert np.isclose(metrics["rmsse"], scaled["rmsse"])


def test_canonical_result_store_generation_and_statistics():
    y = _series(84)
    train, test = y.iloc[:-8], y.iloc[-8:]
    model = S3FastSketchForecaster(
        ar_lags=2,
        ema_spans=(2, 4),
        conv_scales=(2,),
        min_train_samples=6,
        min_calib_samples=2,
    )
    evaluation = evaluate_rolling_model(model, train, test)
    store = result_rows_from_forecast(
        evaluation["forecast"],
        train_series=train,
        test_series=test,
        dataset="synthetic",
        series_id="s1",
        model="S3-FastSketch",
        prior_name="causal_rolling_mean",
        seed=42,
    )

    assert set(REQUIRED_RESULT_COLUMNS).issubset(store.columns)
    assert validate_result_store(store).equals(store)
    metrics = per_series_metrics_from_store(store)
    assert len(metrics) == 1
    assert np.isfinite(metrics["mase"].iloc[0])

    doubled = store.copy()
    doubled["model"] = "Naive"
    doubled["y_pred"] = doubled["y_true"].shift(1).bfill()
    combined = pd.concat([store, doubled], ignore_index=True)
    per_series = per_series_metrics_from_store(combined)
    summary = summarize_metric_by_model(per_series, n_boot=50)
    assert {"successful_series", "average_rank", "ci95_low", "ci95_high"}.issubset(summary.columns)
    assert friedman_test(per_series)["status"] in {"ok", "insufficient_data"}
    assert "p_holm" in holm_posthoc(per_series).columns
    assert paired_bootstrap_difference(per_series, "S3-FastSketch", "Naive", n_boot=50)["n_pairs"] >= 1
    assert diebold_mariano_hln([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])["status"] == "low_power"
    assert "pooled" in coverage_breakdowns(combined)


def test_series_diagnostics_have_no_placeholder_fields():
    y = _series(72)
    diagnostics = series_diagnostics(y, dataset="synthetic", series_id="s1")
    required = {
        "observations",
        "missing_fraction",
        "zero_fraction",
        "coefficient_of_variation",
        "adf_p_value",
        "kpss_p_value",
        "acf_lag1",
        "acf_seasonal_lag",
        "ljung_box_p_value",
        "trend_strength",
        "seasonal_strength",
        "spectral_entropy",
        "structural_break_count",
    }
    assert required.issubset(diagnostics)
    assert diagnostics["observations"] == len(y)
    frame = diagnostics_frame({"s1": y, "s2": y + 1}, dataset="synthetic")
    assert len(frame) == 2
    assert not frame.astype(str).isin(["TBD", "to be audited", ""]).any().any()


def test_deterministic_shock_generators_and_metrics():
    y = _series(60)
    generators = [
        inject_additive_spike(y, 5.0, start=30),
        inject_temporary_pulse(y, 3.0, duration=4, start=30),
        inject_level_shift(y, 2.0, start=30),
        inject_variance_shift(y, 0.5, start=30, seed=1),
        inject_trend_shift(y, 0.1, start=30),
        inject_seasonal_amplitude_shift(y, 1.0, start=30),
    ]
    for result in generators:
        assert {"series", "event_start", "duration", "magnitude", "counterfactual"}.issubset(result)
        assert len(result["series"]) == len(y)

    train, test = y.iloc[:48], generators[0]["series"].iloc[48:]
    forecast = pd.DataFrame(
        {
            "pred": y.iloc[48:].to_numpy(),
            "lower": y.iloc[48:].to_numpy() - 1,
            "upper": y.iloc[48:].to_numpy() + 1,
            "gate": np.linspace(0, 1, len(test)),
            "adapter_active": [True] * len(test),
        },
        index=test.index,
    )
    metrics = controlled_shock_metrics(
        train,
        test,
        forecast,
        event_start_position=max(0, generators[0]["event_start_position"] - len(train)),
        duration=generators[0]["duration"],
    )
    assert {"pre_shock_mase", "shock_window_mase", "post_shock_mase", "peak_error", "recovery_time"}.issubset(metrics)
    assert len(metrics["gate_trajectory"]) == len(test)
    assert all(metrics["adapter_activation"])


def test_ablation_flags_disable_intended_components():
    y = _series(84)
    train, test = y.iloc[:-6], y.iloc[-6:]
    s3_params = {
        "reservoir_size": 8,
        "ar_lags": 2,
        "prior_name": "causal_rolling_mean",
        "prior__window": 6,
        "calibration_split_ratio": 0.70,
    }
    fast_params = {
        "prior_name": "causal_rolling_mean",
        "prior__window": 6,
        "ar_lags": 2,
        "ema_spans": (2, 4),
        "conv_scales": (2,),
        "calibration_split_ratio": 0.65,
    }
    s3 = run_s3_ablation_study(
        train,
        test,
        s3_params,
        modes=["prior_only", "no_reservoir", "no_ar_lags", "no_adapter_selector", "full_model"],
    )
    assert s3["models"]["prior_only"].selector_report_["activated"] is False
    assert "reservoir" in s3["models"]["no_reservoir"].disabled_groups
    assert "ar" in s3["models"]["no_ar_lags"].disabled_groups
    assert s3["models"]["no_adapter_selector"].use_adapter_selector is False

    fast = run_fastsketch_ablation(
        train,
        test,
        fast_params,
        variants={
            "prior only": {"foundation_only": True},
            "no AR": {"disabled_groups": {"ar"}},
            "no contraction": {"use_shrinkage": False},
            "no adapter selector": {"use_adapter_selector": False},
            "full model": {},
        },
    )
    assert fast["models"]["prior only"].foundation_only is True
    assert "ar" in fast["models"]["no AR"].disabled_groups
    assert fast["models"]["no contraction"].use_shrinkage is False
    assert fast["models"]["no adapter selector"].use_adapter_selector is False
