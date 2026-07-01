from __future__ import annotations

import numpy as np
import pandas as pd

from s3paper.ablation_study import run_fastsketch_ablation, run_s3_ablation_study
from s3paper.baselines import evaluate_baseline
from s3paper.data_efficiency import run_data_efficiency_analysis
from s3paper.multi_prior_robustness import RollingPrior, run_multi_prior_robustness
from s3paper.residual_analysis import analyze_forecast_residuals
from s3paper.s3_forecaster_experiment import evaluate_s3_forecaster
from s3paper.shock_analysis import compare_s3_models_on_shocks


def _experiment_data():
    rng = np.random.default_rng(17)
    n = 108
    t = np.arange(n)
    index = pd.date_range("2014-01-31", periods=n, freq="ME")
    values = 15 + 0.03 * t + 1.5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.25, n)
    series = pd.Series(values, index=index)
    return series.iloc[:-12], series.iloc[-12:]


def _params():
    s3 = {
        "reservoir_size": 8,
        "spectral_radius": 0.8,
        "ar_lags": 3,
        "foundation_window": 6,
        "regressor_type": "Ridge",
        "reg_alpha": 1.0,
        "oob_split_ratio": 0.70,
        "aci_step_size": 0.05,
    }
    fast = {
        "foundation_window": 6,
        "ar_lags": 3,
        "ema_spans": (2, 4),
        "conv_scales": (2, 3),
        "use_calendar": True,
        "ridge_alpha": 1.0,
        "shrinkage_max": 1.5,
        "oob_split_ratio": 0.65,
    }
    return s3, fast


def test_required_analysis_sections_execute():
    train, test = _experiment_data()
    s3, fast = _params()

    s3_ablation = run_s3_ablation_study(
        train, test, s3, modes=["base_only", "full_aci"]
    )
    assert set(s3_ablation["results"]["status"]) == {"ok"}

    fast_ablation = run_fastsketch_ablation(
        train,
        test,
        fast,
        variants={"Full": {}, "Without AR lags": {"disabled_groups": {"ar"}}},
    )
    assert set(fast_ablation["results"]["status"]) == {"ok"}

    shock = compare_s3_models_on_shocks(
        train,
        test,
        s3,
        fast,
        threshold_method="quantile",
        quantile=0.80,
    )
    assert len(shock["summary"]) == 6

    efficiency = run_data_efficiency_analysis(
        train,
        test,
        {"S3-Forecaster": s3, "S3-FastSketch": fast},
        history_sizes=(60,),
        min_history=30,
    )
    assert set(efficiency["summary"]["status"]) == {"ok"}

    priors = run_multi_prior_robustness(
        train,
        test,
        {"rolling": lambda: RollingPrior(6)},
        s3_params=s3,
        fastsketch_params=fast,
    )
    assert set(priors["summary"]["status"]) == {"ok"}

    s3_output = evaluate_s3_forecaster(train, test, s3)
    residuals = analyze_forecast_residuals(test, s3_output["forecast"])
    assert residuals["diagnostics"]["n"] == len(test)

    baseline = evaluate_baseline(
        "SeasonalNaive", train, test, {"seasonal_period": 12}
    )
    assert len(baseline["forecast"]) == len(test)
