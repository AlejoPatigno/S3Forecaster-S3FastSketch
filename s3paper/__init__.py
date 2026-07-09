"""Reusable experimental code for the S3-Forecaster research project."""

from .metrics import (
    evaluate_forecast,
    interval_metrics,
    mape,
    mase,
    metrics_frame,
    paper_metric_row,
    point_metrics,
    rmsse,
    seasonal_naive_scale,
    smape,
    wape,
)
from .conformal import SequentialACI, finite_sample_quantile
from .preprocessing import FrozenRobustScaler, IdentityTransformer, PositiveLogTransformer
from .priors import (
    MANDATORY_PRIOR_NAMES,
    OPTIONAL_PRIOR_NAMES,
    PRIOR_REGISTRY,
    ChronosPrior,
    ETSPrior,
    FitPredictPriorAdapter,
    RollingMeanPrior,
    SeasonalNaivePrior,
    ThetaPrior,
    TimesFMPrior,
    build_prior,
    split_model_prior_params,
)
from .residual_analysis import (
    analyze_forecast_residuals,
    analyze_prior_residuals,
    plot_causal_volatility_diagnosis,
    plot_residual_diagnostics,
    residual_diagnostics,
)
from .rolling_evaluation import evaluate_rolling_model, rolling_one_step_forecast
from .selection import AdapterSelectionRule, residual_predictability_score, select_family
from .result_store import (
    REQUIRED_RESULT_COLUMNS,
    per_series_metrics_from_store,
    result_rows_from_forecast,
    save_result_store,
    validate_result_store,
)
from .diagnostics import diagnostics_frame, series_diagnostics
from .s3_forecaster import (
    EchoStateFeatureExtractor,
    S3Forecaster,
    S3ForecasterV6,
    SimpleFoundationProxy,
)
from .s3_fastsketch import FastRollingFoundation, S3FastSketchForecaster
from .chronos import ChronosZeroShotModel, chronos_predict_fixed_horizon

__all__ = [
    "S3Forecaster",
    "S3ForecasterV6",
    "S3FastSketchForecaster",
    "SimpleFoundationProxy",
    "FastRollingFoundation",
    "EchoStateFeatureExtractor",
    "RollingMeanPrior",
    "SeasonalNaivePrior",
    "ETSPrior",
    "ThetaPrior",
    "ChronosPrior",
    "TimesFMPrior",
    "FitPredictPriorAdapter",
    "MANDATORY_PRIOR_NAMES",
    "OPTIONAL_PRIOR_NAMES",
    "PRIOR_REGISTRY",
    "build_prior",
    "split_model_prior_params",
    "FrozenRobustScaler",
    "IdentityTransformer",
    "PositiveLogTransformer",
    "SequentialACI",
    "finite_sample_quantile",
    "AdapterSelectionRule",
    "residual_predictability_score",
    "select_family",
    "evaluate_rolling_model",
    "rolling_one_step_forecast",
    "ChronosZeroShotModel",
    "chronos_predict_fixed_horizon",
    "mape",
    "mase",
    "rmsse",
    "smape",
    "wape",
    "point_metrics",
    "interval_metrics",
    "evaluate_forecast",
    "seasonal_naive_scale",
    "paper_metric_row",
    "metrics_frame",
    "REQUIRED_RESULT_COLUMNS",
    "result_rows_from_forecast",
    "validate_result_store",
    "save_result_store",
    "per_series_metrics_from_store",
    "series_diagnostics",
    "diagnostics_frame",
    "residual_diagnostics",
    "analyze_prior_residuals",
    "analyze_forecast_residuals",
    "plot_causal_volatility_diagnosis",
    "plot_residual_diagnostics",
]

__version__ = "0.1.0"
