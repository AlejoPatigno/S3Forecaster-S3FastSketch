"""Reusable experimental code for the S3-Forecaster research project."""

from .metrics import (
    evaluate_forecast,
    interval_metrics,
    mape,
    metrics_frame,
    paper_metric_row,
    point_metrics,
    seasonal_naive_scale,
    smape,
)
from .residual_analysis import (
    analyze_forecast_residuals,
    analyze_prior_residuals,
    plot_causal_volatility_diagnosis,
    plot_residual_diagnostics,
    residual_diagnostics,
)
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
    "ChronosZeroShotModel",
    "chronos_predict_fixed_horizon",
    "mape",
    "smape",
    "point_metrics",
    "interval_metrics",
    "evaluate_forecast",
    "seasonal_naive_scale",
    "paper_metric_row",
    "metrics_frame",
    "residual_diagnostics",
    "analyze_prior_residuals",
    "analyze_forecast_residuals",
    "plot_causal_volatility_diagnosis",
    "plot_residual_diagnostics",
]

__version__ = "0.1.0"
