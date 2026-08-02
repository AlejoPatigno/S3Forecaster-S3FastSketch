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
from .calibration import InternalCalibrationSplit, split_internal_calibration
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
    available_prior_names,
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
from .prior_cache import PriorForecastCache
from .temporal_cv import aggregate_scores, make_expanding_window_folds
from .single_series_transfer_hpo import (
    evaluate_frozen_configuration_on_collection,
    optimize_on_development_series,
    run_single_series_hpo_transfer_experiment,
    select_development_series,
    temporal_train_cal_test_split,
)
from .result_store import (
    REQUIRED_RESULT_COLUMNS,
    assert_unique_metric_source,
    export_paper_tables,
    per_series_metrics_from_store,
    result_rows_from_forecast,
    save_result_store,
    validate_result_store,
)
from .diagnostics import diagnostics_frame, series_diagnostics
from .statistical_analysis import (
    critical_difference_data,
    diebold_mariano_hln,
    friedman_test,
    holm_corrected_pairwise_test,
    nemenyi_posthoc,
    paired_bootstrap_difference,
)
from .s3_forecaster import (
    EchoStateFeatureExtractor,
    S3Forecaster,
    S3ForecasterV6,
    SimpleFoundationProxy,
)
from .s3_fastsketch import FastRollingFoundation, S3FastSketchForecaster
from .chronos import ChronosZeroShotModel, chronos_predict_fixed_horizon
from .kaggle_loaders import DatasetBundle, load_dataset, load_series_prioritarias

__all__ = [
    "DatasetBundle",
    "load_dataset",
    "load_series_prioritarias",
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
    "available_prior_names",
    "build_prior",
    "split_model_prior_params",
    "FrozenRobustScaler",
    "IdentityTransformer",
    "PositiveLogTransformer",
    "SequentialACI",
    "finite_sample_quantile",
    "InternalCalibrationSplit",
    "split_internal_calibration",
    "AdapterSelectionRule",
    "residual_predictability_score",
    "select_family",
    "PriorForecastCache",
    "make_expanding_window_folds",
    "aggregate_scores",
    "temporal_train_cal_test_split",
    "select_development_series",
    "optimize_on_development_series",
    "evaluate_frozen_configuration_on_collection",
    "run_single_series_hpo_transfer_experiment",
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
    "assert_unique_metric_source",
    "export_paper_tables",
    "series_diagnostics",
    "diagnostics_frame",
    "friedman_test",
    "nemenyi_posthoc",
    "holm_corrected_pairwise_test",
    "paired_bootstrap_difference",
    "diebold_mariano_hln",
    "critical_difference_data",
    "residual_diagnostics",
    "analyze_prior_residuals",
    "analyze_forecast_residuals",
    "plot_causal_volatility_diagnosis",
    "plot_residual_diagnostics",
]

__version__ = "0.1.0"
