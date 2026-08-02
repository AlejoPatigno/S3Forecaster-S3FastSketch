"""Hyperparameter optimization and final testing for S3-Forecaster."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from .metrics import evaluate_forecast, seasonal_naive_scale
from .priors import MANDATORY_PRIOR_NAMES, split_model_prior_params, suggest_prior_params, validate_prior_names
from .rolling_protocol import evaluate_rolling_model
from .s3_forecaster import S3Forecaster
from .temporal_cv import make_expanding_window_folds, aggregate_scores
from .utils import count_trainable_parameters, ensure_series


def _normalize_holdout_collection(train_series: Any, calibration_series: Any):
    """Return aligned, validated holdout splits for one or many series."""

    from collections.abc import Mapping

    train_is_mapping = isinstance(train_series, Mapping)
    calibration_is_mapping = isinstance(calibration_series, Mapping)
    if train_is_mapping != calibration_is_mapping:
        raise TypeError(
            "train_series and calibration_series must both be mappings "
            "or both be individual series."
        )

    if not train_is_mapping:
        return [
            (
                "development",
                ensure_series(train_series, name="train"),
                ensure_series(calibration_series, name="calibration"),
            )
        ]

    train_ids = {str(series_id) for series_id in train_series}
    calibration_ids = {str(series_id) for series_id in calibration_series}
    if train_ids != calibration_ids:
        raise ValueError(
            "train_series and calibration_series must contain the same series IDs."
        )
    if not train_ids:
        raise ValueError("At least one HPO series is required.")

    normalized_train = {str(key): value for key, value in train_series.items()}
    normalized_calibration = {
        str(key): value for key, value in calibration_series.items()
    }
    return [
        (
            series_id,
            ensure_series(normalized_train[series_id], name=f"{series_id}_train"),
            ensure_series(
                normalized_calibration[series_id],
                name=f"{series_id}_calibration",
            ),
        )
        for series_id in sorted(train_ids)
    ]

def suggest_s3_point_params(
    trial,
    *,
    train_length: int,
    seasonal_period: int,
    prior_names=None,
) -> dict:
    params = {
        "reservoir_size": trial.suggest_int("reservoir_size", 1, 200),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.05, 0.99),
        "leak_rate": trial.suggest_float("leak_rate", 0.05, 1.0),
        "ar_lags": trial.suggest_int("ar_lags", 2, min(12, max(2, train_length // 6))),
        "regressor_type": trial.suggest_categorical(
            "regressor_type", ["Ridge", "ElasticNet", "Lasso"]
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 50.0, log=True),
        "aci_step_size": trial.suggest_float("aci_step_size", 0.01, 0.2),
        "calibration_split_ratio": trial.suggest_float("calibration_split_ratio", 0.60, 0.85),
    }
    params.update(
        suggest_prior_params(
            trial,
            train_length=train_length,
            seasonal_period=seasonal_period,
            prior_names=prior_names or MANDATORY_PRIOR_NAMES,
        )
    )
    return params


def optimize_s3_forecaster(
    train_series: Any,
    *,
    seasonal_period: int = 12,
    n_folds: int = 3,
    validation_size: int = 6,
    val_size: int | None = None,
    n_trials: int = 100,
    seed: int = 42,
    objective_metric: str = "paper_point",
    smape_weight: float = 1.0,
    prior_names: tuple[str, ...] | list[str] | None = None,
):
    """Optimize point-forecast hyperparameters using temporal CV."""

    import optuna

    if val_size is not None:
        validation_size = int(val_size)
    full = ensure_series(train_series)
    folds = make_expanding_window_folds(
        full,
        n_folds=n_folds,
        validation_size=validation_size,
    )
    minimum_fold_train_length = len(folds[0][0])

    def objective(trial: optuna.Trial) -> float:
        params = suggest_s3_point_params(
            trial,
            train_length=minimum_fold_train_length,
            seasonal_period=seasonal_period,
            prior_names=prior_names,
        )
        
        try:
            model_params, prior_name, prior_params = split_model_prior_params(params)
            trial.set_user_attr("prior_name", prior_name)
            trial.set_user_attr("prior_params", prior_params)
            
            scores = []
            for fold_train, fold_val in folds:
                model = S3Forecaster(horizon=1, **model_params)
                result = evaluate_rolling_model(
                    model, 
                    fold_train, 
                    fold_val,
                    seasonal_period=seasonal_period,
                )
                metrics = result["metrics"]
                score = (
                    metrics["mase"] + smape_weight * metrics["smape_percent"] / 100.0
                    if objective_metric == "paper_point"
                    else metrics[objective_metric]
                )
                scores.append(score)
            
            final_score = float(np.median(scores))
            return final_score if np.isfinite(final_score) else float("inf")
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    return study


def optimize_s3_forecaster_holdout(
    train_series: Any,
    calibration_series: Any,
    *,
    seasonal_period: int = 12,
    n_trials: int = 100,
    seed: int = 42,
    objective_metric: str = "mape",
    prior_names: tuple[str, ...] | list[str] | None = None,
):
    """Optimize point hyperparameters on one or many aligned holdout splits."""

    import optuna

    effective_prior_names = validate_prior_names(prior_names)
    holdouts = _normalize_holdout_collection(train_series, calibration_series)
    minimum_train_length = min(len(train) for _, train, _ in holdouts)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_s3_point_params(
            trial,
            train_length=minimum_train_length,
            seasonal_period=seasonal_period,
            prior_names=effective_prior_names,
        )
        try:
            model_params, prior_name, prior_params = split_model_prior_params(params)
            trial.set_user_attr("prior_name", prior_name)
            trial.set_user_attr("prior_params", prior_params)
            series_scores = {}
            for series_id, train, calibration in holdouts:
                model = S3Forecaster(horizon=1, **model_params)
                result = evaluate_rolling_model(
                    model,
                    train,
                    calibration,
                    seasonal_period=seasonal_period,
                )
                score = float(result["metrics"][objective_metric])
                if not np.isfinite(score):
                    return float("inf")
                series_scores[series_id] = score

            trial.set_user_attr("series_scores", series_scores)
            return float(np.median(list(series_scores.values())))
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    hpo_series_ids = [series_id for series_id, _, _ in holdouts]
    study.set_user_attr("prior_names", list(effective_prior_names))
    study.set_user_attr("hpo_series_ids", hpo_series_ids)
    study.set_user_attr("n_hpo_series", len(hpo_series_ids))
    study.set_user_attr("aggregation", "median")
    study.set_user_attr("objective_metric", objective_metric)
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    study.set_user_attr("cross_validation", False)
    study.set_user_attr(
        "holdout_protocol",
        "multi_series_train_calibration" if len(holdouts) > 1 else "train_calibration",
    )
    return study


def optimize_s3_uq(
    train_series: Any,
    point_params: dict,
    *,
    seasonal_period: int = 12,
    n_folds: int = 3,
    validation_size: int = 6,
    n_trials: int = 100,
    target_coverage: float = 0.90,
    penalty_strength: float = 100.0,
    seed: int = 42,
    optimize_nominal_level: bool = False,
):
    """Optimize uncertainty parameters using temporal CV."""

    import optuna

    full = ensure_series(train_series)
    folds = make_expanding_window_folds(
        full,
        n_folds=n_folds,
        validation_size=validation_size,
    )
    fixed_alpha = 1.0 - float(target_coverage)

    def objective(trial: optuna.Trial) -> float:
        target_miscoverage = (
            trial.suggest_float("target_miscoverage", 0.01, 0.20)
            if optimize_nominal_level
            else fixed_alpha
        )
        aci_step_size = trial.suggest_float("aci_step_size", 0.005, 0.20, log=True)
        interval_scale = trial.suggest_float("interval_scale", 0.5, 8.0, log=True)
        min_width_factor = trial.suggest_float("min_width_factor", 0.0, 2.0)
        
        try:
            params, prior_name, prior_params = split_model_prior_params(point_params)
            params["aci_step_size"] = aci_step_size
            trial.set_user_attr("prior_name", prior_name)
            trial.set_user_attr("prior_params", prior_params)
            trial.set_user_attr("requested_nominal_coverage", target_coverage)
            trial.set_user_attr("effective_target_coverage", 1.0 - target_miscoverage)
            
            scores = []
            for fold_train, fold_val in folds:
                minimum_width = min_width_factor * seasonal_naive_scale(
                    fold_train,
                    seasonal_period=seasonal_period,
                )
                model = S3Forecaster(
                    horizon=1,
                    target_miscoverage=target_miscoverage,
                    interval_scale=interval_scale,
                    minimum_width=minimum_width,
                    **params,
                )
                result = evaluate_rolling_model(
                    model,
                    fold_train,
                    fold_val,
                    alpha=1.0 - target_coverage,
                    seasonal_period=seasonal_period,
                )
                metrics = result["metrics"]
                penalty = penalty_strength * max(0.0, target_coverage - metrics["ecp"]) ** 2
                scores.append(metrics["msis"] + penalty)
            
            final_score = float(np.median(scores))
            return final_score if np.isfinite(final_score) else float("inf")
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    return study


def optimize_s3_uq_holdout(
    train_series: Any,
    calibration_series: Any,
    point_params: dict,
    *,
    seasonal_period: int = 12,
    n_trials: int = 100,
    target_coverage: float = 0.90,
    penalty_strength: float = 100.0,
    seed: int = 42,
):
    """Optimize uncertainty parameters across one or many holdout splits."""

    import optuna

    holdouts = _normalize_holdout_collection(train_series, calibration_series)
    fixed_alpha = 1.0 - float(target_coverage)
    frozen_point_params = dict(point_params)

    def objective(trial: optuna.Trial) -> float:
        aci_step_size = trial.suggest_float("aci_step_size", 0.005, 0.20, log=True)
        interval_scale = trial.suggest_float("interval_scale", 0.5, 8.0, log=True)
        min_width_factor = trial.suggest_float("min_width_factor", 0.0, 2.0)
        try:
            params, prior_name, prior_params = split_model_prior_params(
                frozen_point_params
            )
            params["aci_step_size"] = aci_step_size
            trial.set_user_attr("prior_name", prior_name)
            trial.set_user_attr("prior_params", prior_params)
            trial.set_user_attr("requested_nominal_coverage", target_coverage)
            trial.set_user_attr("effective_target_coverage", target_coverage)

            series_scores = {}
            for series_id, train, calibration in holdouts:
                minimum_width = min_width_factor * seasonal_naive_scale(
                    train,
                    seasonal_period=seasonal_period,
                )
                model = S3Forecaster(
                    horizon=1,
                    target_miscoverage=fixed_alpha,
                    interval_scale=interval_scale,
                    minimum_width=minimum_width,
                    **params,
                )
                result = evaluate_rolling_model(
                    model,
                    train,
                    calibration,
                    alpha=fixed_alpha,
                    seasonal_period=seasonal_period,
                )
                metrics = result["metrics"]
                penalty = (
                    penalty_strength
                    * max(0.0, target_coverage - metrics["ecp"]) ** 2
                )
                score = float(metrics["msis"] + penalty)
                if not np.isfinite(score):
                    return float("inf")
                series_scores[series_id] = score

            trial.set_user_attr("series_scores", series_scores)
            return float(np.median(list(series_scores.values())))
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    hpo_series_ids = [series_id for series_id, _, _ in holdouts]
    study.set_user_attr("hpo_series_ids", hpo_series_ids)
    study.set_user_attr("n_hpo_series", len(hpo_series_ids))
    study.set_user_attr("aggregation", "median")
    study.set_user_attr("objective_metric", "msis_plus_coverage_penalty")
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    study.set_user_attr("cross_validation", False)
    study.set_user_attr("point_params_frozen", True)
    return study


def evaluate_s3_forecaster(
    train_series: Any,
    test_series: Any,
    point_params: dict,
    *,
    uq_params: Optional[dict] = None,
    seasonal_period: int = 12,
    alpha: float = 0.10,
) -> dict[str, Any]:
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    uq_params = dict(uq_params or {})

    params, _, _ = split_model_prior_params(point_params)
    if "aci_step_size" in uq_params:
        params["aci_step_size"] = uq_params["aci_step_size"]
        
    target_miscoverage = float(uq_params.get("target_miscoverage", alpha))
    target_miscoverage = float(alpha) if not bool(uq_params.get("optimize_nominal_level", False)) else target_miscoverage

    interval_scale = float(uq_params.get("interval_scale", 1.0))
    min_width_factor = float(uq_params.get("min_width_factor", 0.0))
    minimum_width = min_width_factor * seasonal_naive_scale(
        train,
        seasonal_period=seasonal_period,
    )

    model = S3Forecaster(
        horizon=1, 
        target_miscoverage=target_miscoverage, 
        interval_scale=interval_scale,
        minimum_width=minimum_width,
        **params
    )
    
    start = time.perf_counter()
    result = evaluate_rolling_model(
        model,
        train,
        test,
        seasonal_period=seasonal_period,
        alpha=alpha,
    )
    forecast = result["forecast"].copy()
    elapsed = time.perf_counter() - start

    metrics = evaluate_forecast(
        test,
        forecast["pred"],
        y_train=train,
        lower=forecast["lower"],
        upper=forecast["upper"],
        alpha=alpha,
        seasonal_period=seasonal_period,
    )
    
    metrics["elapsed_seconds"] = elapsed
    metrics["trainable_params"] = count_trainable_parameters(model)
    return {"model": model, "forecast": forecast, "metrics": metrics}


def run_s3_experiment(
    train_series: Any,
    test_series: Any,
    *,
    seasonal_period: int = 12,
    n_folds: int = 3,
    validation_size: int = 12,
    point_trials: int = 100,
    uq_trials: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    point_study = optimize_s3_forecaster(
        train_series, 
        seasonal_period=seasonal_period,
        n_folds=n_folds,
        validation_size=validation_size,
        n_trials=point_trials, 
        seed=seed
    )
    uq_study = optimize_s3_uq(
        train_series,
        point_study.best_params,
        seasonal_period=seasonal_period,
        n_folds=n_folds,
        validation_size=validation_size,
        n_trials=uq_trials,
        seed=seed,
    )
    evaluation = evaluate_s3_forecaster(
        train_series,
        test_series,
        point_study.best_params,
        uq_params=uq_study.best_params,
        seasonal_period=seasonal_period,
    )
    return {
        "point_study": point_study,
        "uq_study": uq_study,
        "best_point_params": point_study.best_params,
        "best_uq_params": uq_study.best_params,
        **evaluation,
    }
