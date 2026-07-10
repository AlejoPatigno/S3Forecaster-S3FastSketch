"""Multi-series temporal HPO with one shared configuration per collection."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .metrics import seasonal_naive_scale
from .s3_fastsketch_experiment import evaluate_fastsketch, suggest_fastsketch_point_params
from .s3_forecaster_experiment import evaluate_s3_forecaster, suggest_s3_point_params
from .temporal_cv import make_expanding_window_folds


def _aggregate_two_stage(
    scores: list[dict[str, Any]],
    aggregation: str,
) -> float:
    if not scores:
        return float("inf")
    df = pd.DataFrame(scores)
    # Median across folds per series
    series_agg = df.groupby("series_id")["score"].median()
    # Then median across series
    if aggregation == "median":
        return float(series_agg.median())
    elif aggregation == "mean":
        return float(series_agg.mean())
    else:
        return float(series_agg.median())


def optimize_collection(
    series_map: dict[str, Any],
    objective_factory: Callable[[Any, Any, dict, str, int], float],
    suggest_params: Callable[[Any, int, int], dict[str, Any]],
    *,
    seasonal_period: int = 12,
    prior_names: tuple[str, ...] | list[str] | None = None,
    n_trials: int = 20,
    n_folds: int = 3,
    validation_size: int = 6,
    aggregation: str = "median",
    max_failure_fraction: float = 0.25,
    seed: int = 42,
):
    import optuna

    fold_map = {
        str(series_id): make_expanding_window_folds(
            series,
            n_folds=n_folds,
            validation_size=validation_size,
        )
        for series_id, series in sorted(series_map.items())
    }
    
    # Calculate minimum train length to bound ar_lags safe values
    min_train_length = min(
        len(folds[0][0]) for folds in fold_map.values() if folds
    )

    details: list[dict[str, Any]] = []

    def objective(trial):
        params = suggest_params(trial, min_train_length, seasonal_period)
        scores, failures = [], []
        
        n_series = len(fold_map)
        n_folds_total = sum(len(f) for f in fold_map.values())
        
        for series_id, folds in fold_map.items():
            for fold_id, (train, validation) in enumerate(folds):
                try:
                    score = float(objective_factory(train, validation, params, series_id=series_id, fold_id=fold_id))
                    scores.append({"series_id": series_id, "fold_id": fold_id, "score": score})
                    details.append({"trial": trial.number, "series_id": series_id, "fold_id": fold_id, "score": score, "status": "ok"})
                except Exception as exc:
                    failures.append((series_id, fold_id, str(exc)))
                    details.append({"trial": trial.number, "series_id": series_id, "fold_id": fold_id, "score": np.inf, "status": "failed", "error": str(exc)})
        
        n_successes = len(scores)
        n_failures = len(failures)
        failure_fraction = n_failures / n_folds_total if n_folds_total > 0 else 1.0
        
        trial.set_user_attr("n_series", n_series)
        trial.set_user_attr("n_folds_total", n_folds_total)
        trial.set_user_attr("n_successes", n_successes)
        trial.set_user_attr("n_failures", n_failures)
        trial.set_user_attr("failure_fraction", failure_fraction)
        trial.set_user_attr("failed_series", list({f[0] for f in failures}))
        trial.set_user_attr("prior_name", params.get("prior_name"))
        
        if n_successes == 0 or failure_fraction > max_failure_fraction:
            return float("inf")
            
        return _aggregate_two_stage(scores, aggregation)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    study.set_user_attr("series_fold_scores", details)
    study.set_user_attr("aggregation", aggregation)
    study.set_user_attr("one_configuration_per_collection", True)
    return study


def optimize_s3_collection(series_map: dict[str, Any], **kwargs):
    def suggest(trial, min_train, sp):
        return suggest_s3_point_params(
            trial, 
            train_length=min_train, 
            seasonal_period=sp,
            prior_names=kwargs.get("prior_names")
        )

    def evaluate(train, validation, params, *, series_id, fold_id):
        return evaluate_s3_forecaster(
            train, 
            validation, 
            params, 
            seasonal_period=kwargs.get("seasonal_period", 12)
        )["metrics"]["mase"]

    return optimize_collection(series_map, evaluate, suggest, **kwargs)


def optimize_fastsketch_collection(series_map: dict[str, Any], **kwargs):
    def suggest(trial, min_train, sp):
        return suggest_fastsketch_point_params(
            trial, 
            train_length=min_train, 
            seasonal_period=sp,
            prior_names=kwargs.get("prior_names")
        )

    def evaluate(train, validation, params, *, series_id, fold_id):
        return evaluate_fastsketch(
            train, 
            validation, 
            params, 
            seasonal_period=kwargs.get("seasonal_period", 12)
        )["metrics"]["mase"]

    return optimize_collection(series_map, evaluate, suggest, **kwargs)


def optimize_uq_collection(
    series_map: dict[str, Any],
    frozen_point_params: dict,
    evaluate_fn: Callable,
    *,
    seasonal_period: int = 12,
    target_coverage: float = 0.90,
    n_trials: int = 20,
    n_folds: int = 3,
    validation_size: int = 6,
    aggregation: str = "median",
    max_failure_fraction: float = 0.25,
    penalty_strength: float = 100.0,
    seed: int = 42,
):
    import optuna

    fold_map = {
        str(series_id): make_expanding_window_folds(
            series,
            n_folds=n_folds,
            validation_size=validation_size,
        )
        for series_id, series in sorted(series_map.items())
    }
    
    fixed_alpha = 1.0 - float(target_coverage)

    def objective(trial):
        aci_step_size = trial.suggest_float("aci_step_size", 0.005, 0.20, log=True)
        interval_scale = trial.suggest_float("interval_scale", 0.5, 8.0, log=True)
        min_width_factor = trial.suggest_float("min_width_factor", 0.0, 2.0)

        uq_params = {
            "target_miscoverage": fixed_alpha,
            "aci_step_size": aci_step_size,
            "interval_scale": interval_scale,
            "min_width_factor": min_width_factor,
        }
        
        scores, failures = [], []
        
        for series_id, folds in fold_map.items():
            for fold_id, (train, validation) in enumerate(folds):
                try:
                    result = evaluate_fn(
                        train, 
                        validation, 
                        frozen_point_params, 
                        uq_params=uq_params,
                        seasonal_period=seasonal_period,
                        alpha=fixed_alpha,
                    )
                    metrics = result["metrics"]
                    penalty = penalty_strength * max(0.0, target_coverage - metrics["ecp"]) ** 2
                    score = float(metrics["msis"] + penalty)
                    scores.append({"series_id": series_id, "fold_id": fold_id, "score": score})
                except Exception as exc:
                    failures.append((series_id, fold_id, str(exc)))
        
        n_folds_total = sum(len(f) for f in fold_map.values())
        failure_fraction = len(failures) / n_folds_total if n_folds_total > 0 else 1.0
        
        if len(scores) == 0 or failure_fraction > max_failure_fraction:
            return float("inf")
            
        return _aggregate_two_stage(scores, aggregation)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    return study


def optimize_s3_uq_collection(
    series_map: dict[str, Any],
    frozen_point_params: dict,
    **kwargs
):
    return optimize_uq_collection(
        series_map, 
        frozen_point_params, 
        evaluate_s3_forecaster, 
        **kwargs
    )


def optimize_fastsketch_uq_collection(
    series_map: dict[str, Any],
    frozen_point_params: dict,
    **kwargs
):
    return optimize_uq_collection(
        series_map, 
        frozen_point_params, 
        evaluate_fastsketch, 
        **kwargs
    )
