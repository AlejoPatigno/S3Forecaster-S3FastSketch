"""Multi-series temporal HPO with one shared configuration per collection."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .s3_fastsketch_experiment import optimize_fastsketch_point
from .s3_forecaster_experiment import optimize_s3_forecaster
from .temporal_cv import aggregate_scores, make_expanding_window_folds


def optimize_collection(
    series_map: dict[str, Any],
    objective_factory: Callable[[Any, Any], float],
    suggest_params: Callable[[Any], dict[str, Any]],
    *,
    n_trials: int = 20,
    n_folds: int = 3,
    validation_size: int = 6,
    aggregation: str = "median",
    max_failure_fraction: float = 0.25,
    seed: int = 42,
):
    import optuna

    details: list[dict[str, Any]] = []

    def objective(trial):
        params = suggest_params(trial)
        scores, failures = [], []
        for series_id, series in series_map.items():
            folds = make_expanding_window_folds(
                series,
                n_folds=n_folds,
                validation_size=validation_size,
            )
            for fold_id, (train, validation) in enumerate(folds):
                try:
                    score = float(objective_factory(train, validation, params))
                    scores.append(score)
                    details.append({"trial": trial.number, "series_id": series_id, "fold_id": fold_id, "score": score, "status": "ok"})
                except Exception as exc:
                    failures.append((series_id, fold_id, str(exc)))
                    details.append({"trial": trial.number, "series_id": series_id, "fold_id": fold_id, "score": np.inf, "status": "failed", "error": str(exc)})
        total = len(scores) + len(failures)
        failure_fraction = len(failures) / total if total else 1.0
        trial.set_user_attr("scores", scores)
        trial.set_user_attr("failures", failures)
        trial.set_user_attr("n_successes", len(scores))
        trial.set_user_attr("n_failures", len(failures))
        trial.set_user_attr("failure_fraction", failure_fraction)
        trial.set_user_attr("prior_name", params.get("prior_name"))
        if failure_fraction > max_failure_fraction:
            return float("inf")
        return aggregate_scores(scores, method=aggregation)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    study.set_user_attr("series_fold_scores", details)
    study.set_user_attr("aggregation", aggregation)
    study.set_user_attr("one_configuration_per_collection", True)
    return study


def optimize_s3_collection(series_map: dict[str, Any], **kwargs):
    def suggest(trial):
        sample = next(iter(series_map.values()))
        study = optimize_s3_forecaster(sample, n_trials=1, **{k: v for k, v in kwargs.items() if k in {"seasonal_period", "prior_names"}})
        return dict(study.trials[0].params)

    def evaluate(train, validation, params):
        from .s3_forecaster_experiment import evaluate_s3_forecaster

        return evaluate_s3_forecaster(train, validation, params, seasonal_period=kwargs.get("seasonal_period", 12))["metrics"]["mase"]

    return optimize_collection(series_map, evaluate, suggest, **{k: v for k, v in kwargs.items() if k in {"n_trials", "n_folds", "validation_size", "aggregation", "max_failure_fraction", "seed"}})


def optimize_fastsketch_collection(series_map: dict[str, Any], **kwargs):
    def suggest(trial):
        sample = next(iter(series_map.values()))
        study = optimize_fastsketch_point(sample, n_trials=1, **{k: v for k, v in kwargs.items() if k in {"seasonal_period", "prior_names"}})
        return dict(study.trials[0].params)

    def evaluate(train, validation, params):
        from .s3_fastsketch_experiment import evaluate_fastsketch

        return evaluate_fastsketch(train, validation, params, seasonal_period=kwargs.get("seasonal_period", 12))["metrics"]["mase"]

    return optimize_collection(series_map, evaluate, suggest, **{k: v for k, v in kwargs.items() if k in {"n_trials", "n_folds", "validation_size", "aggregation", "max_failure_fraction", "seed"}})
