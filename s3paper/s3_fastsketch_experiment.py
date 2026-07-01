"""Hyperparameter optimization and final testing for S3-FastSketch."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from .metrics import evaluate_forecast, seasonal_naive_scale
from .s3_fastsketch import S3FastSketchForecaster
from .utils import count_trainable_parameters, ensure_series


POINT_KEYS = {
    "foundation_window",
    "ar_lags",
    "ema_spans",
    "conv_scales",
    "use_calendar",
    "ridge_alpha",
    "shrinkage_max",
    "oob_split_ratio",
}


def _holdout(series: Any, val_size: int):
    y = ensure_series(series)
    if len(y) <= val_size + 15:
        val_size = max(3, len(y) // 4)
    return y.iloc[:-val_size], y.iloc[-val_size:]


def evaluate_fastsketch_on_holdout(
    train_subset: Any,
    val_subset: Any,
    params: dict,
    *,
    eps: float = 1e-5,
    return_objects: bool = False,
):
    train = ensure_series(train_subset)
    validation = ensure_series(val_subset)
    model = S3FastSketchForecaster(
        horizon=1,
        seasonal_period=12,
        aci_target=0.10,
        aci_step_size=0.05,
        min_train_samples=6,
        min_calib_samples=2,
        **params,
    )
    model.fit(train)
    if not model.fitted:
        raise RuntimeError(model.fit_report_)
    forecast = model.predict(steps=len(validation), recursive=True)
    metrics = evaluate_forecast(validation, forecast["pred"], eps=eps)
    if return_objects:
        return metrics["mape"], forecast, model
    return metrics["mape"]


def optimize_fastsketch_point(
    train_series: Any,
    *,
    val_size: int = 12,
    n_trials: int = 100,
    seed: int = 42,
):
    import optuna

    train, validation = _holdout(train_series, val_size)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "foundation_window": trial.suggest_int("foundation_window", 2, 12),
            "ar_lags": trial.suggest_int("ar_lags", 1, 6),
            "ema_spans": trial.suggest_categorical(
                "ema_spans",
                [(2, 4), (2, 4, 8), (3, 6), (3, 6, 12), (2, 3, 5, 8)],
            ),
            "conv_scales": trial.suggest_categorical(
                "conv_scales",
                [(2, 3), (2, 3, 4), (3, 6), (3, 6, 12), (2, 4, 8)],
            ),
            "use_calendar": trial.suggest_categorical("use_calendar", [False, True]),
            "ridge_alpha": trial.suggest_float("ridge_alpha", 1e-3, 100.0, log=True),
            "shrinkage_max": trial.suggest_float("shrinkage_max", 0.25, 3.0),
            "oob_split_ratio": trial.suggest_float("oob_split_ratio", 0.55, 0.80),
        }
        try:
            score = evaluate_fastsketch_on_holdout(train, validation, params)
            return float(score) if np.isfinite(score) else float("inf")
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    return study


def optimize_fastsketch_uq(
    train_series: Any,
    point_params: dict,
    *,
    val_size: int = 12,
    n_trials: int = 100,
    target_coverage: float = 0.90,
    seasonal_period: int = 12,
    penalty_strength: float = 100.0,
    seed: int = 42,
):
    """Optimize UQ on a validation block; the test set is never accessed."""

    import optuna

    train, validation = _holdout(train_series, val_size)
    point_params = {key: value for key, value in point_params.items() if key in POINT_KEYS}

    def objective(trial: optuna.Trial) -> float:
        aci_target = trial.suggest_float("aci_target", 0.01, 0.20)
        aci_step_size = trial.suggest_float("aci_step_size", 0.005, 0.20, log=True)
        interval_scale = trial.suggest_float("interval_scale", 0.75, 8.0, log=True)
        interval_power = trial.suggest_float("interval_power", 0.0, 1.25)
        min_width_factor = trial.suggest_float("min_width_factor", 0.0, 1.50)

        try:
            model = S3FastSketchForecaster(
                horizon=1,
                seasonal_period=seasonal_period,
                aci_target=aci_target,
                aci_step_size=aci_step_size,
                min_train_samples=6,
                min_calib_samples=2,
                **point_params,
            )
            model.fit(train)
            if not model.fitted:
                return float("inf")
            minimum_width = min_width_factor * seasonal_naive_scale(
                train, seasonal_period=seasonal_period
            )
            forecast = model.predict(
                steps=len(validation),
                recursive=True,
                interval_scale=interval_scale,
                interval_power=interval_power,
                min_width=minimum_width,
            )
            metrics = evaluate_forecast(
                validation,
                forecast["pred"],
                y_train=train,
                lower=forecast["lower"],
                upper=forecast["upper"],
                alpha=1.0 - target_coverage,
                seasonal_period=seasonal_period,
            )
            penalty = penalty_strength * max(0.0, target_coverage - metrics["ecp"]) ** 2
            for key in ("ecp", "msis", "mean_width", "mape"):
                trial.set_user_attr(key, metrics[key])
            return float(metrics["msis"] + penalty)
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    return study


def evaluate_fastsketch(
    train_series: Any,
    test_series: Any,
    point_params: dict,
    *,
    uq_params: Optional[dict] = None,
    seasonal_period: int = 12,
    alpha: float = 0.10,
):
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    point_params = {key: value for key, value in point_params.items() if key in POINT_KEYS}
    uq = dict(uq_params or {})

    model = S3FastSketchForecaster(
        horizon=1,
        seasonal_period=seasonal_period,
        aci_target=float(uq.get("aci_target", alpha)),
        aci_step_size=float(uq.get("aci_step_size", 0.05)),
        min_train_samples=6,
        min_calib_samples=2,
        **point_params,
    )
    minimum_width = float(uq.get("min_width_factor", 0.0)) * seasonal_naive_scale(
        train, seasonal_period=seasonal_period
    )

    start = time.perf_counter()
    model.fit(train)
    forecast = model.predict(
        steps=len(test),
        recursive=True,
        interval_scale=float(uq.get("interval_scale", 1.0)),
        interval_power=float(uq.get("interval_power", 0.5)),
        min_width=minimum_width,
    )
    elapsed = time.perf_counter() - start

    metrics = evaluate_forecast(
        test,
        forecast["pred"],
        y_train=train,
        lower=forecast["lower"],
        upper=forecast["upper"],
        alpha=alpha,
        seasonal_period=seasonal_period,
        elapsed_seconds=elapsed,
        trainable_params=count_trainable_parameters(model),
    )
    return {"model": model, "forecast": forecast, "metrics": metrics}


def run_fastsketch_experiment(
    train_series: Any,
    test_series: Any,
    *,
    point_trials: int = 100,
    uq_trials: int = 100,
    val_size: int = 12,
    seed: int = 42,
):
    point_study = optimize_fastsketch_point(
        train_series, val_size=val_size, n_trials=point_trials, seed=seed
    )
    uq_study = optimize_fastsketch_uq(
        train_series,
        point_study.best_params,
        val_size=val_size,
        n_trials=uq_trials,
        seed=seed,
    )
    evaluation = evaluate_fastsketch(
        train_series,
        test_series,
        point_study.best_params,
        uq_params=uq_study.best_params,
    )
    return {
        "point_study": point_study,
        "uq_study": uq_study,
        "best_point_params": point_study.best_params,
        "best_uq_params": uq_study.best_params,
        **evaluation,
    }
