"""Hyperparameter optimization and final testing for S3-Forecaster."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from .metrics import evaluate_forecast
from .s3_forecaster import S3Forecaster
from .utils import count_trainable_parameters, ensure_series


def temporal_holdout(series: Any, val_size: int) -> tuple[pd.Series, pd.Series]:
    y = ensure_series(series)
    val_size = int(val_size)
    if val_size <= 0 or len(y) <= val_size + 15:
        raise ValueError("Insufficient observations for the requested temporal holdout.")
    return y.iloc[:-val_size], y.iloc[-val_size:]


def optimize_s3_forecaster(
    train_series: Any,
    *,
    val_size: int = 12,
    n_trials: int = 100,
    seed: int = 42,
    objective_metric: str = "mape",
):
    """Optimize point-forecast hyperparameters on a chronological validation block."""

    import optuna

    full = ensure_series(train_series)
    if len(full) <= val_size + 15:
        val_size = max(3, len(full) // 4)
    train, validation = temporal_holdout(full, val_size)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "reservoir_size": trial.suggest_int("reservoir_size", 1, 200),
            "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.5),
            "ar_lags": trial.suggest_int("ar_lags", 2, min(12, max(2, len(train) // 6))),
            "foundation_window": trial.suggest_int(
                "foundation_window", 3, min(24, max(3, len(train) // 4))
            ),
            "regressor_type": trial.suggest_categorical(
                "regressor_type", ["Ridge", "ElasticNet", "Lasso"]
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 50.0, log=True),
            "aci_step_size": trial.suggest_float("aci_step_size", 0.01, 0.2),
            "oob_split_ratio": trial.suggest_float("oob_split_ratio", 0.60, 0.85),
        }
        try:
            model = S3Forecaster(horizon=len(validation), **params)
            model.fit(train)
            forecast = model.predict()
            metrics = evaluate_forecast(validation, forecast["pred"], eps=1e-5)
            score = metrics[objective_metric]
            return float(score) if np.isfinite(score) else float("inf")
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    return study


def optimize_s3_uq(
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
    """Optimize uncertainty parameters using only train/validation data."""

    import optuna

    full = ensure_series(train_series)
    if len(full) <= val_size + 15:
        val_size = max(3, len(full) // 4)
    train, validation = temporal_holdout(full, val_size)

    def objective(trial: optuna.Trial) -> float:
        target_miscoverage = trial.suggest_float("target_miscoverage", 0.01, 0.20)
        aci_step_size = trial.suggest_float("aci_step_size", 0.005, 0.20, log=True)
        interval_scale = trial.suggest_float("interval_scale", 0.5, 8.0, log=True)
        try:
            params = dict(point_params)
            params["aci_step_size"] = aci_step_size
            model = S3Forecaster(
                horizon=len(validation),
                target_miscoverage=target_miscoverage,
                **params,
            )
            model.fit(train)
            forecast = model.predict().copy()
            center = forecast["pred"].to_numpy()
            half_width = 0.5 * (
                forecast["upper"].to_numpy() - forecast["lower"].to_numpy()
            )
            forecast["lower"] = center - interval_scale * half_width
            forecast["upper"] = center + interval_scale * half_width
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
            trial.set_user_attr("ecp", metrics["ecp"])
            trial.set_user_attr("msis", metrics["msis"])
            return float(metrics["msis"] + penalty)
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
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

    params = dict(point_params)
    if "aci_step_size" in uq_params:
        params["aci_step_size"] = uq_params["aci_step_size"]
    target_miscoverage = float(uq_params.get("target_miscoverage", alpha))

    model = S3Forecaster(
        horizon=len(test), target_miscoverage=target_miscoverage, **params
    )
    start = time.perf_counter()
    model.fit(train)
    forecast = model.predict().copy()
    elapsed = time.perf_counter() - start

    interval_scale = float(uq_params.get("interval_scale", 1.0))
    if interval_scale != 1.0:
        center = forecast["pred"].to_numpy()
        half_width = 0.5 * (
            forecast["upper"].to_numpy() - forecast["lower"].to_numpy()
        )
        forecast["lower"] = center - interval_scale * half_width
        forecast["upper"] = center + interval_scale * half_width

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


def run_s3_experiment(
    train_series: Any,
    test_series: Any,
    *,
    point_trials: int = 100,
    uq_trials: int = 100,
    val_size: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    point_study = optimize_s3_forecaster(
        train_series, val_size=val_size, n_trials=point_trials, seed=seed
    )
    uq_study = optimize_s3_uq(
        train_series,
        point_study.best_params,
        val_size=val_size,
        n_trials=uq_trials,
        seed=seed,
    )
    evaluation = evaluate_s3_forecaster(
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
