"""Single-development-series HPO and transfer evaluation protocol."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .result_store import current_git_commit
from .s3_fastsketch_experiment import (
    evaluate_fastsketch,
    optimize_fastsketch_holdout,
    optimize_fastsketch_uq_holdout,
)
from .s3_forecaster_experiment import (
    evaluate_s3_forecaster,
    optimize_s3_forecaster_holdout,
    optimize_s3_uq_holdout,
)
from .utils import ensure_series
from .priors import validate_prior_names


MODEL_ALIASES = {
    "s3forecaster": "S3Forecaster",
    "s3-forecaster": "S3Forecaster",
    "S3Forecaster": "S3Forecaster",
    "s3fastsketch": "S3FastSketchForecaster",
    "s3-fastsketch": "S3FastSketchForecaster",
    "S3FastSketch": "S3FastSketchForecaster",
    "S3FastSketchForecaster": "S3FastSketchForecaster",
}


def _canonical_model_name(model_name: str) -> str:
    try:
        return MODEL_ALIASES[str(model_name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported model_name={model_name!r}.") from exc


def temporal_train_cal_test_split(
    series: Any,
    *,
    train_ratio: float = 0.64,
    calibration_ratio: float = 0.16,
    test_ratio: float = 0.20,
    minimum_train_size: int = 12,
    minimum_calibration_size: int = 3,
    minimum_test_size: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split a series chronologically into train, calibration, and test blocks."""

    total_ratio = float(train_ratio) + float(calibration_ratio) + float(test_ratio)
    if not np.isclose(total_ratio, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("train_ratio + calibration_ratio + test_ratio must equal 1.0.")
    if min(train_ratio, calibration_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must be positive.")

    y = ensure_series(series, name="series")
    n = len(y)
    min_total = minimum_train_size + minimum_calibration_size + minimum_test_size
    if n < min_total:
        raise ValueError(f"Series length {n} is shorter than required minimum {min_total}.")

    n_train = int(np.floor(n * train_ratio))
    n_calibration = int(np.floor(n * calibration_ratio))
    n_test = n - n_train - n_calibration

    deficits = [
        minimum_train_size - n_train,
        minimum_calibration_size - n_calibration,
        minimum_test_size - n_test,
    ]
    if any(deficit > 0 for deficit in deficits):
        raise ValueError(
            "Split ratios produced a block shorter than its minimum size: "
            f"train={n_train}, calibration={n_calibration}, test={n_test}."
        )

    train = y.iloc[:n_train].copy()
    calibration = y.iloc[n_train : n_train + n_calibration].copy()
    test = y.iloc[n_train + n_calibration :].copy()
    return train, calibration, test


def select_development_series(
    series_map: dict[str, Any],
    *,
    seed: int = 42,
    minimum_length: int = 30,
    excluded_series_ids=None,
) -> tuple[str, pd.Series, list[str], dict[str, str]]:
    """Select one eligible development series reproducibly before model metrics exist."""

    excluded = set(map(str, excluded_series_ids or []))
    eligible: list[str] = []
    exclusions: dict[str, str] = {}
    normalized: dict[str, pd.Series] = {}

    for raw_id, raw_series in sorted(series_map.items(), key=lambda item: str(item[0])):
        series_id = str(raw_id)
        if series_id in excluded:
            exclusions[series_id] = "excluded"
            continue
        try:
            series = ensure_series(raw_series, name=series_id)
        except Exception as exc:
            exclusions[series_id] = f"invalid_series: {exc}"
            continue
        if len(series) < int(minimum_length):
            exclusions[series_id] = f"too_short: {len(series)} < {int(minimum_length)}"
            continue
        eligible.append(series_id)
        normalized[series_id] = series

    if not eligible:
        raise ValueError("No eligible development series were found.")

    rng = np.random.default_rng(int(seed))
    development_series_id = eligible[int(rng.integers(0, len(eligible)))]
    return development_series_id, normalized[development_series_id].copy(), eligible, exclusions


def _model_functions(model_name: str):
    model = _canonical_model_name(model_name)
    if model == "S3Forecaster":
        return optimize_s3_forecaster_holdout, optimize_s3_uq_holdout, evaluate_s3_forecaster
    return optimize_fastsketch_holdout, optimize_fastsketch_uq_holdout, evaluate_fastsketch


def optimize_on_development_series(
    series_map: dict[str, Any],
    *,
    model_name: str,
    seasonal_period: int = 12,
    n_point_trials: int = 100,
    n_uq_trials: int = 100,
    seed: int = 42,
    target_coverage: float = 0.90,
    train_ratio: float = 0.64,
    calibration_ratio: float = 0.16,
    test_ratio: float = 0.20,
    development_series_id: str | None = None,
    prior_names: tuple[str, ...] | list[str] | None = None,
):
    optimize_point, optimize_uq, evaluate_fn = _model_functions(model_name)
    minimum_length = 30
    if development_series_id is None:
        development_series_id, development_series, eligible, exclusions = select_development_series(
            series_map,
            seed=seed,
            minimum_length=minimum_length,
        )
    else:
        development_series_id = str(development_series_id)
        development_series = ensure_series(series_map[development_series_id], name=development_series_id)
        eligible = [str(key) for key in sorted(series_map, key=str)]
        exclusions = {}

    train, calibration, test = temporal_train_cal_test_split(
        development_series,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        test_ratio=test_ratio,
    )
    effective_prior_names = validate_prior_names(prior_names)
    point_study = optimize_point(
        train,
        calibration,
        seasonal_period=seasonal_period,
        n_trials=n_point_trials,
        seed=seed,
        prior_names=effective_prior_names,
    )
    best_point_params = dict(point_study.best_params)
    uq_study = optimize_uq(
        train,
        calibration,
        best_point_params,
        seasonal_period=seasonal_period,
        n_trials=n_uq_trials,
        target_coverage=target_coverage,
        seed=seed,
    )
    best_uq_params = dict(uq_study.best_params)
    alpha = 1.0 - float(target_coverage)
    fit_history = pd.concat([train, calibration])
    development_evaluation = evaluate_fn(
        fit_history,
        test,
        best_point_params,
        uq_params=best_uq_params,
        seasonal_period=seasonal_period,
        alpha=alpha,
    )
    return {
        "development_series_id": development_series_id,
        "prior_names": effective_prior_names,
        "eligible_series_ids": eligible,
        "excluded_series": exclusions,
        "point_study": point_study,
        "uq_study": uq_study,
        "best_point_params": best_point_params,
        "best_uq_params": best_uq_params,
        "development_split": {
            "train": train,
            "calibration": calibration,
            "test": test,
        },
        "development_metrics": development_evaluation["metrics"],
        "development_forecast": development_evaluation["forecast"],
    }


def _aggregate_metrics(per_series_metrics: pd.DataFrame, failed_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric_columns = [
        column
        for column in per_series_metrics.columns
        if column != "series_id" and pd.api.types.is_numeric_dtype(per_series_metrics[column])
    ]
    for metric in numeric_columns:
        values = per_series_metrics[metric].dropna().astype(float)
        if values.empty:
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        rows.append(
            {
                "metric": metric,
                "median": float(values.median()),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n_successful_series": int(per_series_metrics["series_id"].nunique()),
                "n_failed_series": int(failed_count),
                "failure_fraction": float(failed_count / max(1, per_series_metrics["series_id"].nunique() + failed_count)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_frozen_configuration_on_collection(
    series_map: dict[str, Any],
    *,
    development_series_id,
    model_name,
    best_point_params,
    best_uq_params,
    seasonal_period=12,
    alpha=0.10,
    train_ratio=0.64,
    calibration_ratio=0.16,
    test_ratio=0.20,
):
    _, _, evaluate_fn = _model_functions(model_name)
    per_series: list[dict[str, Any]] = []
    forecasts: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    frozen_point = dict(best_point_params)
    frozen_uq = dict(best_uq_params)
    for series_id, raw_series in sorted(series_map.items(), key=lambda item: str(item[0])):
        series_id = str(series_id)
        if series_id == str(development_series_id):
            continue
        try:
            train, calibration, test = temporal_train_cal_test_split(
                raw_series,
                train_ratio=train_ratio,
                calibration_ratio=calibration_ratio,
                test_ratio=test_ratio,
            )
            fit_history = pd.concat([train, calibration])
            result = evaluate_fn(
                fit_history,
                test,
                dict(frozen_point),
                uq_params=dict(frozen_uq),
                seasonal_period=seasonal_period,
                alpha=alpha,
            )
            per_series.append(
                {
                    "series_id": series_id,
                    **result["metrics"],
                    "point_params_json": json.dumps(frozen_point, sort_keys=True, default=str),
                    "uq_params_json": json.dumps(frozen_uq, sort_keys=True, default=str),
                }
            )
            forecast = result["forecast"].copy()
            forecast["series_id"] = series_id
            forecasts.append(forecast)
        except Exception as exc:
            failures.append({"series_id": series_id, "error": str(exc)})

    per_series_metrics = pd.DataFrame(per_series)
    failed_series = pd.DataFrame(failures, columns=["series_id", "error"])
    aggregate_metrics = _aggregate_metrics(per_series_metrics, len(failures))
    forecast_frame = pd.concat(forecasts) if forecasts else pd.DataFrame()
    return {
        "per_series_metrics": per_series_metrics,
        "aggregate_metrics": aggregate_metrics,
        "forecasts": forecast_frame,
        "failed_series": failed_series,
        "aggregate_series_ids": per_series_metrics["series_id"].tolist() if "series_id" in per_series_metrics else [],
    }


def _study_to_frame(study) -> pd.DataFrame:
    try:
        return study.trials_dataframe()
    except Exception:
        return pd.DataFrame()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def save_single_series_transfer_results(result: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "metadata": output / "metadata.json",
        "best_point_params": output / "best_point_params.json",
        "best_uq_params": output / "best_uq_params.json",
        "development_series": output / "development_series.json",
        "per_series_metrics": output / "per_series_metrics.csv",
        "aggregate_metrics": output / "aggregate_metrics.csv",
        "forecasts": output / "forecasts.parquet",
        "failed_series": output / "failed_series.csv",
        "point_study": output / "point_study.csv",
        "uq_study": output / "uq_study.csv",
    }
    _write_json(paths["metadata"], result["metadata"])
    _write_json(paths["best_point_params"], result["best_point_params"])
    _write_json(paths["best_uq_params"], result["best_uq_params"])
    _write_json(
        paths["development_series"],
        {
            "development_series_id": result["development_series_id"],
            "eligible_series_ids": result.get("eligible_series_ids", []),
            "excluded_series": result.get("excluded_series", {}),
        },
    )
    result["evaluation_results"]["per_series_metrics"].to_csv(paths["per_series_metrics"], index=False)
    result["aggregate_metrics"].to_csv(paths["aggregate_metrics"], index=False)
    result["evaluation_results"]["failed_series"].to_csv(paths["failed_series"], index=False)
    _study_to_frame(result["point_study"]).to_csv(paths["point_study"], index=False)
    _study_to_frame(result["uq_study"]).to_csv(paths["uq_study"], index=False)
    forecasts = result["evaluation_results"]["forecasts"]
    if len(forecasts):
        forecasts.to_parquet(paths["forecasts"], index=True)
    else:
        pd.DataFrame().to_parquet(paths["forecasts"], index=False)
    return paths


def run_single_series_hpo_transfer_experiment(
    series_map,
    *,
    model_name,
    dataset_name,
    seasonal_period=12,
    development_series_id=None,
    seed=42,
    n_point_trials=100,
    n_uq_trials=100,
    target_coverage=0.90,
    train_ratio=0.64,
    calibration_ratio=0.16,
    test_ratio=0.20,
    git_commit=None,
    output_dir: str | Path | None = None,
    prior_names: tuple[str, ...] | list[str] | None = None,
):
    model_name = _canonical_model_name(model_name)
    hpo = optimize_on_development_series(
        series_map,
        model_name=model_name,
        seasonal_period=seasonal_period,
        n_point_trials=n_point_trials,
        n_uq_trials=n_uq_trials,
        seed=seed,
        target_coverage=target_coverage,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        test_ratio=test_ratio,
        development_series_id=development_series_id,
        prior_names=prior_names,
    )
    alpha = 1.0 - float(target_coverage)
    evaluation = evaluate_frozen_configuration_on_collection(
        series_map,
        development_series_id=hpo["development_series_id"],
        model_name=model_name,
        best_point_params=hpo["best_point_params"],
        best_uq_params=hpo["best_uq_params"],
        seasonal_period=seasonal_period,
        alpha=alpha,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        test_ratio=test_ratio,
    )
    commit = git_commit or current_git_commit()
    metadata = {
        "dataset": dataset_name,
        "model": model_name,
        "development_series_id": hpo["development_series_id"],
        "selection_seed": int(seed),
        "git_commit": commit,
        "split_ratios": {
            "train": float(train_ratio),
            "calibration": float(calibration_ratio),
            "test": float(test_ratio),
        },
        "cross_validation": False,
        "one_hpo_series_per_dataset": True,
        "hpo_repeated_per_series": False,
        "prior_names": list(hpo["prior_names"]),
        "n_point_trials": int(n_point_trials),
        "n_uq_trials": int(n_uq_trials),
        "target_coverage": float(target_coverage),
        "calibration_strategy": "fit final evaluation on train+calibration; HPO objective uses calibration only; test remains isolated",
    }
    result = {
        "dataset": dataset_name,
        "model": model_name,
        "development_series_id": hpo["development_series_id"],
        "eligible_series_ids": hpo["eligible_series_ids"],
        "excluded_series": hpo["excluded_series"],
        "point_study": hpo["point_study"],
        "uq_study": hpo["uq_study"],
        "best_point_params": hpo["best_point_params"],
        "best_uq_params": hpo["best_uq_params"],
        "development_metrics": hpo["development_metrics"],
        "evaluation_results": evaluation,
        "aggregate_metrics": evaluation["aggregate_metrics"],
        "metadata": metadata,
    }
    if output_dir is not None:
        result["artifact_paths"] = save_single_series_transfer_results(result, output_dir)
    return result


__all__ = [
    "temporal_train_cal_test_split",
    "select_development_series",
    "optimize_on_development_series",
    "evaluate_frozen_configuration_on_collection",
    "run_single_series_hpo_transfer_experiment",
    "save_single_series_transfer_results",
]
