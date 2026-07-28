# ============================================================
# Archivo sugerido:
# s3paper/kaggle_workflow.py
# ============================================================

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from s3paper.baselines import ConformalizedBaseline
from s3paper.diagnostics import diagnostics_frame
from s3paper.metrics import seasonal_naive_scale
from s3paper.multiseries_hpo import (
    optimize_fastsketch_collection,
    optimize_fastsketch_uq_collection,
    optimize_s3_collection,
    optimize_s3_uq_collection,
)
from s3paper.priors import split_model_prior_params
from s3paper.result_store import (
    current_git_commit,
    export_paper_tables,
    per_series_metrics_from_store,
    result_rows_from_forecast,
    save_result_store,
    validate_result_store,
)
from s3paper.rolling_protocol import evaluate_rolling_model
from s3paper.s3_fastsketch_experiment import evaluate_fastsketch
from s3paper.s3_forecaster_experiment import evaluate_s3_forecaster
from s3paper.single_series_transfer_hpo import run_single_series_hpo_transfer_experiment

from s3paper.kaggle_loaders import DatasetBundle


@dataclass(frozen=True)
class NotebookRunConfig:
    run_mode: str = "standard"
    seed: int = 42

    development_fraction: float = 0.30

    n_folds: int = 3
    validation_size: int = 6

    point_trials: int = 100
    uq_trials: int = 60

    target_coverage: float = 0.90

    maximum_series_smoke: int = 6
    maximum_series_standard: int = 200

    prior_names: tuple[str, ...] = (
        "causal_rolling_mean",
        "seasonal_naive",
        "ets",
        "theta",
        "chronos",
        "timesfm",
    )


def set_reproducibility(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def prepare_repository(
    *,
    repository_url: str,
    repository_ref: str,
    working_directory: str | Path = "/kaggle/working",
    repository_name: str = "S3Forecaster-S3FastSketch",
    source_directory: str | Path | None = None,
    github_token_environment: str = "GITHUB_TOKEN",
) -> tuple[Path, str]:
    """
    Copia un repositorio montado como Kaggle Dataset o clona el repositorio
    privado utilizando GITHUB_TOKEN.
    """

    working_directory = Path(working_directory)
    working_directory.mkdir(parents=True, exist_ok=True)

    repository_directory = working_directory / repository_name

    if repository_directory.exists():
        shutil.rmtree(repository_directory)

    if source_directory is not None:
        source_directory = Path(source_directory)

        if not source_directory.exists():
            raise FileNotFoundError(source_directory)

        shutil.copytree(
            source_directory,
            repository_directory,
            ignore=shutil.ignore_patterns(
                ".git",
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
            ),
        )
    else:
        token = os.environ.get(github_token_environment)

        clone_url = repository_url

        if token and repository_url.startswith("https://"):
            clone_url = repository_url.replace(
                "https://",
                f"https://x-access-token:{token}@",
                1,
            )

        subprocess.run(
            ["git", "clone", "--quiet", clone_url, str(repository_directory)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        subprocess.run(
            ["git", "fetch", "--quiet", "origin", repository_ref],
            cwd=repository_directory,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--quiet", repository_ref],
            cwd=repository_directory,
            check=True,
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "-e",
            str(repository_directory),
        ],
        check=True,
    )

    commit = current_git_commit(repository_directory)

    return repository_directory, commit


def create_run_directory(
    dataset_name: str,
    *,
    root: str | Path = "/kaggle/working/outputs",
    run_id: str | None = None,
) -> tuple[Path, str]:
    run_id = run_id or (
        pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )

    output = Path(root) / dataset_name / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()

    return output, run_id


def save_json(
    value: Any,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (set, tuple)):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=default,
        ),
        encoding="utf-8",
    )

    return path


def save_optuna_study(
    study: Any,
    output_directory: str | Path,
    name: str,
) -> dict[str, Path]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    pickle_path = output_directory / f"{name}.joblib"
    trials_path = output_directory / f"{name}_trials.csv"
    best_path = output_directory / f"{name}_best_params.json"

    joblib.dump(study, pickle_path)
    study.trials_dataframe().to_csv(trials_path, index=False)
    save_json(study.best_params, best_path)

    return {
        "study": pickle_path,
        "trials": trials_path,
        "best_params": best_path,
    }


def _stable_score(
    series_id: str,
    *,
    seed: int,
) -> int:
    digest = hashlib.sha256(
        f"{seed}:{series_id}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def limit_series_for_mode(
    bundle: DatasetBundle,
    config: NotebookRunConfig,
) -> DatasetBundle:
    mode = config.run_mode.lower().strip()

    ids = sorted(
        bundle.series_ids,
        key=lambda series_id: _stable_score(series_id, seed=config.seed),
    )

    if mode == "smoke":
        ids = ids[: config.maximum_series_smoke]
    elif mode == "standard":
        ids = ids[: config.maximum_series_standard]
    elif mode != "full":
        raise ValueError("run_mode debe ser smoke, standard o full.")

    return bundle.subset(ids)


def split_development_evaluation(
    series_ids: Sequence[str],
    *,
    development_fraction: float = 0.30,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    ids = sorted(
        {str(series_id) for series_id in series_ids},
        key=lambda series_id: _stable_score(series_id, seed=seed),
    )

    if len(ids) < 2:
        raise ValueError(
            "Se necesitan al menos dos series para separar desarrollo y evaluación."
        )

    n_development = int(round(len(ids) * float(development_fraction)))
    n_development = max(1, min(n_development, len(ids) - 1))

    development_ids = ids[:n_development]
    evaluation_ids = ids[n_development:]

    return development_ids, evaluation_ids


def build_series_map(
    bundle: DatasetBundle,
    series_ids: Sequence[str],
) -> dict[str, pd.Series]:
    return {
        str(series_id): bundle.train_series_map[str(series_id)]
        for series_id in series_ids
    }


def optimize_proposed_models_collection(
    bundle: DatasetBundle,
    development_ids: Sequence[str],
    config: NotebookRunConfig,
) -> dict[str, Any]:
    import warnings

    warnings.warn(
        "optimize_proposed_models_collection is the legacy fold-based Kaggle "
        "path. Use run_common_kaggle_transfer_pipeline for the main protocol.",
        DeprecationWarning,
        stacklevel=2,
    )
    development_map = build_series_map(bundle, development_ids)

    common = {
        "seasonal_period": bundle.seasonal_period,
        "prior_names": list(config.prior_names),
        "n_folds": config.n_folds,
        "validation_size": config.validation_size,
        "seed": config.seed,
    }

    s3_point_study = optimize_s3_collection(
        development_map,
        n_trials=config.point_trials,
        **common,
    )
    fast_point_study = optimize_fastsketch_collection(
        development_map,
        n_trials=config.point_trials,
        **common,
    )

    s3_uq_study = optimize_s3_uq_collection(
        development_map,
        s3_point_study.best_params,
        seasonal_period=bundle.seasonal_period,
        target_coverage=config.target_coverage,
        n_trials=config.uq_trials,
        n_folds=config.n_folds,
        validation_size=config.validation_size,
        seed=config.seed,
    )
    fast_uq_study = optimize_fastsketch_uq_collection(
        development_map,
        fast_point_study.best_params,
        seasonal_period=bundle.seasonal_period,
        target_coverage=config.target_coverage,
        n_trials=config.uq_trials,
        n_folds=config.n_folds,
        validation_size=config.validation_size,
        seed=config.seed,
    )

    return {
        "s3_point_study": s3_point_study,
        "s3_uq_study": s3_uq_study,
        "fast_point_study": fast_point_study,
        "fast_uq_study": fast_uq_study,
        "s3_point_params": dict(s3_point_study.best_params),
        "s3_uq_params": dict(s3_uq_study.best_params),
        "fast_point_params": dict(fast_point_study.best_params),
        "fast_uq_params": dict(fast_uq_study.best_params),
    }


def _model_status(model: Any) -> dict[str, Any]:
    if hasattr(model, "summary"):
        try:
            value = model.summary()
            if isinstance(value, Mapping):
                return dict(value)
        except Exception:
            pass

    return {
        "fitted": bool(getattr(model, "fitted", True)),
        "fit_report": getattr(model, "fit_report_", {}),
    }


def _failure_forecast(
    train: pd.Series,
    test: pd.Series,
    error: Exception,
) -> pd.DataFrame:
    target_timestamp = test.index[0]
    information_cutoff = train.index[-1]

    return pd.DataFrame(
        {
            "pred": [np.nan],
            "lower": [np.nan],
            "upper": [np.nan],
            "target": [float(test.iloc[0])],
            "forecast_origin": [information_cutoff],
            "information_cutoff": [information_cutoff],
            "target_timestamp": [target_timestamp],
            "error": [str(error)],
        },
        index=pd.DatetimeIndex([target_timestamp]),
    )


def evaluate_proposed_models_collection(
    bundle: DatasetBundle,
    evaluation_ids: Sequence[str],
    *,
    s3_point_params: dict[str, Any],
    s3_uq_params: dict[str, Any],
    fast_point_params: dict[str, Any],
    fast_uq_params: dict[str, Any],
    run_id: str,
    git_commit: str,
    seed: int = 42,
    target_coverage: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    alpha = 1.0 - float(target_coverage)

    model_specs = (
        (
            "S3-Forecaster",
            evaluate_s3_forecaster,
            s3_point_params,
            s3_uq_params,
        ),
        (
            "S3-FastSketch",
            evaluate_fastsketch,
            fast_point_params,
            fast_uq_params,
        ),
    )

    for series_id in evaluation_ids:
        train = bundle.train_series_map[str(series_id)]
        test = bundle.test_series_map[str(series_id)]

        for model_name, evaluator, point_params, uq_params in model_specs:
            started = time.perf_counter()

            try:
                result = evaluator(
                    train,
                    test,
                    point_params,
                    uq_params=uq_params,
                    seasonal_period=bundle.seasonal_period,
                    alpha=alpha,
                )

                forecast = result["forecast"].copy()
                model = result["model"]
                status = "ok"
                error = None
                model_status = _model_status(model)

            except Exception as exc:
                forecast = _failure_forecast(train, test, exc)
                status = "failed"
                error = str(exc)
                model_status = {"fitted": False, "error": error}
                model = None

            elapsed = time.perf_counter() - started

            _, prior_name, prior_params = split_model_prior_params(
                point_params
            )

            rows = result_rows_from_forecast(
                forecast,
                train_series=train,
                test_series=test,
                dataset=bundle.name,
                series_id=str(series_id),
                model=model_name,
                run_id=run_id,
                git_commit=git_commit,
                history_budget=len(train),
                split_id="evaluation",
                prior_name=prior_name,
                prior_params=prior_params,
                model_params=point_params,
                uq_params=uq_params,
                seed=seed,
                seasonal_period=bundle.seasonal_period,
                requested_coverage=target_coverage,
                effective_target_coverage=target_coverage,
                interval_method="SequentialACI",
                model_status=model_status,
                runtime_fit=elapsed,
                status=status,
                error=error,
            )
            result_frames.append(rows)

            audit_rows.append(
                {
                    "dataset": bundle.name,
                    "series_id": str(series_id),
                    "model": model_name,
                    "status": status,
                    "elapsed_seconds": elapsed,
                    "error": "" if error is None else error,
                }
            )

    store = pd.concat(result_frames, ignore_index=True)
    store = validate_result_store(store)

    return store, pd.DataFrame(audit_rows)


def default_baseline_parameters(
    seasonal_period: int = 12,
) -> dict[str, dict[str, Any]]:
    return {
        "Naive": {},
        "SeasonalNaive": {
            "seasonal_period": seasonal_period,
        },
        "Theta": {
            "period": seasonal_period,
            "deseasonalize": True,
        },
        "ETS": {
            "error": "add",
            "trend": "add",
            "seasonal": "add",
            "seasonal_periods": seasonal_period,
            "damped_trend": True,
        },
        "ARIMA": {
            "p": 1,
            "d": 1,
            "q": 1,
            "trend": "t",
        },
        "AR": {
            "lags": min(12, seasonal_period),
            "trend": "ct",
            "seasonal": False,
            "period": seasonal_period,
        },
        "KRR": {
            "input_window": min(12, seasonal_period),
            "alpha": 1.0,
            "kernel": "rbf",
            "gamma": 0.1,
        },
        "GPR": {
            "input_window": min(12, seasonal_period),
            "kernel_type": "matern32",
            "length_scale": 1.0,
            "constant_value": 1.0,
            "noise_level": 1e-2,
        },
        "NLinear": {
            "input_window": min(12, seasonal_period),
        },
        "DLinear": {
            "input_window": min(12, seasonal_period),
            "kernel_size": min(7, seasonal_period),
        },
    }


def evaluate_baselines_collection(
    bundle: DatasetBundle,
    evaluation_ids: Sequence[str],
    *,
    baseline_parameters: Mapping[str, Mapping[str, Any]],
    run_id: str,
    git_commit: str,
    seed: int = 42,
    target_coverage: float = 0.90,
    aci_step_size: float = 0.05,
    interval_scale: float = 1.0,
    minimum_width_factor: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha = 1.0 - float(target_coverage)

    result_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    for series_id in evaluation_ids:
        train = bundle.train_series_map[str(series_id)]
        test = bundle.test_series_map[str(series_id)]

        minimum_width = minimum_width_factor * seasonal_naive_scale(
            train,
            seasonal_period=bundle.seasonal_period,
        )

        for model_name, params in baseline_parameters.items():
            started = time.perf_counter()

            try:
                model = ConformalizedBaseline(
                    model_name,
                    dict(params),
                    alpha=alpha,
                    aci_step_size=aci_step_size,
                    interval_scale=interval_scale,
                    minimum_width=minimum_width,
                )

                result = evaluate_rolling_model(
                    model,
                    train,
                    test,
                    alpha=alpha,
                    seasonal_period=bundle.seasonal_period,
                )

                forecast = result["forecast"].copy()
                status = "ok"
                error = None
                model_status = _model_status(model)

            except Exception as exc:
                forecast = _failure_forecast(train, test, exc)
                status = "failed"
                error = str(exc)
                model_status = {"fitted": False, "error": error}

            elapsed = time.perf_counter() - started

            rows = result_rows_from_forecast(
                forecast,
                train_series=train,
                test_series=test,
                dataset=bundle.name,
                series_id=str(series_id),
                model=model_name,
                run_id=run_id,
                git_commit=git_commit,
                history_budget=len(train),
                split_id="evaluation",
                model_params=dict(params),
                uq_params={
                    "aci_step_size": aci_step_size,
                    "interval_scale": interval_scale,
                    "minimum_width_factor": minimum_width_factor,
                },
                seed=seed,
                seasonal_period=bundle.seasonal_period,
                requested_coverage=target_coverage,
                effective_target_coverage=target_coverage,
                interval_method="CommonSequentialACI",
                model_status=model_status,
                runtime_fit=elapsed,
                status=status,
                error=error,
            )
            result_frames.append(rows)

            audit_rows.append(
                {
                    "dataset": bundle.name,
                    "series_id": str(series_id),
                    "model": model_name,
                    "status": status,
                    "elapsed_seconds": elapsed,
                    "error": "" if error is None else error,
                }
            )

    store = pd.concat(result_frames, ignore_index=True)
    store = validate_result_store(store)

    return store, pd.DataFrame(audit_rows)


def export_notebook_run(
    *,
    bundle: DatasetBundle,
    config: NotebookRunConfig,
    output_directory: str | Path,
    run_id: str,
    git_commit: str,
    development_ids: Sequence[str],
    evaluation_ids: Sequence[str],
    studies: Mapping[str, Any],
    proposed_store: pd.DataFrame,
    baseline_store: pd.DataFrame,
    proposed_audit: pd.DataFrame,
    baseline_audit: pd.DataFrame,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    complete_store = pd.concat(
        [proposed_store, baseline_store],
        ignore_index=True,
    )
    complete_store = validate_result_store(complete_store)

    paths: dict[str, Path] = {}

    paths["result_store"] = save_result_store(
        complete_store,
        output / "per_origin_forecasts.parquet",
    )

    per_series = per_series_metrics_from_store(
        complete_store,
        seasonal_period=bundle.seasonal_period,
        alpha=1.0 - config.target_coverage,
    )
    per_series_path = output / "per_series_metrics.parquet"
    per_series.to_parquet(per_series_path, index=False)
    paths["per_series_metrics"] = per_series_path

    aggregate = (
        per_series.groupby(["dataset", "model"], dropna=False)
        .agg(
            n_series=("series_id", "nunique"),
            mase_mean=("mase", "mean"),
            mase_median=("mase", "median"),
            mase_std=("mase", "std"),
            smape_mean=("smape_percent", "mean"),
            smape_median=("smape_percent", "median"),
            rmse_mean=("rmse", "mean"),
            rmse_median=("rmse", "median"),
            msis_mean=("msis", "mean"),
            msis_median=("msis", "median"),
            ecp_mean=("ecp", "mean"),
        )
        .reset_index()
    )
    aggregate_path = output / "aggregate_metrics.csv"
    aggregate.to_csv(aggregate_path, index=False)
    paths["aggregate_metrics"] = aggregate_path

    diagnostics = diagnostics_frame(
        {
            series_id: bundle.train_series_map[series_id]
            for series_id in evaluation_ids
        },
        dataset=bundle.name,
        seasonal_period=bundle.seasonal_period,
    )
    diagnostics_path = output / "diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    paths["diagnostics"] = diagnostics_path

    audit = pd.concat(
        [proposed_audit, baseline_audit],
        ignore_index=True,
    )
    audit_path = output / "model_audit.csv"
    audit.to_csv(audit_path, index=False)
    paths["model_audit"] = audit_path

    pd.DataFrame({"series_id": development_ids}).to_csv(
        output / "development_series.csv",
        index=False,
    )
    pd.DataFrame({"series_id": evaluation_ids}).to_csv(
        output / "evaluation_series.csv",
        index=False,
    )

    bundle.metadata.to_csv(
        output / "dataset_metadata.csv",
        index=False,
    )

    save_json(
        {
            **asdict(config),
            "dataset": bundle.name,
            "run_id": run_id,
            "git_commit": git_commit,
            "n_development_series": len(development_ids),
            "n_evaluation_series": len(evaluation_ids),
        },
        output / "config.json",
    )

    save_json(
        {
            "python": sys.version,
            "platform": sys.platform,
            "git_commit": git_commit,
        },
        output / "environment.json",
    )

    for study_name, study in studies.items():
        if study_name.endswith("_study"):
            save_optuna_study(
                study,
                output,
                study_name,
            )

    paths.update(
        export_paper_tables(
            complete_store,
            output / "paper_tables",
        )
    )

    return paths


def run_common_kaggle_pipeline(
    bundle: DatasetBundle,
    *,
    config: NotebookRunConfig,
    git_commit: str,
    output_root: str | Path = "/kaggle/working/outputs",
    baseline_parameters: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    import warnings

    warnings.warn(
        "run_common_kaggle_pipeline uses the legacy development/evaluation "
        "split and fold-based HPO. Use run_common_kaggle_transfer_pipeline for "
        "the main single-development-series transfer protocol.",
        DeprecationWarning,
        stacklevel=2,
    )
    set_reproducibility(config.seed)

    bundle = limit_series_for_mode(bundle, config)

    development_ids, evaluation_ids = split_development_evaluation(
        bundle.series_ids,
        development_fraction=config.development_fraction,
        seed=config.seed,
    )

    output_directory, run_id = create_run_directory(
        bundle.name,
        root=output_root,
    )

    studies = optimize_proposed_models_collection(
        bundle,
        development_ids,
        config,
    )

    proposed_store, proposed_audit = evaluate_proposed_models_collection(
        bundle,
        evaluation_ids,
        s3_point_params=studies["s3_point_params"],
        s3_uq_params=studies["s3_uq_params"],
        fast_point_params=studies["fast_point_params"],
        fast_uq_params=studies["fast_uq_params"],
        run_id=run_id,
        git_commit=git_commit,
        seed=config.seed,
        target_coverage=config.target_coverage,
    )

    baseline_parameters = (
        dict(baseline_parameters)
        if baseline_parameters is not None
        else default_baseline_parameters(bundle.seasonal_period)
    )

    baseline_store, baseline_audit = evaluate_baselines_collection(
        bundle,
        evaluation_ids,
        baseline_parameters=baseline_parameters,
        run_id=run_id,
        git_commit=git_commit,
        seed=config.seed,
        target_coverage=config.target_coverage,
    )

    paths = export_notebook_run(
        bundle=bundle,
        config=config,
        output_directory=output_directory,
        run_id=run_id,
        git_commit=git_commit,
        development_ids=development_ids,
        evaluation_ids=evaluation_ids,
        studies=studies,
        proposed_store=proposed_store,
        baseline_store=baseline_store,
        proposed_audit=proposed_audit,
        baseline_audit=baseline_audit,
    )

    return {
        "bundle": bundle,
        "config": config,
        "run_id": run_id,
        "output_directory": output_directory,
        "development_ids": development_ids,
        "evaluation_ids": evaluation_ids,
        "studies": studies,
        "proposed_store": proposed_store,
        "baseline_store": baseline_store,
        "paths": paths,
    }


def run_common_kaggle_transfer_pipeline(
    bundle: DatasetBundle,
    *,
    config: NotebookRunConfig,
    git_commit: str,
    output_root: str | Path = "/kaggle/working/outputs",
) -> dict[str, Any]:
    """Run the protocol-compliant proposed-model Kaggle workflow.

    This path selects one development series per dataset/model, runs point and
    UQ HPO once on that series, freezes the resulting hyperparameters, and
    evaluates every other series with no per-series Optuna calls.
    """

    set_reproducibility(config.seed)
    bundle = limit_series_for_mode(bundle, config)
    output_directory, run_id = create_run_directory(
        bundle.name,
        root=output_root,
    )
    series_map = build_series_map(bundle, bundle.series_ids)

    s3_result = run_single_series_hpo_transfer_experiment(
        series_map=series_map,
        model_name="S3Forecaster",
        dataset_name=bundle.name,
        seasonal_period=bundle.seasonal_period,
        seed=config.seed,
        n_point_trials=config.point_trials,
        n_uq_trials=config.uq_trials,
        target_coverage=config.target_coverage,
        git_commit=git_commit,
        output_dir=output_directory / "s3_forecaster",
    )
    fast_result = run_single_series_hpo_transfer_experiment(
        series_map=series_map,
        model_name="S3FastSketchForecaster",
        dataset_name=bundle.name,
        seasonal_period=bundle.seasonal_period,
        seed=config.seed,
        n_point_trials=config.point_trials,
        n_uq_trials=config.uq_trials,
        target_coverage=config.target_coverage,
        git_commit=git_commit,
        output_dir=output_directory / "s3_fastsketch",
    )

    save_json(
        {
            **asdict(config),
            "dataset": bundle.name,
            "run_id": run_id,
            "git_commit": git_commit,
            "cross_validation": False,
            "one_hpo_series_per_dataset": True,
            "hpo_repeated_per_series": False,
        },
        output_directory / "config.json",
    )

    return {
        "bundle": bundle,
        "config": config,
        "run_id": run_id,
        "output_directory": output_directory,
        "s3_result": s3_result,
        "fastsketch_result": fast_result,
        "development_ids": [
            s3_result["development_series_id"],
            fast_result["development_series_id"],
        ],
        "evaluation_ids": sorted(
            set(s3_result["evaluation_results"]["aggregate_series_ids"])
            | set(fast_result["evaluation_results"]["aggregate_series_ids"])
        ),
    }
