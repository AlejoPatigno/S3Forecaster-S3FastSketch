"""Generate the M4 repository-backed reproduction notebook.

The source notebook copied model classes into the notebook. This generator
creates a smaller notebook that imports the consolidated package modules and
keeps only dataset loading, configuration, execution, and exports.
"""

from __future__ import annotations

import base64
import io
import json
import os
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "3sforecaster_m4_repository_reproduction.ipynb"
KAGGLE_NOTEBOOK_PATH = ROOT / "kaggle" / "s3forecaster_s3fastsketch_m4_cpu.ipynb"
KAGGLE_METADATA_PATH = ROOT / "kaggle" / "kernel-metadata.json"


def build_embedded_source_literal() -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "s3paper").rglob("*.py")):
            archive.write(path, path.relative_to(ROOT).as_posix())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    chunks = [encoded[index : index + 76] for index in range(0, len(encoded), 76)]
    lines = ["EMBEDDED_SOURCE_ZIP_B64 = ''.join(["]
    lines.extend(f"    {chunk!r}," for chunk in chunks)
    lines.append("])")
    return "\n".join(lines)


EMBEDDED_SOURCE_LITERAL = build_embedded_source_literal()


PACKAGE_BOOTSTRAP_SOURCE = f"""
candidate_roots = [
    Path.cwd(),
    Path.cwd().parent,
    Path("/kaggle/working"),
    Path("/kaggle/input/s3forecaster-s3fastsketch"),
    Path("/kaggle/input/s3forecaster-s3fastsketch/S3Forecaster-S3FastSketch"),
]

kaggle_input = Path("/kaggle/input")
if kaggle_input.exists():
    candidate_roots.extend(path for path in kaggle_input.iterdir() if path.is_dir())

{EMBEDDED_SOURCE_LITERAL}

PROJECT_ROOT = None
for root in candidate_roots:
    if (root / "s3paper").exists():
        PROJECT_ROOT = root
        sys.path.insert(0, str(root))
        break

if importlib.util.find_spec("s3paper") is None:
    import base64
    import io
    import zipfile

    bootstrap_root = Path("/kaggle/working/s3forecaster_embedded_source") if Path("/kaggle/working").exists() else Path(".embedded_s3paper")
    if not (bootstrap_root / "s3paper").exists():
        bootstrap_root.mkdir(parents=True, exist_ok=True)
        archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(EMBEDDED_SOURCE_ZIP_B64)))
        archive.extractall(bootstrap_root)
    sys.path.insert(0, str(bootstrap_root))
    PROJECT_ROOT = bootstrap_root

if importlib.util.find_spec("s3paper") is None:
    raise ImportError(
        "Could not find the s3paper package. Attach this repository as a "
        "Kaggle input dataset or run the notebook from the repository root."
    )

print("Project root:", PROJECT_ROOT)
"""


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # S3Forecaster-S3FastSketch M4 Reproduction

        This notebook replaces the large `3sforecaster-m4 (3).ipynb` research
        notebook with a repository-backed workflow. It keeps the same M4 Monthly
        dataset, the same first series (`M1`), the same positive-only log
        transform, and the same result families: proposed models, baselines,
        ablation, shock/non-shock analysis, data efficiency, and multi-prior
        robustness.

        Repository references:
        - `README.md` for project structure and entry points.
        - `NOTEBOOK_AUDIT.md` for the consolidation decisions.
        - `VALIDATION_REPORT.md` for local validation status.
        """
    ),
    markdown(
        """
        ## Setup

        Run this notebook on Kaggle with CPU only. If it is pushed through the
        Kaggle CLI, keep internet enabled so the M4 Monthly CSVs can be read from
        the official M4 GitHub repository. The notebook expects the repository
        source to be available either as the current working tree or as an
        attached Kaggle input dataset containing the `s3paper/` package.
        """
    ),
    code(
        """
        from pathlib import Path
        import importlib.util
        import os
        import sys

        CPU_ONLY = True
        RUN_OPTUNA = False
        RUN_OPTIONAL_DEEP_BASELINES = False
        RUN_PROPHET_PRIOR = False
        RUN_CHRONOS_BASELINE = False
        RUN_CHRONOS_PRIOR = False

        SERIES_ID = "M1"
        ROW_INDEX = 0
        SEASONAL_PERIOD = 12
        ALPHA = 0.10
        SEED = 42

        if CPU_ONLY:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/m4_repository_reproduction")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print("Output directory:", OUTPUT_DIR.resolve())
        """
    ),
    code(PACKAGE_BOOTSTRAP_SOURCE),
    code(
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from IPython.display import display

        from s3paper.ablation_study import run_s3_ablation_study
        from s3paper.baselines import evaluate_baseline
        from s3paper.data import apply_log_if_positive
        from s3paper.data_efficiency import make_data_efficiency_pivot, run_data_efficiency_analysis
        from s3paper.metrics import paper_metric_row
        from s3paper.multi_prior_robustness import default_prior_factories, run_multi_prior_robustness
        from s3paper.s3_fastsketch_experiment import evaluate_fastsketch, run_fastsketch_experiment
        from s3paper.s3_forecaster_experiment import evaluate_s3_forecaster, run_s3_experiment
        from s3paper.shock_analysis import compare_s3_models_on_shocks
        from s3paper.utils import set_global_seed

        set_global_seed(SEED)
        """
    ),
    markdown(
        """
        ## Data

        The source notebook downloaded the official M4 Monthly train/test files
        directly from `Mcompetitions/M4-methods` and selected row 0, `M1`.
        """
    ),
    code(
        """
        URL_TRAIN = "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Train/Monthly-train.csv"
        URL_TEST = "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Test/Monthly-test.csv"
        URL_INFO = "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/M4-info.csv"

        def download_m4_monthly(save_dir="m4_monthly"):
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            files = {
                "train": save_dir / "Monthly-train.csv",
                "test": save_dir / "Monthly-test.csv",
                "info": save_dir / "M4-info.csv",
            }
            if not files["train"].exists():
                pd.read_csv(URL_TRAIN).to_csv(files["train"], index=False)
            if not files["test"].exists():
                pd.read_csv(URL_TEST).to_csv(files["test"], index=False)
            if not files["info"].exists():
                pd.read_csv(URL_INFO).to_csv(files["info"], index=False)
            return files

        def extract_series_by_id(frame, series_id):
            row = frame[frame.iloc[:, 0] == series_id]
            if row.empty:
                raise ValueError(f"Series {series_id!r} was not found.")
            return pd.to_numeric(row.iloc[0, 1:], errors="coerce").dropna().to_numpy(dtype=float)

        data_files = download_m4_monthly()
        train_df = pd.read_csv(data_files["train"])
        test_df = pd.read_csv(data_files["test"])
        info_df = pd.read_csv(data_files["info"])

        if SERIES_ID is None:
            SERIES_ID = str(train_df.iloc[ROW_INDEX, 0])

        y_train_raw = extract_series_by_id(train_df, SERIES_ID)
        y_test_raw = extract_series_by_id(test_df, SERIES_ID)

        print("Shape train:", train_df.shape)
        print("Shape test:", test_df.shape)
        print("Series:", SERIES_ID)
        print("Train length:", len(y_train_raw), "Test length:", len(y_test_raw))
        print("First 10 train values:", y_train_raw[:10])
        print("First 5 test values:", y_test_raw[:5])
        """
    ),
    code(
        """
        train_input = pd.Series(y_train_raw, name=SERIES_ID)
        test_input = pd.Series(y_test_raw, name=SERIES_ID)
        train_transformed, test_transformed, transformation = apply_log_if_positive(train_input, test_input)

        train_index = pd.date_range("2000-01-31", periods=len(train_transformed), freq="ME")
        test_index = pd.date_range(train_index[-1] + pd.offsets.MonthEnd(1), periods=len(test_transformed), freq="ME")

        train_series = pd.Series(train_transformed.to_numpy(dtype=float), index=train_index, name=SERIES_ID)
        test_series = pd.Series(test_transformed.to_numpy(dtype=float), index=test_index, name=SERIES_ID)

        print("Transformation:", transformation)
        print("First transformed train values:", train_series.head().round(6).to_list())
        print("First transformed test values:", test_series.head().round(6).to_list())
        """
    ),
    code(
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(train_series.index, train_series.values, label="train")
        ax.plot(test_series.index, test_series.values, label="test")
        ax.set_title(f"M4 Monthly {SERIES_ID} after {transformation} transform")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.show()
        """
    ),
    markdown(
        """
        ## Fixed Parameters From The Source Notebook

        The values below are the fixed M4 parameters recovered from the source
        notebook outputs/cells. Set `RUN_OPTUNA=True` in the setup cell to rerun
        training-only hyperparameter searches instead.
        """
    ),
    code(
        """
        S3_POINT_PARAMS = {
            "reservoir_size": 181,
            "spectral_radius": 0.626583409908866,
            "ar_lags": 12,
            "foundation_window": 8,
            "regressor_type": "Ridge",
            "reg_alpha": 49.324403914508146,
            "aci_step_size": 0.028807805715445868,
            "oob_split_ratio": 0.671715016966498,
        }

        S3_OPTUNA_PARAMS = {
            "reservoir_size": 127,
            "spectral_radius": 1.4238478079088417,
            "ar_lags": 6,
            "foundation_window": 8,
            "regressor_type": "Ridge",
            "reg_alpha": 21.67091719066294,
            "aci_step_size": 0.17552014573858696,
            "oob_split_ratio": 0.6078272486295386,
        }

        FASTSKETCH_POINT_PARAMS = {
            "foundation_window": 4,
            "ar_lags": 4,
            "ema_spans": (2, 4, 8),
            "conv_scales": (3, 6),
            "use_calendar": False,
            "ridge_alpha": 0.0032777556842991887,
            "shrinkage_max": 1.4703693557887536,
            "oob_split_ratio": 0.6927253088904638,
        }

        FASTSKETCH_UQ_PARAMS = {
            "aci_target": 0.14113733722772895,
            "aci_step_size": 0.038742467891629724,
            "interval_scale": 1.6520958120485625,
            "interval_power": 0.22472972379718356,
            "min_width_factor": 1.3391029964418557,
        }

        BASELINE_PARAMS = {
            "Chronos": {"model_id": "amazon/chronos-bolt-small", "device_map": "cpu"},
            "ETS": {"error": "add", "trend": "add", "seasonal": "add", "seasonal_periods": 12, "damped_trend": True},
            "ARIMA": {"p": 4, "d": 2, "q": 5, "trend": "n"},
            "AR": {"lags": 34, "trend": "t", "seasonal": True},
            "KernelRidge": {"kernel": "linear", "input_window": 33, "alpha": 0.15760720653055849},
            "GaussianProcess": {
                "input_window": 16,
                "kernel_type": "matern52",
                "length_scale": 0.1483207151065772,
                "constant_value": 0.507542862566679,
                "noise_level": 0.004868162136200997,
                "alpha": 4.041864401612213e-05,
                "normalize_y": False,
            },
        }

        if not RUN_CHRONOS_BASELINE:
            BASELINE_PARAMS.pop("Chronos", None)
        """
    ),
    markdown("## Proposed Models"),
    code(
        """
        if RUN_OPTUNA:
            s3_result = run_s3_experiment(
                train_series,
                test_series,
                point_trials=100,
                uq_trials=100,
                val_size=12,
                seed=SEED,
            )
            fastsketch_result = run_fastsketch_experiment(
                train_series,
                test_series,
                point_trials=300,
                uq_trials=200,
                val_size=12,
                seed=SEED,
            )
            S3_POINT_PARAMS = s3_result["best_point_params"]
            FASTSKETCH_POINT_PARAMS = fastsketch_result["best_point_params"]
            FASTSKETCH_UQ_PARAMS = fastsketch_result["best_uq_params"]
        else:
            s3_result = evaluate_s3_forecaster(
                train_series,
                test_series,
                S3_POINT_PARAMS,
                seasonal_period=SEASONAL_PERIOD,
                alpha=ALPHA,
            )
            fastsketch_result = evaluate_fastsketch(
                train_series,
                test_series,
                FASTSKETCH_POINT_PARAMS,
                uq_params=FASTSKETCH_UQ_PARAMS,
                seasonal_period=SEASONAL_PERIOD,
                alpha=ALPHA,
            )

        proposed_table = pd.DataFrame(
            [
                paper_metric_row("S3-Forecaster", s3_result["metrics"]),
                paper_metric_row("S3-FastSketch", fastsketch_result["metrics"]),
            ]
        )
        display(proposed_table)
        proposed_table.to_csv(OUTPUT_DIR / "proposed_models_summary.csv", index=False)
        """
    ),
    code(
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(train_series.index[-36:], train_series.iloc[-36:], label="history")
        ax.plot(test_series.index, test_series, label="actual", color="black")
        ax.plot(s3_result["forecast"].index, s3_result["forecast"]["pred"], label="S3-Forecaster")
        ax.plot(fastsketch_result["forecast"].index, fastsketch_result["forecast"]["pred"], label="S3-FastSketch")
        ax.set_title("Proposed model forecasts")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.show()
        """
    ),
    markdown("## Baselines"),
    code(
        """
        baseline_rows = []
        baseline_outputs = {}
        baseline_model_names = list(BASELINE_PARAMS)

        for model_name in baseline_model_names:
            try:
                output = evaluate_baseline(
                    model_name,
                    train_series,
                    test_series,
                    BASELINE_PARAMS[model_name],
                    seasonal_period=SEASONAL_PERIOD,
                )
                baseline_outputs[model_name] = output
                baseline_rows.append({"model": model_name, "status": "ok", "error": None, **output["metrics"]})
            except Exception as exc:
                baseline_outputs[model_name] = None
                baseline_rows.append({"model": model_name, "status": "failed", "error": str(exc)})

        baseline_summary = pd.DataFrame(baseline_rows)
        display(baseline_summary)
        baseline_summary.to_csv(OUTPUT_DIR / "baseline_summary.csv", index=False)
        """
    ),
    markdown("## Ablation Study"),
    code(
        """
        ablation_result = run_s3_ablation_study(
            train_series=train_series,
            test_series=test_series,
            best_params=S3_POINT_PARAMS,
            alpha=ALPHA,
            seasonal_period=SEASONAL_PERIOD,
        )
        ablation_summary = ablation_result["results"]
        display(ablation_summary)
        ablation_summary.to_csv(OUTPUT_DIR / "ablation_summary.csv", index=False)
        """
    ),
    markdown("## Shock / Non-Shock Analysis"),
    code(
        """
        shock_result = compare_s3_models_on_shocks(
            train_series,
            test_series,
            S3_POINT_PARAMS,
            FASTSKETCH_POINT_PARAMS,
            fastsketch_uq=FASTSKETCH_UQ_PARAMS,
            threshold_method="quantile",
            quantile=0.90,
            seasonal_period=4,
            alpha=ALPHA,
        )
        shock_summary = shock_result["summary"]
        display(shock_summary)
        shock_summary.to_csv(OUTPUT_DIR / "shock_vs_non-shock_results_final.csv", index=False)
        """
    ),
    markdown("## Data Efficiency Analysis"),
    code(
        """
        data_efficiency_params = {
            "S3-Forecaster": S3_POINT_PARAMS,
            "S3-FastSketch": FASTSKETCH_POINT_PARAMS,
            **BASELINE_PARAMS,
            "base_only": S3_POINT_PARAMS,
            "base_residual": S3_POINT_PARAMS,
            "base_residual_gate": S3_POINT_PARAMS,
        }
        data_efficiency_models = [
            "S3-Forecaster",
            "S3-FastSketch",
            "base_only",
            "base_residual",
            "base_residual_gate",
            "ETS",
            "ARIMA",
            "AR",
            "KernelRidge",
            "GaussianProcess",
        ]
        if RUN_CHRONOS_BASELINE:
            data_efficiency_models.append("Chronos")

        de_result = run_data_efficiency_analysis(
            train_series=train_series,
            test_series=test_series,
            best_params_by_model=data_efficiency_params,
            uq_params_by_model={"S3-FastSketch": FASTSKETCH_UQ_PARAMS},
            history_sizes=(36, 60, 84, 120, 160),
            model_names=data_efficiency_models,
            auto_clip_windows=True,
            seasonal_period=SEASONAL_PERIOD,
        )
        de_summary = de_result["summary"]
        display(de_summary)
        display(make_data_efficiency_pivot(de_summary, metric="rmse"))
        de_summary.to_csv(OUTPUT_DIR / "data_efficiency_analysis.csv", index=False)
        """
    ),
    markdown("## Multi-Prior Robustness"),
    code(
        """
        prior_factories = default_prior_factories(
            foundation_window=S3_POINT_PARAMS["foundation_window"],
            include_chronos=RUN_CHRONOS_PRIOR,
            chronos_kwargs={"model_id": "amazon/chronos-bolt-small", "device_map": "cpu"},
        )
        if not RUN_PROPHET_PRIOR:
            prior_factories = {name: factory for name, factory in prior_factories.items() if name != "prophet"}
        if not RUN_CHRONOS_PRIOR:
            prior_factories = {name: factory for name, factory in prior_factories.items() if name != "chronos"}

        robustness_result = run_multi_prior_robustness(
            train_series,
            test_series,
            prior_factories,
            s3_params=S3_POINT_PARAMS,
            fastsketch_params=FASTSKETCH_POINT_PARAMS,
            fastsketch_uq=FASTSKETCH_UQ_PARAMS,
            include_prior_only=True,
            alpha=ALPHA,
            seasonal_period=SEASONAL_PERIOD,
        )
        multi_prior_summary = robustness_result["summary"]
        display(multi_prior_summary)

        robustness_table = multi_prior_summary.pivot_table(
            index=["prior", "model"],
            values=["mae", "rmse", "mape_percent", "r2", "ecp", "msis", "elapsed_seconds"],
            aggfunc="first",
        )
        display(robustness_table)
        robustness_table.to_csv(OUTPUT_DIR / "robustness_table.csv")
        """
    ),
    markdown("## Takeaways"),
    code(
        """
        exported_files = sorted(path.name for path in OUTPUT_DIR.glob("*.csv"))
        print("Exported CSV files:")
        for name in exported_files:
            print("-", name)

        print("\\nPrimary proposed-model metrics:")
        display(proposed_table)

        print("\\nNotebook completed on CPU:", CPU_ONLY)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
        "kaggle": {
            "accelerator": "none",
            "dataSources": [],
            "dockerImageVersionId": 31286,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": False,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "alejopatio")
SOURCE_DATASET = f"{KAGGLE_USERNAME}/s3forecaster-s3fastsketch-source"


metadata = {
    "id": f"{KAGGLE_USERNAME}/s3forecaster-s3fastsketch-m4-cpu",
    "title": "S3Forecaster S3FastSketch M4 CPU",
    "code_file": KAGGLE_NOTEBOOK_PATH.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "dataset_sources": [SOURCE_DATASET],
    "competition_sources": [],
    "kernel_sources": [],
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    for index, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{index:02d}")
    write_json(NOTEBOOK_PATH, notebook)
    write_json(KAGGLE_NOTEBOOK_PATH, notebook)
    write_json(KAGGLE_METADATA_PATH, metadata)
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Wrote {KAGGLE_NOTEBOOK_PATH}")
    print(f"Wrote {KAGGLE_METADATA_PATH}")


if __name__ == "__main__":
    main()
