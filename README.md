# S3Forecaster-S3FastSketch

This package consolidates the repeated experimental code from the CIF, M3, M4, Tourism, and CASAGRES notebooks into one reusable project. Dataset-loading cells remain dataset-specific; all model, optimization, metric, and analysis code is centralized here.

## Project structure

| Paper component | File |
|---|---|
| Shared array/index utilities | `s3paper/utils.py` |
| Dataset transformation helpers | `s3paper/data.py` |
| Causal prior registry | `s3paper/priors.py` |
| Frozen preprocessing transforms | `s3paper/preprocessing.py` |
| Residual feature transformers | `s3paper/residual_features.py` |
| Sequential ACI intervals | `s3paper/conformal.py` |
| Rolling one-step protocol | `s3paper/rolling_evaluation.py` |
| Internal calibration split | `s3paper/calibration.py` |
| Point and interval metrics | `s3paper/metrics.py` |
| Canonical result store | `s3paper/result_store.py` |
| Temporal CV folds | `s3paper/temporal_cv.py` |
| Prior forecast cache | `s3paper/prior_cache.py` |
| Multi-series HPO helpers | `s3paper/multiseries_hpo.py` |
| Residual diagnostics | `s3paper/residual_analysis.py` |
| S3-Forecaster class | `s3paper/s3_forecaster.py` |
| S3-Forecaster HPO and final test | `s3paper/s3_forecaster_experiment.py` |
| S3-FastSketch class | `s3paper/s3_fastsketch.py` |
| S3-FastSketch HPO and final test | `s3paper/s3_fastsketch_experiment.py` |
| Statistical, kernel, neural, linear, and KAN baselines | `s3paper/baselines.py` |
| Ablation studies for both approaches | `s3paper/ablation_study.py` |
| Shock versus non-shock analysis | `s3paper/shock_analysis.py` |
| Data-efficiency analysis | `s3paper/data_efficiency.py` |
| Multi-prior robustness | `s3paper/multi_prior_robustness.py` |
| Common paper-level orchestration | `s3paper/paper_pipeline.py` |
| Minimal executable example | `examples/minimal_pipeline.py` |
| Core smoke tests | `tests/test_smoke.py` |

## Repository setup

This folder is initialized as a Git repository named after the project directory: `S3Forecaster-S3FastSketch`.

The installable distribution name is `s3forecaster-s3fastsketch`; the import package remains `s3paper`:

```python
from s3paper.s3_forecaster_experiment import evaluate_s3_forecaster
from s3paper.s3_fastsketch_experiment import evaluate_fastsketch
```

## Installation

From the project root:

```bash
python -m pip install -e .
```

Optional neural, Prophet, Chronos, and KAN dependencies are imported lazily. They are not required for S3-Forecaster, S3-FastSketch, the statistical baselines, or the main analyses.

## M4 reproduction notebook

The original `3sforecaster-m4 (3).ipynb` notebook has been reduced to a repository-backed notebook:

```text
notebooks/3sforecaster_m4_repository_reproduction.ipynb
```

It reproduces the same M4 Monthly workflow at a higher level:

1. downloads the official M4 Monthly train/test files;
2. selects the same first monthly series, `M1`;
3. applies the same log transform when all values are strictly positive;
4. evaluates S3-Forecaster and S3-FastSketch with fixed parameters recovered from the source notebook;
5. exports the same result families: proposed model summary, baseline summary, ablation summary, shock/non-shock table, data-efficiency table, and multi-prior robustness table.

The notebook intentionally imports the package modules instead of redeclaring model classes and helper functions inline.

Regenerate the notebook after changing the project docs or workflow with:

```bash
python tools/make_m4_kaggle_notebook.py
```

## Kaggle CPU run

The Kaggle-ready copy is:

```text
kaggle/s3forecaster_s3fastsketch_m4_cpu.ipynb
```

The notebook metadata sets the accelerator to CPU/no GPU and enables internet so the M4 CSVs can be read from the official M4 GitHub repository. `kaggle/kernel-metadata.json` is generated with the configured Kaggle username, defaulting to `alejopatio` for this workspace.

The Kaggle notebook must be able to import `s3paper`. The generated notebook includes an embedded zipped fallback copy of `s3paper/`, and the metadata also references the private source dataset `alejopatio/s3forecaster-s3fastsketch-source`. For future runs, the cleanest options are:

1. attach this repository as a Kaggle input dataset containing the `s3paper/` directory;
2. run from a Kaggle environment where this repository has been checked out;
3. after publishing the Git repository remotely, add a notebook setup cell that installs from that remote URL.

Notebook and Kaggle execution are outside the current package-only validation boundary. Corrected paper results should be regenerated from the package APIs and canonical result store.

## Canonical metric convention

`evaluate_forecast` always returns both representations of relative error:

- `mape`: fractional MAPE, e.g. `0.052`;
- `mape_percent`: percentage MAPE, e.g. `5.2`;
- `smape` and `smape_percent` follow the same convention.

The publication-oriented table uses percent columns for relative metrics. Interval metrics are computed only when lower and upper bounds are provided. MASE, RMSSE, and MSIS use train-only seasonal-naive scales; result-store-derived metrics use the precomputed scales saved before test evaluation.

## Basic evaluation

```python
from s3paper.paper_pipeline import evaluate_proposed_models

results = evaluate_proposed_models(
    train_series,
    test_series,
    s3_params=best_s3_params,
    fastsketch_params=best_fastsketch_params,
    s3_uq_params=best_s3_uq,
    fastsketch_uq_params=best_fastsketch_uq,
    seasonal_period=12,
)

print(results["table"])
```

## Hyperparameter optimization

```python
from s3paper.s3_forecaster_experiment import run_s3_experiment
from s3paper.s3_fastsketch_experiment import run_fastsketch_experiment

s3_result = run_s3_experiment(
    train_series,
    test_series,
    point_trials=100,
    uq_trials=100,
    val_size=12,
    seed=42,
)

fast_result = run_fastsketch_experiment(
    train_series,
    test_series,
    point_trials=100,
    uq_trials=100,
    val_size=12,
    seed=42,
)
```

Both functions optimize on a chronological holdout extracted from the training set. The test set is used once for final evaluation.

The proposed-model HPO objective includes the prior as a first-class hyperparameter. Mandatory lightweight choices are `causal_rolling_mean`, `seasonal_naive`, `ets`, and `theta`; optional foundation choices `chronos` and `timesfm` fail explicitly if their adapters or dependencies are unavailable. Prior-specific parameters are stored with `prior__*` names such as `prior__window`, `prior__seasonal_period`, `prior__ets_trend`, `prior__ets_damped`, and `prior__theta_period`.

## Baselines

`baselines.py` provides a common interface for:

- Seasonal Naive;
- ETS;
- ARIMA;
- AutoReg;
- Kernel Ridge Regression;
- Gaussian Process Regression;
- LSTM and CNN direct forecasters;
- NLinear and DLinear;
- ARKAN;
- Chronos zero-shot forecasts through a lazy optional adapter;
- externally supplied models through `CallableBaseline`.

```python
from s3paper.baselines import run_baseline_suite

baseline_results, baseline_table = run_baseline_suite(
    train_series,
    test_series,
    trials_by_model={
        "SeasonalNaive": 3,
        "ETS": 30,
        "ARIMA": 50,
        "AR": 40,
        "KRR": 40,
        "GPR": 30,
    },
    val_size=12,
    seed=42,
)
```

Chronos can be evaluated with a notebook mock, a custom callable, or the
optional `chronos-forecasting` package:

```python
from s3paper.baselines import evaluate_baseline

chronos_output = evaluate_baseline(
    "Chronos",
    train_series,
    test_series,
    {
        # For reproducible CPU notebooks, pass a mock/factory:
        # "model_or_factory": MockChronos,
        #
        # For real Chronos, install requirements-optional.txt and choose:
        "model_id": "amazon/chronos-bolt-small",
        "device_map": "cpu",
    },
)
```

## Ablation study

```python
from s3paper.ablation_study import (
    run_s3_ablation_study,
    run_fastsketch_ablation,
    paired_ablation_wilcoxon,
)

s3_ablation = run_s3_ablation_study(
    train_series,
    test_series,
    best_s3_params,
)

fast_ablation = run_fastsketch_ablation(
    train_series,
    test_series,
    best_fastsketch_params,
    uq_params=best_fastsketch_uq,
)
```

Ablations use the same fixed optimized hyperparameters. This avoids attributing compensation produced by independent reoptimization to the removed component.

## Shock analysis

```python
from s3paper.shock_analysis import compare_s3_models_on_shocks

shock_results = compare_s3_models_on_shocks(
    train_series,
    test_series,
    best_s3_params,
    best_fastsketch_params,
    s3_uq=best_s3_uq,
    fastsketch_uq=best_fastsketch_uq,
    seasonal_period=12,
    threshold_method="iqr",
)
```

The shock threshold is estimated exclusively from the training set. Test observations are classified afterward, preventing future information from affecting the detector.

## Data efficiency

```python
from s3paper.data_efficiency import run_data_efficiency_analysis

curves = run_data_efficiency_analysis(
    train_series,
    test_series,
    best_params_by_model={
        "S3-Forecaster": best_s3_params,
        "S3-FastSketch": best_fastsketch_params,
        "ETS": best_ets_params,
    },
    uq_params_by_model={
        "S3-Forecaster": best_s3_uq,
        "S3-FastSketch": best_fastsketch_uq,
    },
    history_sizes=(36, 60, 84, 120, 160),
)
```

Hyperparameters are selected once and reused. Only history-dependent windows are clipped when a reduced sample makes the original setting infeasible.

## Multi-prior robustness

```python
from s3paper.multi_prior_robustness import (
    default_prior_factories,
    run_multi_prior_robustness,
)

prior_factories = default_prior_factories(
    foundation_window=best_s3_params["foundation_window"],
    # Use the same callable pattern as the source notebook.
    chronos_predict_fn=chronos_predict_fn,
)
# Include Chronos for the source-notebook comparison; use {"rolling", "ets"}
# instead when you want a dependency-free robustness run.
prior_factories = {k: v for k, v in prior_factories.items() if k in {"rolling", "ets", "chronos"}}

prior_results = run_multi_prior_robustness(
    train_series,
    test_series,
    prior_factories,
    s3_params=best_s3_params,
    fastsketch_params=best_fastsketch_params,
    s3_uq=best_s3_uq,
    fastsketch_uq=best_fastsketch_uq,
)
```

Custom foundation models, including TimesFM, Moirai, and TimeGPT-like models, can be supplied through `CallableAutoregressivePrior` without coupling the project to a particular external API.

## Reproducibility rules adopted by the project

1. Point and UQ hyperparameters are selected from training data only.
2. UQ optimization never queries the final test set.
3. Proposed-model test evaluation uses `predict_one()` before target reveal and `update(observation)` after forecast storage.
4. Priors expose causal fitted values and sequential update methods.
5. Residual scalers are fitted once and remain frozen on calibration/test transforms.
6. Shock thresholds are estimated from training observations only.
7. Ablation, data-efficiency, and multi-prior experiments reuse fixed hyperparameters.
8. The canonical result store records train-only scales, hashes, model/prior status, internal block sizes, information cutoffs, runtimes, and errors.
9. Every final result can report wall-clock time and the number of fitted trainable parameters through the shared metric interface.
10. Randomized components receive an explicit seed.

See `NOTEBOOK_AUDIT.md` for the consolidation decisions and differences identified among the original notebooks.
See `METHODOLOGY_CONTRACT.md` for the current causal evaluation contract and `RESULTS_AUDIT.md` for the result-reconciliation status.
