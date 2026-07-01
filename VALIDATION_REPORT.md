# Validation report

Validation date: 2026-07-01

## Static validation

- Every Python module was compiled with `python -m compileall`.
- The package was installed in editable mode from `pyproject.toml` without dependency resolution.
- Public imports for `S3Forecaster` and `S3FastSketchForecaster` were verified.

## Automated tests

The test suite completed successfully:

```text
4 passed
```

The tests cover:

1. consistency between fractional and percentage MAPE;
2. end-to-end S3-Forecaster fitting, forecasting, intervals, and parameter counts;
3. end-to-end S3-FastSketch fitting, recursive forecasting, intervals, and parameter counts;
4. reduced S3 and FastSketch ablation studies;
5. shock and non-shock stratification;
6. data-efficiency evaluation;
7. multi-prior robustness using the rolling prior;
8. residual diagnostics;
9. Seasonal Naive baseline evaluation.

## Additional runtime checks

Kernel Ridge (`KRR`) and Gaussian Process (`GPR`) aliases were executed on a synthetic monthly series. A minimal Optuna study for Seasonal Naive was also completed.

## Dependency boundary

The following pathways were not executed because they require optional external dependencies or APIs:

- TensorFlow LSTM and CNN;
- PyTorch NLinear and DLinear when PyTorch is absent;
- ARKAN when a compatible KAN implementation is absent;
- Prophet prior when Prophet is absent;
- Chronos and other foundation models, which require a user-supplied prediction callable.

These pathways use lazy imports, so their absence does not affect the core package.

## M4 notebook and Kaggle packaging update

Update date: 2026-07-01

Generated artifacts:

- `notebooks/3sforecaster_m4_repository_reproduction.ipynb`;
- `kaggle/s3forecaster_s3fastsketch_m4_cpu.ipynb`;
- `kaggle/kernel-metadata.json`;
- `tools/make_m4_kaggle_notebook.py`;
- `tools/build_kaggle_source_dataset.py`;
- `KAGGLE.md`.

Validation completed:

- Both generated notebooks parse as valid JSON notebook documents.
- Each generated notebook contains 26 cells.
- Kaggle notebook metadata sets CPU/no GPU execution.
- `kernel-metadata.json` points to `s3forecaster_s3fastsketch_m4_cpu.ipynb`.
- `python -m pytest` completed successfully with `4 passed`.
- Kaggle CLI `2.2.3` was installed and authenticated with the configured account.
- Private source dataset `alejopatio/s3forecaster-s3fastsketch-source` was created.
- Kaggle notebook version 4 completed successfully on CPU:
  `https://www.kaggle.com/code/alejopatio/s3forecaster-s3fastsketch-m4-cpu`.
- Completed Kaggle outputs were downloaded under `kaggle/run_output_v4/` and included:
  `proposed_models_summary.csv`, `baseline_summary.csv`, `ablation_summary.csv`,
  `shock_vs_non-shock_results_final.csv`, `data_efficiency_analysis.csv`, and
  `robustness_table.csv`.

Local notebook execution was not repeated end-to-end after the Kaggle success because the notebook downloads the full M4 Monthly CSV files and the requested target environment was Kaggle CPU. Package-level tests and Kaggle execution cover the project code path.

To rerun end-to-end on Kaggle:

```bash
kaggle kernels push -p kaggle
kaggle kernels status alejopatio/s3forecaster-s3fastsketch-m4-cpu
```
