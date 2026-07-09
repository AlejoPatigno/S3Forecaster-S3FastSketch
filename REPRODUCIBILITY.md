# Reproducibility

## Package Verification

Use:

```bash
python -m compileall s3paper tools
python -m pytest -q
```

Current workspace result:

```text
27 passed
```

## Artifact Discipline

All future experimental runs should save one canonical long-format result store plus derived tables:

- `results/per_origin_results.parquet`;
- per-series metrics;
- aggregate metrics;
- diagnostics;
- shock results;
- runtime summaries;
- paper table CSV and LaTeX exports.

The result store records commit, configuration hash, data hash, prior parameters, train-only metric scales, internal block sizes, information cutoffs, forecasts, intervals, status, and errors.

## Execution Boundary

Notebooks were not executed in this package repair task.
Kaggle experiments were not executed in this package repair task.
Final paper results were not regenerated.

Run full experiments only after package tests pass and dataset configuration is fixed.
