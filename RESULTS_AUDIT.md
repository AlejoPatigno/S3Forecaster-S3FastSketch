# Results Audit

## Current Status

No legacy paper value was manually selected or preserved during this repair.

The local work repaired causal model code and added tests, but did not rerun full public-dataset experiments or Kaggle notebooks. Therefore corrected empirical tables are not claimed in this file.

## Reconciliation Policy

Every future result table must be regenerated from the canonical package path:

- causal priors from `s3paper/priors.py`;
- residual features from `s3paper/residual_features.py`;
- rolling test forecasts from `s3paper/rolling_evaluation.py`;
- metrics from `s3paper/metrics.py`.

Any old/new discrepancy must record:

- source artifact;
- dataset;
- series id or benchmark subset;
- model;
- metric;
- old value;
- corrected value;
- cause.

## Known Unresolved Legacy Targets

The brief listed ICMD, M4, and ARIMA-ICMD disagreements. They remain unresolved until the corrected multi-series pipeline is executed. In particular, ARIMA-ICMD still needs a transformation-aware regression run before any corrected claim can be made.

## Local Validation Completed

```text
python -m compileall s3paper tools
python -m pytest -q
```

Result:

```text
20 passed
```
