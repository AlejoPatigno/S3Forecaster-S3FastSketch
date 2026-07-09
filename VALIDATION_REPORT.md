# Validation report

Validation date: 2026-07-09

## Current package validation

Commands executed in this workspace:

```bash
python -m compileall s3paper tools
python -m pytest -q
```

Result:

```text
27 passed
```

Pytest collection:

```text
tests/test_analysis_sections.py: 3
tests/test_methodology_contract.py: 14
tests/test_result_store_statistics_diagnostics.py: 7
tests/test_smoke.py: 3
```

The current tests cover causal priors, frozen scalers, reservoir determinism, ACI order/update, rolling one-step reveal order, prior-HPO parameters, full ablation equivalence, train-only scaled metrics from the result store, internal calibration block separation, diagnostics, shock APIs, and ablation flags.

## Current execution boundary

No dataset notebooks were executed in this task.
No Kaggle kernels were executed in this task.
No final paper tables or figures were regenerated in this task.

## Legacy execution

Older Kaggle/notebook runs may exist in repository history or prior artifacts, but they are not treated as current scientific evidence after the methodology repair. Corrected empirical claims must be regenerated from the package APIs and canonical result store.

## Pending experimental execution

The user will later configure datasets, run notebooks or scripts on Kaggle, download artifacts, and regenerate paper tables and figures from the canonical result store.
