# Reproducibility

## Local Verification

Use the Python runtime available in your environment:

```bash
python -m compileall s3paper
python -m pytest -q
```

In this Codex workspace the bundled runtime was:

```text
C:\Users\LOQ\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

The local result after the repair was:

```text
19 passed
```

## Required Artifact Discipline

Full experimental runs should save:

- configuration JSON;
- environment JSON;
- git commit;
- per-origin forecasts;
- per-series metrics;
- aggregate metrics;
- diagnostics;
- shock results;
- runtime summary;
- generated paper tables.

The repository currently contains the causal code path and tests, but not a newly generated full result store from all datasets.

## Execution Boundary

Do not run final paper experiments until causality, scaler, reservoir, conformal, and rolling-protocol tests pass.
