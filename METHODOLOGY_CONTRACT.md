# Methodology Contract

This repository now treats S3-Forecaster and S3-FastSketch as causal residual adapters evaluated under one rolling one-step protocol.

## Causality

- Prior fitted values at time `t` must use only observations before `t`.
- Residual features are transformed with statistics fitted on the readout-training block only.
- Forecast rows are stored before the corresponding test target is revealed.
- `update(observation)` may be called only after `predict_one()`.

## Priors

Canonical priors live in `s3paper/priors.py` and expose `fit`, `fitted_values`, `predict_one`, `predict`, `update(timestamp, observed_value)`, `get_params`, `clone`, and `status`.

Mandatory lightweight priors are:

- `causal_rolling_mean`
- `seasonal_naive`
- `ets`
- `theta`

Optional foundation priors are:

- `chronos`
- `timesfm`

Optional foundation failures must raise explicit errors. They must not fall back silently to `causal_rolling_mean`.

Namespaced prior parameters use the `prior__*` convention, including `prior__window`, `prior__seasonal_period`, `prior__ets_trend`, `prior__ets_damped`, `prior__theta_period`, `prior__chronos_model_id`, and `prior__timesfm_model_id`.

Legacy priors with `fit_predict(series, horizon)` are wrapped by `FitPredictPriorAdapter` so failures are explicit instead of silently replaced.

## Models

`S3Forecaster` uses a deterministic ESN residual transformer with:

- robust residual scaling fitted once;
- explicit input bias;
- explicit leak rate;
- spectral radius clipped below one;
- Ridge, BayesianRidge, Lasso, or ElasticNet according to `regressor_type`.

`S3FastSketchForecaster` uses causal multiscale residual features and the canonical bounded nonnegative contraction:

```text
gamma = clip(sum(rhat * r) / (sum(rhat^2) + eps), 0, gamma_max)
```

Signed contraction is available only through `signed_shrinkage=True`.

Both proposed models split the pretest training series into three chronological internal blocks:

```text
readout_train -> adapter_calibration -> conformal_calibration
```

Readouts and feature scalers are fitted only on `readout_train`. Gate, contraction, and adapter selector are fitted only on `adapter_calibration`. `SequentialACI` conformity scores are fitted only on `conformal_calibration`.

## Rolling Protocol

The shared evaluator is `s3paper/rolling_evaluation.py`; `s3paper/rolling_protocol.py` remains a compatibility wrapper.

For each test timestamp:

1. call `predict_one()`;
2. store point forecast, interval, and metadata;
3. reveal the target;
4. call `update(target)`.

## Conformal Intervals

`SequentialACI` generates intervals before observing the new target. It uses a finite-sample conformal order statistic and updates both score history and `alpha_t` after target reveal.

The interval passed to `SequentialACI.update()` must be the same interval reported in the forecast row.

## Metrics

All metrics are computed on the supplied original-scale arrays. MASE, RMSSE, and MSIS denominators use training data only. When metrics are derived from the canonical result store, precomputed train-only scales (`mase_scale`, `rmsse_scale`, `msis_scale`) are used; test targets are never used to reconstruct denominators.

## HPO Scope

The prior is part of the joint HPO search for both proposed models. Temporal HPO must use development folds only; final test observations must not influence priors, hyperparameters, transforms, gates, contraction, selectors, or intervals.

The primary UQ objective keeps nominal coverage fixed (`target_miscoverage = alpha`). Optimizing the nominal level is allowed only when explicitly requested with `optimize_nominal_level=True` and should be reported as supplemental.

## Result Store

The canonical result store is one row per forecast origin. It records run identifiers, git commit, dataset, series, split, model, prior parameters, model/UQ parameters, train-only scales, internal block sizes, hashes, information cutoff, target timestamp, prediction, interval, adapter activation, runtime, status, and error. Paper tables and figures must be derived from this store.

## Known Limits

This local repair did not execute dataset notebooks, Kaggle benchmark runs, regenerate paper tables, or reconcile legacy empirical values. Those tasks remain blocked on dataset/runtime execution, not on package import or local unit tests.
