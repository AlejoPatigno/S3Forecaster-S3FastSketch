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

## Rolling Protocol

The shared evaluator is `s3paper/rolling_evaluation.py`; `s3paper/rolling_protocol.py` remains a compatibility wrapper.

For each test timestamp:

1. call `predict_one()`;
2. store point forecast, interval, and metadata;
3. reveal the target;
4. call `update(target)`.

## Conformal Intervals

`SequentialACI` generates intervals before observing the new target. It uses a finite-sample conformal order statistic and updates both score history and `alpha_t` after target reveal.

## Metrics

All metrics are computed on the supplied original-scale arrays. MSIS denominators use training data only through `seasonal_naive_scale`.

## Known Limits

This local repair did not execute full Kaggle benchmark runs, regenerate paper tables, or reconcile legacy empirical values. Those tasks remain blocked on dataset/runtime execution, not on package import or local unit tests.
