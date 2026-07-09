# Migration Notes

## Renamed Concepts

- `oob_split_ratio` is deprecated in favor of `calibration_split_ratio`.
- The old "OOB gate" terminology should be read as calibration-fitted volatility gate.

Backward-compatible aliases emit `DeprecationWarning`.

## API Additions

Both proposed models now support:

```python
fit(train_series)
predict_one()
update(observation)
get_point_components()  # S3-Forecaster
trainable_parameter_count()
summary()
```

`predict()` remains for compatibility, but final evaluation should use `rolling_one_step_forecast`.

Prior HPO parameters are now namespaced. Replace `foundation_window` as a prior search parameter with `prior__window` for `causal_rolling_mean`; keep `foundation_window` only as a legacy compatibility value.

Examples:

```python
old = {"oob_split_ratio": 0.70, "foundation_window": 6}
new = {
    "calibration_split_ratio": 0.70,
    "prior_name": "causal_rolling_mean",
    "prior__window": 6,
}
```

```python
old = {"prior_name": "rolling_mean", "foundation_window": 12}
new = {"prior_name": "causal_rolling_mean", "prior__window": 12}
```

## New Canonical Modules

- `s3paper/priors.py`
- `s3paper/preprocessing.py`
- `s3paper/residual_features.py`
- `s3paper/conformal.py`
- `s3paper/rolling_protocol.py`
- `s3paper/rolling_evaluation.py`
- `s3paper/result_store.py`
- `s3paper/calibration.py`
- `s3paper/temporal_cv.py`
- `s3paper/prior_cache.py`

## Behavioral Changes

- Prior fitted values are strictly one-step causal.
- `prior_name` is optimized for both proposed models across `causal_rolling_mean`, `seasonal_naive`, `ets`, and `theta` by default.
- Residual scalers are fitted once and reused.
- S3 reservoir radius is clipped below one.
- S3-FastSketch shrinkage is nonnegative by default.
- Test evaluation appends observed targets only after forecast storage.
- Internal model fitting separates readout training, adapter calibration, and conformal calibration blocks.
- MASE, RMSSE, and MSIS derived from the result store use train-only scales saved before test evaluation.
- Primary UQ HPO keeps nominal coverage fixed unless `optimize_nominal_level=True`.
