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

## New Canonical Modules

- `s3paper/priors.py`
- `s3paper/preprocessing.py`
- `s3paper/residual_features.py`
- `s3paper/conformal.py`
- `s3paper/rolling_protocol.py`

## Behavioral Changes

- Prior fitted values are strictly one-step causal.
- `prior_name` is optimized for both proposed models across `causal_rolling_mean`, `seasonal_naive`, `ets`, and `theta` by default.
- Residual scalers are fitted once and reused.
- S3 reservoir radius is clipped below one.
- S3-FastSketch shrinkage is nonnegative by default.
- Test evaluation appends observed targets only after forecast storage.
