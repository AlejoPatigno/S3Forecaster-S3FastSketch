"""Explicit cache for costly causal prior forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .result_store import stable_hash


def _to_float_list(values: Any) -> list[float]:
    try:
        import pandas as pd
        if isinstance(values, pd.Series):
            return values.to_numpy(dtype=float).tolist()
    except ImportError:
        pass
    import numpy as np
    return np.asarray(values, dtype=float).tolist()


@dataclass
class PriorForecastCache:
    store: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, key: str) -> dict[str, Any] | None:
        value = self.store.get(key)
        return None if value is None else dict(value)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.store[key] = dict(value)

    def make_key(
        self,
        *,
        dataset_id: str,
        series_id: str,
        fold_id: str,
        forecast_origin: Any,
        prior_name: str,
        prior_params: dict[str, Any] | None,
        history_values: Any,
        seasonal_period: int,
        horizon: int = 1,
        code_version: str = "unknown",
    ) -> str:
        d = {
            "dataset_id": str(dataset_id),
            "series_id": str(series_id),
            "fold_id": str(fold_id),
            "forecast_origin": str(forecast_origin),
            "prior_name": prior_name,
            "prior_params": prior_params or {},
            "history_values": _to_float_list(history_values),
            "seasonal_period": int(seasonal_period),
            "horizon": int(horizon),
            "code_version": str(code_version),
        }
        return stable_hash(d)


def compute_causal_prior_path(
    prior_name: str,
    prior_params: dict[str, Any],
    train: Any,
    validation: Any,
    *,
    cache: PriorForecastCache | None = None,
    cache_context: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    import numpy as np
    import pandas as pd
    from .priors import build_prior

    train_series = train if isinstance(train, pd.Series) else pd.Series(train)
    val_series = validation if isinstance(validation, pd.Series) else pd.Series(validation)

    if cache is None or cache_context is None:
        prior = build_prior(prior_name, **prior_params).fit(train_series)
        base_train = prior.fitted_values().to_numpy(dtype=float)
        base_cal = []
        for obs in val_series.to_numpy(dtype=float):
            base_cal.append(float(prior.predict(1).iloc[0]))
            prior.update(float(obs))
        return base_train, np.asarray(base_cal, dtype=float)

    ctx = dict(cache_context)
    ctx["prior_name"] = prior_name
    ctx["prior_params"] = prior_params
    ctx["horizon"] = 1
    
    # Required cache context keys
    for k in ["dataset_id", "series_id", "fold_id", "seasonal_period"]:
        if k not in ctx:
            ctx[k] = "unknown"
            
    train_key = cache.make_key(
        history_values=train_series,
        forecast_origin=train_series.index[-1] if len(train_series) else "none",
        **ctx
    )
    
    train_res = cache.get(train_key)
    if train_res is not None and "fitted_values" in train_res:
        base_train = np.asarray(train_res["fitted_values"], dtype=float)
    else:
        prior = build_prior(prior_name, **prior_params).fit(train_series)
        base_train = prior.fitted_values().to_numpy(dtype=float)
        cache.set(train_key, {"fitted_values": base_train.tolist()})

    base_cal = []
    current_history = train_series.to_numpy(dtype=float).tolist()
    current_index = train_series.index.tolist()
    val_values = val_series.to_numpy(dtype=float)
    val_index = val_series.index.tolist()
    
    prior = None
    for i, obs in enumerate(val_values):
        origin = current_index[-1] if len(current_index) else "none"
        step_key = cache.make_key(
            history_values=current_history,
            forecast_origin=origin,
            **ctx
        )
        step_res = cache.get(step_key)
        if step_res is not None and "forecast_values" in step_res:
            base_cal.append(step_res["forecast_values"][0])
        else:
            if prior is None:
                prior = build_prior(prior_name, **prior_params).fit(
                    pd.Series(current_history, index=current_index)
                )
            
            pred = float(prior.predict(1).iloc[0])
            cache.set(step_key, {"forecast_values": [pred]})
            base_cal.append(pred)
            
        current_history.append(obs)
        current_index.append(val_index[i])
        if prior is not None:
            prior.update(float(obs))
            
    return base_train, np.asarray(base_cal, dtype=float)
