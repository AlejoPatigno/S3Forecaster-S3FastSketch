"""Causal prior models used by the S3 residual adapters."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .utils import ensure_series, make_future_index


class CausalPrior(Protocol):
    def fit(self, series: Any) -> "CausalPrior": ...
    def fitted_values(self) -> pd.Series: ...
    def predict_one(self) -> float: ...
    def predict(self, horizon: int) -> pd.Series: ...
    def update(self, timestamp_or_observation: Any, observed_value: float | None = None) -> "CausalPrior": ...
    def get_params(self) -> dict[str, Any]: ...
    def clone(self) -> "CausalPrior": ...
    def status(self) -> dict[str, Any]: ...


def _future_index_from_history(history: pd.Series, horizon: int) -> pd.Index:
    return make_future_index(history, horizon)


@dataclass
class RollingMeanPrior:
    """Strictly causal moving-average prior.

    The fitted value at timestamp t is computed from observations strictly before
    t. Forecasts are recursive and append previous forecasts, never test targets.
    """

    window_size: int = 6
    fallback: str = "first"
    name: str = "causal_rolling_mean"
    _history: pd.Series | None = field(default=None, init=False, repr=False)
    _fitted: pd.Series | None = field(default=None, init=False, repr=False)
    _warnings: list[str] = field(default_factory=list, init=False, repr=False)

    def fit(self, series: Any) -> "RollingMeanPrior":
        y = ensure_series(series)
        if len(y) == 0:
            raise ValueError("RollingMeanPrior requires at least one observation.")
        values = y.to_numpy(dtype=float)
        fitted = np.empty(len(values), dtype=float)
        for t in range(len(values)):
            if t == 0:
                fitted[t] = values[0]
            else:
                start = max(0, t - self.window_size)
                fitted[t] = float(np.mean(values[start:t]))
        self._history = y.copy()
        self._fitted = pd.Series(fitted, index=y.index, name=self.name)
        return self

    def fitted_values(self) -> pd.Series:
        if self._fitted is None:
            raise RuntimeError("Prior must be fitted before fitted_values().")
        return self._fitted.copy()

    def predict(self, horizon: int) -> pd.Series:
        if self._history is None:
            raise RuntimeError("Prior must be fitted before predict().")
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        history = list(self._history.to_numpy(dtype=float))
        forecasts: list[float] = []
        for _ in range(horizon):
            start = max(0, len(history) - self.window_size)
            value = float(np.mean(history[start:]))
            forecasts.append(value)
            history.append(value)
        return pd.Series(
            forecasts,
            index=_future_index_from_history(self._history, horizon),
            name=self.name,
        )

    def predict_one(self) -> float:
        return float(self.predict(1).iloc[0])

    def update(self, timestamp_or_observation: Any, observed_value: float | None = None) -> "RollingMeanPrior":
        if self._history is None:
            raise RuntimeError("Prior must be fitted before update().")
        if observed_value is None:
            next_index = _future_index_from_history(self._history, 1)[0]
            observation = float(timestamp_or_observation)
        else:
            next_index = timestamp_or_observation
            observation = float(observed_value)
        self._history.loc[next_index] = float(observation)
        previous = self._history.iloc[:-1].to_numpy(dtype=float)
        start = max(0, len(previous) - self.window_size)
        fitted_value = float(np.mean(previous[start:])) if len(previous) else float(observation)
        self._fitted = pd.concat(
            [
                self.fitted_values(),
                pd.Series([fitted_value], index=[next_index], name=self.name),
            ]
        )
        return self

    def get_params(self) -> dict[str, Any]:
        return {"prior_name": self.name, "prior__window": int(self.window_size)}

    def clone(self) -> "RollingMeanPrior":
        return copy.deepcopy(self)

    def status(self) -> dict[str, Any]:
        return {
            "prior_name": self.name,
            "fitted": self._history is not None,
            "n_observations": 0 if self._history is None else int(len(self._history)),
            "warnings": list(self._warnings),
        }


@dataclass
class SeasonalNaivePrior:
    seasonal_period: int = 12
    name: str = "seasonal_naive"
    _history: pd.Series | None = field(default=None, init=False, repr=False)
    _fitted: pd.Series | None = field(default=None, init=False, repr=False)
    _warnings: list[str] = field(default_factory=list, init=False, repr=False)

    def fit(self, series: Any) -> "SeasonalNaivePrior":
        y = ensure_series(series)
        if len(y) == 0:
            raise ValueError("SeasonalNaivePrior requires at least one observation.")
        values = y.to_numpy(dtype=float)
        fitted = np.empty(len(values), dtype=float)
        for t in range(len(values)):
            if t >= self.seasonal_period:
                fitted[t] = values[t - self.seasonal_period]
            elif t > 0:
                fitted[t] = values[t - 1]
            else:
                fitted[t] = values[0]
        self._history = y.copy()
        self._fitted = pd.Series(fitted, index=y.index, name=self.name)
        return self

    def fitted_values(self) -> pd.Series:
        if self._fitted is None:
            raise RuntimeError("Prior must be fitted before fitted_values().")
        return self._fitted.copy()

    def predict(self, horizon: int) -> pd.Series:
        if self._history is None:
            raise RuntimeError("Prior must be fitted before predict().")
        history = list(self._history.to_numpy(dtype=float))
        forecasts: list[float] = []
        for _ in range(int(horizon)):
            if len(history) >= self.seasonal_period:
                value = float(history[-self.seasonal_period])
            else:
                value = float(history[-1])
            forecasts.append(value)
            history.append(value)
        return pd.Series(
            forecasts,
            index=_future_index_from_history(self._history, int(horizon)),
            name=self.name,
        )

    def predict_one(self) -> float:
        return float(self.predict(1).iloc[0])

    def update(self, timestamp_or_observation: Any, observed_value: float | None = None) -> "SeasonalNaivePrior":
        if self._history is None:
            raise RuntimeError("Prior must be fitted before update().")
        if observed_value is None:
            next_index = _future_index_from_history(self._history, 1)[0]
            observation = float(timestamp_or_observation)
        else:
            next_index = timestamp_or_observation
            observation = float(observed_value)
        previous = self._history.to_numpy(dtype=float)
        fitted_value = (
            float(previous[-self.seasonal_period])
            if len(previous) >= self.seasonal_period
            else float(previous[-1])
        )
        self._history.loc[next_index] = float(observation)
        self._fitted = pd.concat(
            [
                self.fitted_values(),
                pd.Series([fitted_value], index=[next_index], name=self.name),
            ]
        )
        return self

    def get_params(self) -> dict[str, Any]:
        return {
            "prior_name": self.name,
            "prior__seasonal_period": int(self.seasonal_period),
        }

    def clone(self) -> "SeasonalNaivePrior":
        return copy.deepcopy(self)

    def status(self) -> dict[str, Any]:
        return {
            "prior_name": self.name,
            "fitted": self._history is not None,
            "n_observations": 0 if self._history is None else int(len(self._history)),
            "warnings": list(self._warnings),
        }


class WalkForwardStatsPrior:
    """Base class for statsmodels priors with causal fitted values."""

    name = "stats_prior"

    def __init__(self, min_history: int = 8):
        self.min_history = int(min_history)
        self._history: pd.Series | None = None
        self._fitted: pd.Series | None = None
        self._warnings: list[str] = []

    def _warmup_forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        if len(history) == 0:
            return np.zeros(int(horizon), dtype=float)
        return np.repeat(float(history.iloc[-1]), int(horizon))

    def _forecast_from_history(self, history: pd.Series, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def fit(self, series: Any) -> "WalkForwardStatsPrior":
        y = ensure_series(series)
        fitted = np.empty(len(y), dtype=float)
        for t in range(len(y)):
            history = y.iloc[:t]
            if len(history) < self.min_history:
                fitted[t] = self._warmup_forecast(history, 1)[0] if len(history) else float(y.iloc[0])
                if t == self.min_history - 1:
                    self._warnings.append(
                        f"{self.name} used deterministic warmup forecasts before min_history={self.min_history}."
                    )
            else:
                fitted[t] = float(self._forecast_from_history(history, 1)[0])
        self._history = y.copy()
        self._fitted = pd.Series(fitted, index=y.index, name=self.name)
        return self

    def fitted_values(self) -> pd.Series:
        if self._fitted is None:
            raise RuntimeError("Prior must be fitted before fitted_values().")
        return self._fitted.copy()

    def predict(self, horizon: int) -> pd.Series:
        if self._history is None:
            raise RuntimeError("Prior must be fitted before predict().")
        values = self._forecast_from_history(self._history, int(horizon))
        return pd.Series(
            values,
            index=make_future_index(self._history, int(horizon)),
            name=self.name,
        )

    def predict_one(self) -> float:
        return float(self.predict(1).iloc[0])

    def update(self, timestamp_or_observation: Any, observed_value: float | None = None) -> "WalkForwardStatsPrior":
        if self._history is None:
            raise RuntimeError("Prior must be fitted before update().")
        previous = self._history.copy()
        if observed_value is None:
            next_index = make_future_index(previous, 1)[0]
            observation = float(timestamp_or_observation)
        else:
            next_index = timestamp_or_observation
            observation = float(observed_value)
        fitted_value = (
            self._forecast_from_history(previous, 1)[0]
            if len(previous) >= self.min_history
            else self._warmup_forecast(previous, 1)[0]
        )
        self._history.loc[next_index] = float(observation)
        self._fitted = pd.concat(
            [
                self.fitted_values(),
                pd.Series([float(fitted_value)], index=[next_index], name=self.name),
            ]
        )
        return self

    def clone(self) -> "WalkForwardStatsPrior":
        return copy.deepcopy(self)

    def status(self) -> dict[str, Any]:
        return {
            "prior_name": self.name,
            "fitted": self._history is not None,
            "n_observations": 0 if self._history is None else int(len(self._history)),
            "warnings": list(self._warnings),
        }


class ETSPrior(WalkForwardStatsPrior):
    name = "ets"

    def __init__(
        self,
        trend: str | None = "add",
        damped: bool = True,
        seasonal_period: int = 12,
        seasonal: str | None = "add",
        error: str = "add",
        min_history: int | None = None,
    ):
        super().__init__(min_history=min_history or max(8, int(seasonal_period) + 2))
        self.trend = trend
        self.damped = bool(damped)
        self.seasonal_period = int(seasonal_period)
        self.seasonal = seasonal
        self.error = error

    def _forecast_from_history(self, history: pd.Series, horizon: int) -> np.ndarray:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        seasonal = self.seasonal
        seasonal_periods = self.seasonal_period
        if seasonal_periods and len(history) < 2 * seasonal_periods + 2:
            seasonal = None
            seasonal_periods = None
        model = ETSModel(
            history,
            error=self.error,
            trend=self.trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=self.damped,
        ).fit(disp=False)
        return np.asarray(model.forecast(int(horizon)), dtype=float).reshape(-1)

    def get_params(self) -> dict[str, Any]:
        return {
            "prior_name": self.name,
            "prior__ets_trend": self.trend,
            "prior__ets_damped": self.damped,
            "prior__seasonal_period": self.seasonal_period,
        }


class ThetaPrior(WalkForwardStatsPrior):
    name = "theta"

    def __init__(
        self,
        period: int = 12,
        deseasonalize: bool = True,
        min_history: int | None = None,
    ):
        super().__init__(min_history=min_history or max(8, int(period) + 2))
        self.period = int(period)
        self.deseasonalize = bool(deseasonalize)

    def _forecast_from_history(self, history: pd.Series, horizon: int) -> np.ndarray:
        from statsmodels.tsa.forecasting.theta import ThetaModel

        period = self.period if len(history) >= 2 * self.period else None
        model = ThetaModel(
            history,
            period=period,
            deseasonalize=self.deseasonalize and period is not None,
        ).fit()
        return np.asarray(model.forecast(int(horizon)), dtype=float).reshape(-1)

    def get_params(self) -> dict[str, Any]:
        return {
            "prior_name": self.name,
            "prior__theta_period": self.period,
        }


class ChronosPrior:
    name = "chronos"

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-small",
        model_or_factory: Any = None,
        predict_fn: Any = None,
        min_history: int = 24,
    ):
        self.model_id = model_id
        self.model_or_factory = model_or_factory
        self.predict_fn = predict_fn
        self.min_history = int(min_history)

    def _make_legacy(self):
        from .multi_prior_robustness import ChronosPrior as LegacyChronosPrior

        return LegacyChronosPrior(
            predict_fn=self.predict_fn,
            model_or_factory=self.model_or_factory,
            min_history=self.min_history,
            model_id=self.model_id,
            fallback="last",
        )

    def _adapter(self) -> "FitPredictPriorAdapter":
        if self.model_or_factory is None and self.predict_fn is None:
            try:
                import chronos  # noqa: F401
            except Exception as exc:
                raise ImportError(
                    "Chronos prior requested but chronos-forecasting is unavailable; "
                    "pass prior__chronos_model_id with installed dependencies or a custom predictor."
                ) from exc
        return FitPredictPriorAdapter(self._make_legacy())

    def fit(self, series: Any) -> CausalPrior:
        self._inner = self._adapter().fit(series)
        return self

    def fitted_values(self) -> pd.Series:
        return self._inner.fitted_values()

    def predict(self, horizon: int) -> pd.Series:
        return self._inner.predict(horizon)

    def predict_one(self) -> float:
        return float(self.predict(1).iloc[0])

    def update(self, timestamp_or_observation: Any, observed_value: float | None = None) -> "ChronosPrior":
        self._inner.update(timestamp_or_observation, observed_value)
        return self

    def get_params(self) -> dict[str, Any]:
        return {"prior_name": self.name, "prior__chronos_model_id": self.model_id}

    def clone(self) -> "ChronosPrior":
        return copy.deepcopy(self)

    def status(self) -> dict[str, Any]:
        return {
            "prior_name": self.name,
            "fitted": hasattr(self, "_inner"),
            "warnings": [],
        }


class TimesFMPrior:
    """
    TimesFM foundation prior compatible with the S3 prior interface.

    This mirrors ChronosPrior: pass ``predict_fn`` or ``model_or_factory`` for
    notebook/mock runs. If both are omitted, the prior lazily loads TimesFM and
    wraps it through the common fit_predict adapter.
    """

    name = "timesfm"

    _MODEL_CACHE: dict[tuple, Any] = {}

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        *,
        model_or_factory: Any = None,
        predict_fn: Any = None,
        min_history: int = 24,
        max_context: int = 1024,
        max_horizon: int = 256,
        normalize_inputs: bool = True,
        model: Any = None,
    ):
        self.model_id = str(model_id)
        if model_or_factory is None and model is not None:
            model_or_factory = model
        self.model_or_factory = model_or_factory
        self.predict_fn = predict_fn
        self.min_history = int(min_history)
        self.max_context = int(max_context)
        self.max_horizon = int(max_horizon)
        self.normalize_inputs = bool(normalize_inputs)

        if self.min_history < 2:
            raise ValueError("min_history must be at least 2.")

        if self.max_context < self.min_history:
            raise ValueError(
                "max_context must be greater than or equal "
                "to min_history."
            )

        if self.max_horizon < 1:
            raise ValueError(
                "max_horizon must be positive."
            )

    def _load_model(self):
        if self.model_or_factory is not None:
            if callable(self.model_or_factory):
                return self.model_or_factory()
            return self.model_or_factory

        cache_key = (
            self.model_id,
            self.max_context,
            self.max_horizon,
            self.normalize_inputs,
        )

        if cache_key in self._MODEL_CACHE:
            return self._MODEL_CACHE[cache_key]

        try:
            import torch
            import timesfm
        except Exception as exc:
            raise ImportError(
                "TimesFM could not be imported. Install it with "
                "`pip install 'timesfm[torch]==2.0.2'`. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        if not hasattr(
            timesfm,
            "TimesFM_2p5_200M_torch",
        ):
            raise ImportError(
                "The installed TimesFM package does not expose "
                "`TimesFM_2p5_200M_torch`. Install the current "
                "`timesfm[torch]` package or use the adapter "
                "corresponding to the installed version."
            )

        torch.set_float32_matmul_precision("high")

        model = (
            timesfm
            .TimesFM_2p5_200M_torch
            .from_pretrained(self.model_id)
        )

        model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=self.normalize_inputs,
                use_continuous_quantile_head=False,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )

        self._MODEL_CACHE[cache_key] = model
        return model

    @staticmethod
    def _clean_history(
        history: pd.Series,
    ) -> np.ndarray:
        values = pd.Series(
            history.to_numpy(dtype=float),
            dtype=float,
        )

        if values.isna().any():
            values = (
                values
                .interpolate(limit_direction="both")
                .ffill()
                .bfill()
            )

        array = values.to_numpy(dtype=float)

        if len(array) == 0:
            raise ValueError(
                "TimesFM requires a non-empty history."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "TimesFM history contains non-finite values."
            )

        return array

    def _forecast_values(
        self,
        history: pd.Series,
        horizon: int,
    ) -> np.ndarray:
        horizon = int(horizon)

        if horizon <= 0:
            raise ValueError(
                "horizon must be positive."
            )

        if horizon > self.max_horizon:
            raise ValueError(
                f"horizon={horizon} exceeds the compiled "
                f"max_horizon={self.max_horizon}."
            )

        values = self._clean_history(history)

        # TimesFM will also manage padding, but explicit truncation
        # prevents unnecessary memory use.
        values = values[-self.max_context:]

        model = self._load_model()

        if callable(model) and not hasattr(model, "forecast"):
            point_forecast = model(history, horizon)
        else:
            output = model.forecast(
                horizon=horizon,
                inputs=[values],
            )
            point_forecast = output[0] if isinstance(output, tuple) else output

        forecast = np.asarray(
            point_forecast,
            dtype=float,
        )

        if forecast.ndim == 2:
            forecast = forecast[0]

        forecast = forecast.reshape(-1)

        if len(forecast) < horizon:
            raise ValueError(
                "TimesFM returned fewer predictions than "
                f"requested: {len(forecast)} < {horizon}."
            )

        forecast = forecast[:horizon]

        if not np.isfinite(forecast).all():
            raise ValueError(
                "TimesFM returned non-finite forecasts."
            )

        return forecast

    def _make_legacy(self):
        from .multi_prior_robustness import CallableAutoregressivePrior

        predict_fn = self.predict_fn
        if predict_fn is None:
            predict_fn = self._forecast_values

        return CallableAutoregressivePrior(
            predict_fn,
            min_history=self.min_history,
            fallback="last",
            name=self.name,
        )

    def _adapter(self) -> "FitPredictPriorAdapter":
        if self.model_or_factory is None and self.predict_fn is None:
            try:
                import timesfm  # noqa: F401
            except Exception as exc:
                raise ImportError(
                    "TimesFM prior requested but TimesFM is unavailable; "
                    "install 'timesfm[torch]' or provide a custom predictor."
                ) from exc
        return FitPredictPriorAdapter(self._make_legacy())

    def fit(self, series: Any) -> CausalPrior:
        self._inner = self._adapter().fit(series)
        return self

    def fitted_values(self) -> pd.Series:
        return self._inner.fitted_values()

    def predict(self, horizon: int) -> pd.Series:
        return self._inner.predict(horizon)

    def predict_one(self) -> float:
        return float(self.predict(1).iloc[0])

    def update(
        self,
        timestamp_or_observation: Any,
        observed_value: float | None = None,
    ) -> "TimesFMPrior":
        self._inner.update(timestamp_or_observation, observed_value)
        return self

    def get_params(self) -> dict[str, Any]:
        return {
            "prior_name": self.name,
            "prior__timesfm_model_id": self.model_id,
            "prior__timesfm_min_history": self.min_history,
            "prior__timesfm_max_context": self.max_context,
            "prior__timesfm_max_horizon": self.max_horizon,
        }

    def clone(self) -> "TimesFMPrior":
        return copy.deepcopy(self)

    def status(self) -> dict[str, Any]:
        return {
            "prior_name": self.name,
            "model_id": self.model_id,
            "fitted": hasattr(self, "_inner"),
            "warnings": [],
        }

PRIOR_REGISTRY = {
    "causal_rolling_mean": RollingMeanPrior,
    "rolling_mean": RollingMeanPrior,
    "seasonal_naive": SeasonalNaivePrior,
    "ets": ETSPrior,
    "theta": ThetaPrior,
    "chronos": ChronosPrior,
    "timesfm": TimesFMPrior,
}

MANDATORY_PRIOR_NAMES = ("causal_rolling_mean", "seasonal_naive", "ets", "theta")
OPTIONAL_PRIOR_NAMES = ("chronos", "timesfm")


def normalize_prior_name(prior_name: str) -> str:
    return "causal_rolling_mean" if prior_name == "rolling_mean" else str(prior_name)


def available_prior_names(include_optional: bool = False) -> tuple[str, ...]:
    names = list(MANDATORY_PRIOR_NAMES)
    if include_optional:
        for name in OPTIONAL_PRIOR_NAMES:
            try:
                if name == "chronos":
                    import chronos  # noqa: F401
                elif name == "timesfm":
                    import timesfm  # noqa: F401
                names.append(name)
            except Exception:
                continue
    return tuple(names)


def prior_params_from_namespace(params: dict[str, Any]) -> dict[str, Any]:
    prior_name = normalize_prior_name(params.get("prior_name", "causal_rolling_mean"))
    out: dict[str, Any] = {}
    if prior_name == "causal_rolling_mean":
        if "prior__window" in params:
            out["window_size"] = params["prior__window"]
        elif "foundation_window" in params:
            out["window_size"] = params["foundation_window"]
    elif prior_name == "seasonal_naive":
        if "prior__seasonal_period" in params:
            out["seasonal_period"] = params["prior__seasonal_period"]
    elif prior_name == "ets":
        if "prior__ets_trend" in params:
            out["trend"] = params["prior__ets_trend"]
        if "prior__ets_damped" in params:
            out["damped"] = params["prior__ets_damped"]
        if "prior__seasonal_period" in params:
            out["seasonal_period"] = params["prior__seasonal_period"]
    elif prior_name == "theta":
        if "prior__theta_period" in params:
            out["period"] = params["prior__theta_period"]
    elif prior_name == "chronos":
        if "prior__chronos_model_id" in params:
            out["model_id"] = params["prior__chronos_model_id"]
    elif prior_name == "timesfm":
        if "prior__timesfm_model_id" in params:
            out["model_id"] = params["prior__timesfm_model_id"]
    return out


def split_model_prior_params(params: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    params = dict(params)
    prior_name = normalize_prior_name(params.get("prior_name", "causal_rolling_mean"))
    prior_params = prior_params_from_namespace(params)
    model_params = {k: v for k, v in params.items() if not k.startswith("prior__")}
    model_params["prior_name"] = prior_name
    model_params["prior_params"] = prior_params
    return model_params, prior_name, prior_params


def suggest_prior_params(
    trial: Any,
    *,
    train_length: int,
    seasonal_period: int = 12,
    prior_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    names = tuple(prior_names or MANDATORY_PRIOR_NAMES)
    names = tuple(normalize_prior_name(name) for name in names)
    prior_name = trial.suggest_categorical("prior_name", list(names))
    params: dict[str, Any] = {"prior_name": prior_name}
    if prior_name == "causal_rolling_mean":
        params["prior__window"] = trial.suggest_int(
            "prior__window",
            2,
            max(2, min(24, int(train_length) // 3)),
        )
    elif prior_name == "seasonal_naive":
        params["prior__seasonal_period"] = trial.suggest_categorical(
            "prior__seasonal_period",
            [1, max(1, int(seasonal_period))],
        )
    elif prior_name == "ets":
        params["prior__seasonal_period"] = max(1, int(seasonal_period))
        params["prior__ets_trend"] = trial.suggest_categorical(
            "prior__ets_trend",
            [None, "add"],
        )
        params["prior__ets_damped"] = trial.suggest_categorical(
            "prior__ets_damped",
            [False, True],
        )
    elif prior_name == "theta":
        params["prior__theta_period"] = trial.suggest_categorical(
            "prior__theta_period",
            [1, max(1, int(seasonal_period))],
        )
    elif prior_name == "chronos":
        params["prior__chronos_model_id"] = trial.suggest_categorical(
            "prior__chronos_model_id",
            ["amazon/chronos-bolt-small"],
        )
    elif prior_name == "timesfm":
        params["prior__timesfm_model_id"] = trial.suggest_categorical(
            "prior__timesfm_model_id",
            ["google/timesfm-2.5-200m-pytorch"],
        )
    return params


def build_prior(prior_name: str = "causal_rolling_mean", **params: Any) -> CausalPrior:
    prior_name = normalize_prior_name(prior_name)
    if prior_name not in PRIOR_REGISTRY:
        raise ValueError(f"Unknown prior {prior_name!r}. Available: {sorted(PRIOR_REGISTRY)}")
    return PRIOR_REGISTRY[prior_name](**params)


class SimpleFoundationProxy(RollingMeanPrior):
    """Backward-compatible alias for the causal rolling-mean prior."""

    def __init__(self, window_size: int = 6, seasonal_period: int = 12):
        super().__init__(window_size=window_size, name="causal_rolling_mean")
        self.seasonal_period = int(seasonal_period)

    def fit_predict(self, series: Any, horizon: int):
        self.fit(series)
        return self.fitted_values(), self.predict(horizon)


class FitPredictPriorAdapter:
    """Adapter for legacy priors exposing fit_predict(series, horizon)."""

    def __init__(self, legacy_prior: Any):
        self.legacy_prior = legacy_prior
        self._history: pd.Series | None = None
        self._fitted: pd.Series | None = None
        self._warnings: list[str] = []

    def fit(self, series: Any) -> "FitPredictPriorAdapter":
        y = ensure_series(series)
        fitted, _ = self.legacy_prior.fit_predict(y, 1)
        self._history = y.copy()
        self._fitted = ensure_series(fitted).reindex(y.index).ffill().bfill()
        if len(self._fitted) != len(y):
            raise ValueError("Legacy prior returned fitted values with invalid length.")
        return self

    def fitted_values(self) -> pd.Series:
        if self._fitted is None:
            raise RuntimeError("Prior must be fitted before fitted_values().")
        return self._fitted.copy()

    def predict(self, horizon: int) -> pd.Series:
        if self._history is None:
            raise RuntimeError("Prior must be fitted before predict().")
        _, forecast = self.legacy_prior.fit_predict(self._history, int(horizon))
        return ensure_series(forecast, drop_nonfinite=False)

    def predict_one(self) -> float:
        return float(self.predict(1).iloc[0])

    def update(self, timestamp_or_observation: Any, observed_value: float | None = None) -> "FitPredictPriorAdapter":
        if self._history is None:
            raise RuntimeError("Prior must be fitted before update().")
        if observed_value is None:
            next_index = make_future_index(self._history, 1)[0]
            observation = float(timestamp_or_observation)
        else:
            next_index = timestamp_or_observation
            observation = float(observed_value)
        self._history.loc[next_index] = float(observation)
        fitted, _ = self.legacy_prior.fit_predict(self._history, 1)
        self._fitted = ensure_series(fitted).reindex(self._history.index).ffill().bfill()
        return self

    def get_params(self) -> dict[str, Any]:
        return {"prior_name": type(self.legacy_prior).__name__}

    def clone(self) -> "FitPredictPriorAdapter":
        return FitPredictPriorAdapter(copy.deepcopy(self.legacy_prior))

    def status(self) -> dict[str, Any]:
        return {
            "prior_name": type(self.legacy_prior).__name__,
            "fitted": self._history is not None,
            "n_observations": 0 if self._history is None else int(len(self._history)),
            "warnings": list(self._warnings),
        }


def ensure_causal_prior(prior: Any) -> CausalPrior:
    if all(hasattr(prior, name) for name in ("fit", "fitted_values", "predict", "update")):
        if not hasattr(prior, "predict_one"):
            prior.predict_one = lambda: float(prior.predict(1).iloc[0])  # type: ignore[attr-defined]
        return prior
    if hasattr(prior, "fit_predict"):
        return FitPredictPriorAdapter(prior)
    raise TypeError("Prior must expose either the causal API or fit_predict(series, horizon).")
