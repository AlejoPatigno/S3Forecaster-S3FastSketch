"""Canonical S3-Forecaster implementation extracted from the research notebooks."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler

from .utils import ensure_series, make_future_index


class SimpleFoundationProxy:
    """Causal rolling-trend prior with a monthly seasonal correction."""

    def __init__(self, window_size: int = 6, seasonal_period: int = 12):
        self.window_size = int(window_size)
        self.seasonal_period = int(seasonal_period)

    def fit_predict(self, series: Any, horizon: int):
        y = ensure_series(series)
        trend = y.rolling(window=self.window_size, min_periods=1).mean().bfill()
        detrended = y - trend
        future_index = make_future_index(y, horizon)

        if isinstance(y.index, pd.DatetimeIndex) and len(y) > self.seasonal_period:
            seasonal = detrended.groupby(y.index.month).mean()
            future_season = np.asarray(
                [float(seasonal.get(timestamp.month, 0.0)) for timestamp in future_index],
                dtype=float,
            )
        elif len(y) > self.seasonal_period:
            pattern = np.asarray(
                [detrended.iloc[k :: self.seasonal_period].mean() for k in range(self.seasonal_period)],
                dtype=float,
            )
            last_position = len(y) - 1
            future_season = np.asarray(
                [pattern[(last_position + h + 1) % self.seasonal_period] for h in range(horizon)],
                dtype=float,
            )
        else:
            future_season = np.full(horizon, float(detrended.mean()))

        forecast = float(trend.iloc[-1]) + future_season
        return trend, pd.Series(forecast, index=future_index, name="foundation")


class EchoStateFeatureExtractor:
    """Fixed-weight reservoir used as a nonlinear residual feature map."""

    def __init__(
        self,
        reservoir_size: int = 50,
        spectral_radius: float = 0.9,
        seed: int = 42,
    ):
        self.reservoir_size = int(reservoir_size)
        self.spectral_radius = float(spectral_radius)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.Win: Optional[np.ndarray] = None
        self.Wres: Optional[np.ndarray] = None

    def _init_weights(self) -> None:
        self.Win = self.rng.uniform(-1, 1, (self.reservoir_size, 1))
        matrix = self.rng.normal(0, 1, (self.reservoir_size, self.reservoir_size))
        eigenvalues = np.linalg.eigvals(matrix)
        max_eigenvalue = float(np.max(np.abs(eigenvalues)))
        if max_eigenvalue <= 0:
            max_eigenvalue = 1.0
        self.Wres = matrix * (self.spectral_radius / max_eigenvalue)

    def transform(self, residuals: Any) -> np.ndarray:
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        if self.Win is None or self.Wres is None:
            self._init_weights()

        scaled = self.scaler.fit_transform(residuals.reshape(-1, 1)).reshape(-1)
        states = np.zeros((len(residuals), self.reservoir_size), dtype=float)
        hidden = np.zeros(self.reservoir_size, dtype=float)
        for t, value in enumerate(scaled):
            hidden = np.tanh(self.Win[:, 0] * value + self.Wres @ hidden)
            states[t] = hidden
        return states


class S3Forecaster:
    """Selective shock-aware small-data forecaster.

    The trainable part is a regularized multi-output residual readout. The
    reservoir is fixed. A volatility gate and adaptive conformal calibration
    are estimated on a chronological out-of-bag block.
    """

    def __init__(
        self,
        reservoir_size: int = 50,
        spectral_radius: float = 0.9,
        ar_lags: int = 3,
        horizon: int = 12,
        aci_step_size: float = 0.05,
        foundation_window: int = 6,
        regressor_type: str = "Ridge",
        reg_alpha: float = 1.0,
        oob_split_ratio: float = 0.7,
        target_miscoverage: float = 0.10,
        seed: int = 42,
        foundation: Optional[Any] = None,
    ):
        self.reservoir_size = int(reservoir_size)
        self.spectral_radius = float(spectral_radius)
        self.ar_lags = int(ar_lags)
        self.horizon = int(horizon)
        self.aci_step_size = float(aci_step_size)
        self.foundation_window = int(foundation_window)
        self.regressor_type = str(regressor_type)
        self.reg_alpha = float(reg_alpha)
        self.oob_split_ratio = float(oob_split_ratio)
        self.target_miscoverage = float(target_miscoverage)
        self.seed = int(seed)

        self.foundation = foundation or SimpleFoundationProxy(window_size=self.foundation_window)
        self.esn = EchoStateFeatureExtractor(
            reservoir_size=self.reservoir_size,
            spectral_radius=self.spectral_radius,
            seed=self.seed,
        )
        self.readout = self._build_readout()

        self.gate_alpha = 1.0
        self.gate_beta = 0.0
        self.vol_stats: Optional[tuple[float, float]] = None
        self.conformity_scores = np.asarray([], dtype=float)
        self.current_alpha_t = self.target_miscoverage
        self.fitted = False
        self.fit_report_: dict[str, Any] = {}

    def _build_readout(self) -> MultiOutputRegressor:
        if self.regressor_type == "Ridge":
            base = Ridge(alpha=self.reg_alpha)
        elif self.regressor_type == "Lasso":
            base = Lasso(alpha=self.reg_alpha, max_iter=10000)
        elif self.regressor_type == "ElasticNet":
            base = ElasticNet(alpha=self.reg_alpha, l1_ratio=0.5, max_iter=10000)
        else:
            raise ValueError(f"Unknown regressor_type={self.regressor_type!r}.")
        return MultiOutputRegressor(base)

    def _compute_volatility_z(self, residuals: Any) -> np.ndarray:
        volatility = pd.Series(np.asarray(residuals, dtype=float)).ewm(span=4).std().fillna(0).to_numpy()
        if self.vol_stats is None:
            iqr = float(np.percentile(volatility, 75) - np.percentile(volatility, 25))
            if iqr <= 0:
                iqr = 1.0
            self.vol_stats = (float(np.median(volatility)), iqr + 1e-6)
        median, iqr = self.vol_stats
        return (volatility - median) / (iqr / 1.35)

    def _get_gate(self, residuals: Any, alpha: float, beta: float) -> np.ndarray:
        z = self._compute_volatility_z(residuals)
        alpha = float(np.clip(alpha, 0.01, 50.0))
        logits = np.clip(alpha * (z - beta), -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def create_features(self, residuals: Any) -> np.ndarray:
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        esn_states = self.esn.transform(residuals)
        autoregressive = np.zeros((len(residuals), self.ar_lags), dtype=float)
        for index in range(self.ar_lags, len(residuals)):
            autoregressive[index] = residuals[index - self.ar_lags : index][::-1]
        return np.hstack([esn_states, autoregressive])

    def fit(self, series: Any):
        y = ensure_series(series)
        n = len(y)
        split_index = int(n * self.oob_split_ratio)
        split_index = max(split_index, self.ar_lags + self.horizon + 2)
        split_index = min(split_index, n - self.horizon - 1)

        self.fitted = False
        self.fit_report_ = {
            "n": n,
            "split_index": split_index,
            "horizon": self.horizon,
            "status": "initializing",
        }

        if split_index <= self.ar_lags or n < self.ar_lags + self.horizon + 5:
            self.fit_report_["status"] = "insufficient_data"
            return self

        train = y.iloc[:split_index]
        base_train, _ = self.foundation.fit_predict(train, self.horizon)
        base_train = ensure_series(base_train).reindex(train.index).ffill().bfill()
        residual_train = train.to_numpy() - base_train.to_numpy()

        x_train = self.create_features(residual_train)
        y_train = np.full((len(residual_train), self.horizon), np.nan, dtype=float)
        for i in range(len(residual_train) - self.horizon):
            y_train[i] = residual_train[i + 1 : i + 1 + self.horizon]

        valid = np.isfinite(y_train).all(axis=1)
        valid[: self.ar_lags] = False
        if int(valid.sum()) < 2:
            self.fit_report_["status"] = "insufficient_readout_samples"
            return self

        self.readout = self._build_readout()
        self.readout.fit(x_train[valid], y_train[valid])

        full_base, _ = self.foundation.fit_predict(y, self.horizon)
        full_base = ensure_series(full_base).reindex(y.index).ffill().bfill()
        full_residual = y.to_numpy() - full_base.to_numpy()
        x_full = self.create_features(full_residual)
        calibration_indices = np.arange(split_index, n - self.horizon)

        self.last_series = y
        self.last_residuals = full_residual
        self.full_base_fit_ = full_base
        self.X_full_ = x_full

        if len(calibration_indices) == 0:
            self.fit_report_["status"] = "fit_without_calibration"
            self.fitted = True
            return self

        y_calibration = np.asarray(
            [full_residual[i + 1 : i + 1 + self.horizon] for i in calibration_indices],
            dtype=float,
        )
        raw_calibration = self.readout.predict(x_full[calibration_indices])
        z_calibration = self._compute_volatility_z(full_residual)[calibration_indices]

        def gate_loss(parameters: np.ndarray) -> float:
            alpha, beta = parameters
            alpha = float(np.clip(alpha, 0.01, 20.0))
            gate = 1.0 / (1.0 + np.exp(-np.clip(alpha * (z_calibration - beta), -50, 50)))
            return float(np.mean(np.abs(y_calibration - raw_calibration * gate[:, None])))

        optimum = minimize(gate_loss, [1.0, 0.0], bounds=[(0.1, 10.0), (-3.0, 3.0)])
        self.gate_alpha, self.gate_beta = map(float, optimum.x)

        gate = 1.0 / (
            1.0
            + np.exp(
                -np.clip(self.gate_alpha * (z_calibration - self.gate_beta), -50, 50)
            )
        )
        score_matrix = np.abs(y_calibration - raw_calibration * gate[:, None])
        alpha_t = self.target_miscoverage
        flattened_buffer: list[float] = []
        for row in score_matrix:
            row_score = float(np.max(row))
            if len(flattened_buffer) >= 2:
                q = float(np.quantile(flattened_buffer, 1.0 - alpha_t))
                miss = int(row_score > q)
                alpha_t += self.aci_step_size * (self.target_miscoverage - miss)
                alpha_t = float(np.clip(alpha_t, 0.01, 0.50))
            flattened_buffer.append(row_score)

        self.current_alpha_t = alpha_t
        self.conformity_scores = np.asarray(flattened_buffer, dtype=float)
        self.fit_report_.update(
            {
                "status": "success",
                "n_readout_samples": int(valid.sum()),
                "n_calibration_samples": int(len(calibration_indices)),
                "n_features": int(x_full.shape[1]),
            }
        )
        self.fitted = True
        return self

    def predict(self) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError(f"S3Forecaster is not fitted: {self.fit_report_}.")

        _, base_forecast = self.foundation.fit_predict(self.last_series, self.horizon)
        base_forecast = ensure_series(base_forecast, name="base")
        x_last = self.create_features(self.last_residuals)[-1].reshape(1, -1)
        residual_prediction = self.readout.predict(x_last)[0]
        gate_value = float(
            self._get_gate(self.last_residuals, self.gate_alpha, self.gate_beta)[-1]
        )
        prediction = base_forecast.to_numpy() + residual_prediction * gate_value

        width = (
            float(np.quantile(self.conformity_scores, 1.0 - self.current_alpha_t))
            if len(self.conformity_scores)
            else 0.0
        )
        return pd.DataFrame(
            {
                "base": base_forecast.to_numpy(),
                "pred": prediction,
                "lower": prediction - width,
                "upper": prediction + width,
                "gate": gate_value,
                "q_width": width,
            },
            index=base_forecast.index,
        )

    def trainable_parameter_count(self) -> int:
        total = 0
        for estimator in getattr(self.readout, "estimators_", []):
            total += np.asarray(estimator.coef_).size
            total += np.asarray(estimator.intercept_).size
        if self.fitted:
            total += 2  # optimized gate alpha and beta
        return int(total)

    def summary(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "fit_report": dict(self.fit_report_),
            "gate_alpha": self.gate_alpha,
            "gate_beta": self.gate_beta,
            "current_alpha_t": self.current_alpha_t,
            "n_conformity_scores": len(self.conformity_scores),
            "trainable_params": self.trainable_parameter_count(),
        }


# Backward-compatible name retained from the notebooks.
S3ForecasterV6 = S3Forecaster
