"""Baseline models, model registry, Optuna optimization, and final evaluation.

Optional deep-learning dependencies are imported lazily. Statistical and
scikit-learn baselines remain usable without TensorFlow, PyTorch, or pykan.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .chronos import chronos_predict_fixed_horizon, make_chronos_predictor
from .metrics import evaluate_forecast
from .utils import (
    count_trainable_parameters,
    ensure_series,
    make_future_index,
    parse_forecast_output,
    recursive_point_forecast,
    set_global_seed,
    to_1d_array,
)

SEED = 42


def make_direct_windows(values: Any, input_window: int, horizon: int):
    y = to_1d_array(values)
    X, Y = [], []
    for i in range(max(0, len(y) - input_window - horizon + 1)):
        X.append(y[i : i + input_window])
        Y.append(y[i + input_window : i + input_window + horizon])
    if not X:
        return np.empty((0, input_window)), np.empty((0, horizon))
    return np.asarray(X), np.asarray(Y)


def make_one_step_windows(values: Any, window_size: int):
    y = to_1d_array(values)
    X, target = [], []
    for i in range(max(0, len(y) - window_size)):
        X.append(y[i : i + window_size])
        target.append(y[i + window_size])
    if not X:
        return np.empty((0, window_size)), np.empty((0,))
    return np.asarray(X), np.asarray(target)


def split_train_val(train_series: Any, val_horizon: int):
    y = ensure_series(train_series)
    if len(y) <= val_horizon + 15:
        raise ValueError("Insufficient observations for temporal validation.")
    return y.iloc[:-val_horizon], y.iloc[-val_horizon:]


def _scaled_direct_data(train_series: Any, input_window: int, horizon: int):
    y = to_1d_array(train_series)
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
    X, Y = make_direct_windows(y_scaled, input_window, horizon)
    if len(X) < 8:
        raise ValueError(f"Only {len(X)} direct windows are available.")
    return scaler, y_scaled, X, Y


class ForecastBaseline:
    def fit(self, series: Any):
        raise NotImplementedError

    def predict(self, horizon: int):
        raise NotImplementedError

    def trainable_parameter_count(self) -> int:
        return count_trainable_parameters(getattr(self, "model_", None))


class SeasonalNaiveBaseline(ForecastBaseline):
    def __init__(self, seasonal_period: int = 12):
        self.seasonal_period = int(seasonal_period)

    def fit(self, series: Any):
        self.series_ = ensure_series(series)
        return self

    def predict(self, horizon: int):
        y = self.series_.to_numpy()
        m = self.seasonal_period
        values = [float(y[-m + (h % m)]) if len(y) >= m else float(y[-1]) for h in range(horizon)]
        return pd.Series(values, index=make_future_index(self.series_, horizon), name="pred")


class ETSBaseline(ForecastBaseline):
    def __init__(
        self,
        error: str = "add",
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        seasonal_periods: Optional[int] = 12,
        damped_trend: bool = True,
    ):
        self.params = locals().copy()
        self.params.pop("self")

    def fit(self, series: Any):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        self.series_ = ensure_series(series)
        params = dict(self.params)
        if params["seasonal_periods"] and len(self.series_) < 2 * params["seasonal_periods"] + 2:
            params["seasonal"] = None
            params["seasonal_periods"] = None
        self.model_ = ETSModel(self.series_, **params).fit(disp=False)
        return self

    def predict(self, horizon: int):
        values = np.asarray(self.model_.forecast(horizon), dtype=float).reshape(-1)
        return pd.Series(values, index=make_future_index(self.series_, horizon), name="pred")

    def trainable_parameter_count(self) -> int:
        return int(np.asarray(getattr(self.model_, "params", [])).size)


class ARIMABaseline(ForecastBaseline):
    def __init__(self, p: int = 1, d: int = 0, q: int = 0, trend: str = "c"):
        self.order = (int(p), int(d), int(q))
        self.trend = trend

    def fit(self, series: Any):
        from statsmodels.tsa.arima.model import ARIMA

        self.series_ = ensure_series(series)
        self.model_ = ARIMA(
            self.series_,
            order=self.order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        return self

    def predict(self, horizon: int):
        values = np.asarray(self.model_.forecast(steps=horizon), dtype=float).reshape(-1)
        return pd.Series(values, index=make_future_index(self.series_, horizon), name="pred")

    def trainable_parameter_count(self) -> int:
        return int(np.asarray(self.model_.params).size)


class AutoRegBaseline(ForecastBaseline):
    def __init__(self, lags: int = 12, trend: str = "ct", seasonal: bool = False, period: int = 12):
        self.lags = int(lags)
        self.trend = trend
        self.seasonal = bool(seasonal)
        self.period = int(period)

    def fit(self, series: Any):
        from statsmodels.tsa.ar_model import AutoReg

        self.series_ = ensure_series(series)
        self.model_ = AutoReg(
            self.series_,
            lags=min(self.lags, max(1, len(self.series_) // 3)),
            trend=self.trend,
            seasonal=self.seasonal and len(self.series_) > self.lags + self.period + 2,
            period=self.period if self.seasonal else None,
            old_names=False,
        ).fit()
        return self

    def predict(self, horizon: int):
        start = len(self.series_)
        values = np.asarray(
            self.model_.predict(start=start, end=start + horizon - 1, dynamic=False), dtype=float
        ).reshape(-1)
        return pd.Series(values, index=make_future_index(self.series_, horizon), name="pred")

    def trainable_parameter_count(self) -> int:
        return int(np.asarray(self.model_.params).size)


class KernelRidgeBaseline(ForecastBaseline):
    def __init__(
        self,
        input_window: int = 12,
        alpha: float = 1.0,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
    ):
        self.input_window = int(input_window)
        self.params = dict(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

    def fit(self, series: Any, horizon: int = 1):
        self.series_ = ensure_series(series)
        self.horizon_ = int(horizon)
        self.scaler_, self.y_scaled_, X, Y = _scaled_direct_data(
            self.series_, self.input_window, self.horizon_
        )
        self.model_ = KernelRidge(**self.params).fit(X, Y)
        return self

    def predict(self, horizon: Optional[int] = None):
        horizon = self.horizon_ if horizon is None else int(horizon)
        if horizon != self.horizon_:
            raise ValueError("KernelRidgeBaseline must be refit when the horizon changes.")
        x = self.y_scaled_[-self.input_window :].reshape(1, -1)
        scaled = np.asarray(self.model_.predict(x)).reshape(-1)
        values = self.scaler_.inverse_transform(scaled.reshape(-1, 1)).reshape(-1)
        return pd.Series(values, index=make_future_index(self.series_, horizon), name="pred")

    def trainable_parameter_count(self) -> int:
        return int(np.asarray(getattr(self.model_, "dual_coef_", [])).size)


def _gp_kernel(kernel_type: str, length_scale: float, constant_value: float, noise_level: float):
    const = ConstantKernel(constant_value, (1e-3, 1e3))
    if kernel_type == "rbf":
        base = RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e3))
    elif kernel_type == "matern32":
        base = Matern(length_scale=length_scale, nu=1.5)
    else:
        base = Matern(length_scale=length_scale, nu=2.5)
    return const * base + WhiteKernel(noise_level, (1e-8, 1e1))


class GaussianProcessBaseline(ForecastBaseline):
    def __init__(
        self,
        input_window: int = 12,
        kernel_type: str = "rbf",
        length_scale: float = 1.0,
        constant_value: float = 1.0,
        noise_level: float = 1e-3,
        alpha: float = 1e-8,
        normalize_y: bool = True,
    ):
        self.input_window = int(input_window)
        self.params = locals().copy()
        self.params.pop("self")
        self.params.pop("input_window")

    def fit(self, series: Any, horizon: int = 1):
        self.series_ = ensure_series(series)
        self.horizon_ = int(horizon)
        self.scaler_, self.y_scaled_, X, Y = _scaled_direct_data(
            self.series_, self.input_window, self.horizon_
        )
        kernel = _gp_kernel(
            self.params["kernel_type"],
            self.params["length_scale"],
            self.params["constant_value"],
            self.params["noise_level"],
        )
        base = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.params["alpha"],
            normalize_y=self.params["normalize_y"],
            n_restarts_optimizer=0,
            random_state=SEED,
        )
        self.model_ = MultiOutputRegressor(base).fit(X, Y)
        return self

    def predict(self, horizon: Optional[int] = None):
        horizon = self.horizon_ if horizon is None else int(horizon)
        if horizon != self.horizon_:
            raise ValueError("GaussianProcessBaseline must be refit when the horizon changes.")
        x = self.y_scaled_[-self.input_window :].reshape(1, -1)
        scaled = np.asarray(self.model_.predict(x)).reshape(-1)
        values = self.scaler_.inverse_transform(scaled.reshape(-1, 1)).reshape(-1)
        return pd.Series(values, index=make_future_index(self.series_, horizon), name="pred")

    def trainable_parameter_count(self) -> int:
        total = 0
        for estimator in getattr(self.model_, "estimators_", []):
            total += np.asarray(getattr(estimator, "alpha_", [])).size
        return int(total)


class KerasDirectBaseline(ForecastBaseline):
    def __init__(self, architecture: str = "LSTM", **params):
        self.architecture = architecture.upper()
        self.params = params
        self.input_window = int(params.get("input_window", 12))

    def fit(self, series: Any, horizon: int = 1):
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, LSTM
        from tensorflow.keras.optimizers import Adam

        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED)
        self.series_ = ensure_series(series)
        self.horizon_ = int(horizon)
        self.scaler_, self.y_scaled_, X, Y = _scaled_direct_data(
            self.series_, self.input_window, self.horizon_
        )
        X = X[..., None]
        split = min(max(int(0.8 * len(X)), 1), len(X) - 1)
        X_train, X_val, y_train, y_val = X[:split], X[split:], Y[:split], Y[split:]

        p = self.params
        model = Sequential([Input(shape=(self.input_window, 1))])
        if self.architecture == "LSTM":
            if int(p.get("n_layers", 1)) == 2:
                model.add(LSTM(int(p.get("units_1", 32)), return_sequences=True, dropout=float(p.get("dropout", 0.0))))
                model.add(LSTM(int(p.get("units_2", 16)), dropout=float(p.get("dropout", 0.0))))
            else:
                model.add(LSTM(int(p.get("units_1", 32)), dropout=float(p.get("dropout", 0.0))))
        elif self.architecture == "CNN":
            filters = int(p.get("filters", 32))
            model.add(Conv1D(filters, int(p.get("kernel_size", 3)), activation="relu", padding="causal"))
            model.add(Dropout(float(p.get("dropout", 0.0))))
            model.add(Conv1D(max(8, filters // 2), 2, activation="relu", padding="causal"))
            model.add(Flatten())
        else:
            raise ValueError(f"Unknown Keras architecture {self.architecture}.")

        model.add(Dense(int(p.get("dense_units", 32)), activation="relu"))
        model.add(Dense(self.horizon_))
        model.compile(optimizer=Adam(float(p.get("learning_rate", 1e-3))), loss="mae")
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=int(p.get("epochs", 100)),
            batch_size=int(p.get("batch_size", 16)),
            shuffle=False,
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)],
        )
        self.model_ = model
        return self

    def predict(self, horizon: Optional[int] = None):
        horizon = self.horizon_ if horizon is None else int(horizon)
        if horizon != self.horizon_:
            raise ValueError("KerasDirectBaseline must be refit when the horizon changes.")
        x = self.y_scaled_[-self.input_window :].reshape(1, self.input_window, 1)
        scaled = self.model_.predict(x, verbose=0).reshape(-1)
        values = self.scaler_.inverse_transform(scaled.reshape(-1, 1)).reshape(-1)
        return pd.Series(values, index=make_future_index(self.series_, horizon), name="pred")


class _NLinearNet:
    @staticmethod
    def build(window_size: int):
        import torch
        import torch.nn as nn

        class NLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(window_size, 1)

            def forward(self, x):
                last = x[:, -1:].detach()
                return self.linear(x - last) + last

        return NLinear()


class _DLinearNet:
    @staticmethod
    def build(window_size: int, kernel_size: int = 3):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class DLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.seasonal = nn.Linear(window_size, 1)
                self.trend = nn.Linear(window_size, 1)

            def forward(self, x):
                pad = kernel_size // 2
                trend = F.avg_pool1d(x[:, None, :], kernel_size, stride=1, padding=pad)[:, 0, :]
                trend = trend[:, : x.shape[1]]
                seasonal = x - trend
                return self.seasonal(seasonal) + self.trend(trend)

        return DLinear()


class TorchLinearBaseline(ForecastBaseline):
    def __init__(
        self,
        architecture: str = "NLinear",
        window_size: int = 12,
        kernel_size: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 150,
        patience: int = 15,
        batch_size: int = 16,
        clip_grad: Optional[float] = None,
    ):
        self.architecture = architecture
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = max(1, int(batch_size))
        self.clip_grad = (
            None if clip_grad is None else float(clip_grad)
        )

    def fit(self, series: Any):
        import torch
        import torch.nn as nn

        set_global_seed(SEED)
        self.series_ = ensure_series(series)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(self.series_.to_numpy().reshape(-1, 1)).reshape(-1)
        X, y = make_one_step_windows(scaled, self.window_size)
        if len(X) < 10:
            raise ValueError("Too few one-step windows.")
        split = min(max(int(0.8 * len(X)), 1), len(X) - 1)
        X_train = torch.tensor(X[:split], dtype=torch.float32)
        y_train = torch.tensor(y[:split, None], dtype=torch.float32)
        X_val = torch.tensor(X[split:], dtype=torch.float32)
        y_val = torch.tensor(y[split:, None], dtype=torch.float32)

        if self.architecture == "NLinear":
            model = _NLinearNet.build(self.window_size)
        elif self.architecture == "DLinear":
            model = _DLinearNet.build(self.window_size, self.kernel_size)
        else:
            raise ValueError(f"Unknown architecture {self.architecture}.")

        from torch.utils.data import DataLoader, TensorDataset

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=False,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        criterion = nn.MSELoss()

        best_state = None
        best_loss = np.inf
        bad = 0

        for _ in range(self.epochs):
            model.train()

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)

                prediction = model(x_batch)
                loss = criterion(prediction, y_batch)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "NLinear/DLinear produced a non-finite training loss."
                    )

                loss.backward()

                if self.clip_grad is not None and self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=self.clip_grad,
                    )

                optimizer.step()

            model.eval()

            with torch.no_grad():
                val_prediction = model(X_val)
                val_loss = float(
                    criterion(val_prediction, y_val).item()
                )

            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                bad = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1

                if bad >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model_ = model
        self.scaled_history_ = list(scaled)

        return self

    def predict(self, horizon: int):
        import torch

        history = list(self.scaled_history_)
        predictions = []
        self.model_.eval()
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(history[-self.window_size :], dtype=torch.float32).reshape(1, -1)
                value = float(self.model_(x).reshape(-1)[0].item())
                predictions.append(value)
                history.append(value)
        values = self.scaler_.inverse_transform(np.asarray(predictions).reshape(-1, 1)).reshape(-1)
        return pd.Series(values, index=make_future_index(self.series_, horizon), name="pred")


class ARKANBaseline(ForecastBaseline):
    """Autoregressive KAN baseline using the optional pykan package."""

    def __init__(
        self,
        window_size: int = 12,
        h1: int = 8,
        h2: int = 4,
        n_hidden_layers: int = 1,
        grid: int = 3,
        k: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 100,
        patience: int = 12,
        batch_size: int = 16,
        clip_grad: Optional[float] = None,
    ):
        self.params = {
            "window_size": int(window_size),
            "h1": int(h1),
            "h2": int(h2),
            "n_hidden_layers": int(n_hidden_layers),
            "grid": int(grid),
            "k": int(k),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "epochs": int(epochs),
            "patience": int(patience),
            "batch_size": max(1, int(batch_size)),
            "clip_grad": (
                None if clip_grad is None else float(clip_grad)
            ),
        }

        self.window_size = int(window_size)

    @staticmethod
    def _kan_class():
        try:
            from kan import KAN

            return KAN
        except Exception:
            try:
                from pykan import KAN

                return KAN
            except Exception as exc:
                raise ImportError("Install pykan/kan to use ARKANBaseline.") from exc

    def fit(self, series: Any):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        set_global_seed(SEED)

        self.series_ = ensure_series(series)
        self.scaler_ = StandardScaler()

        scaled = self.scaler_.fit_transform(
            self.series_.to_numpy().reshape(-1, 1)
        ).reshape(-1)

        X_raw, y = make_one_step_windows(
            scaled,
            self.window_size,
        )

        if len(X_raw) < 12:
            raise ValueError("Too few one-step windows for ARKAN.")

        split = min(
            max(int(0.8 * len(X_raw)), 1),
            len(X_raw) - 1,
        )

        # Linear autoregressive relevance weights
        ar = LinearRegression(
            fit_intercept=False
        ).fit(
            X_raw[:split],
            y[:split],
        )

        self.ar_weights_ = ar.coef_.reshape(-1)
        X = X_raw * self.ar_weights_[None, :]

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = torch.float64

        X_train = torch.tensor(
            X[:split],
            dtype=dtype,
        )
        y_train = torch.tensor(
            y[:split, None],
            dtype=dtype,
        )

        X_val = torch.tensor(
            X[split:],
            dtype=dtype,
            device=device,
        )
        y_val = torch.tensor(
            y[split:, None],
            dtype=dtype,
            device=device,
        )

        p = self.params

        width = [
            self.window_size,
            p["h1"],
            1,
        ]

        if p["n_hidden_layers"] == 2:
            width = [
                self.window_size,
                p["h1"],
                p["h2"],
                1,
            ]

        KANClass = self._kan_class()

        # MultKAN.to() accepts a device only. Pass the device through
        # the constructor and use Module.double() for float64.
        model = KANClass(
            width=width,
            grid=p["grid"],
            k=p["k"],
            auto_save=False,
            device=str(device),
        )

        model = model.double()

        # Recommended by pykan when using a custom training loop and
        # not using symbolic regression.
        if hasattr(model, "speed"):
            model.speed()

        train_dataset = TensorDataset(
            X_train,
            y_train,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(
                p["batch_size"],
                len(train_dataset),
            ),
            shuffle=False,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=p["learning_rate"],
            weight_decay=p["weight_decay"],
        )

        criterion = nn.MSELoss()

        best_state = None
        best_loss = np.inf
        bad = 0

        for _ in range(p["epochs"]):
            model.train()

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad(set_to_none=True)

                prediction = model(x_batch)
                loss = criterion(prediction, y_batch)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "ARKAN produced a non-finite training loss."
                    )

                loss.backward()

                if p["clip_grad"] is not None and p["clip_grad"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=p["clip_grad"],
                    )

                optimizer.step()

            model.eval()

            with torch.no_grad():
                val_prediction = model(X_val)
                val_loss = float(
                    criterion(
                        val_prediction,
                        y_val,
                    ).item()
                )

            if val_loss < best_loss - 1e-10:
                best_loss = val_loss
                bad = 0
                best_state = copy.deepcopy(
                    model.state_dict()
                )
            else:
                bad += 1

                if bad >= p["patience"]:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model_ = model
        self.device_ = device
        self.dtype_ = dtype
        self.scaled_history_ = list(scaled)

        return self

    def predict(self, horizon: int):
        import torch

        history = list(self.scaled_history_)
        predictions = []

        self.model_.eval()

        with torch.no_grad():
            for _ in range(int(horizon)):
                raw = np.asarray(
                    history[-self.window_size:],
                    dtype=np.float64,
                )

                weighted = raw * self.ar_weights_

                x = torch.tensor(
                    weighted.reshape(1, -1),
                    dtype=self.dtype_,
                    device=self.device_,
                )

                value = float(
                    self.model_(x)
                    .reshape(-1)[0]
                    .detach()
                    .cpu()
                    .item()
                )

                predictions.append(value)
                history.append(value)

        forecast = self.scaler_.inverse_transform(
            np.asarray(predictions).reshape(-1, 1)
        ).reshape(-1)

        return pd.Series(
            forecast,
            index=make_future_index(
                self.series_,
                horizon,
            ),
            name="pred",
        )

class CallableBaseline(ForecastBaseline):
    """Adapter for Chronos, TimesFM, Moirai, or any external predictor."""

    def __init__(self, model_or_factory: Any):
        self.model_or_factory = model_or_factory

    def fit(self, series: Any):
        self.series_ = ensure_series(series)
        self.model_ = self.model_or_factory() if callable(self.model_or_factory) else self.model_or_factory
        if hasattr(self.model_, "fit"):
            try:
                self.model_.fit(self.series_)
            except TypeError:
                pass
        return self

    def predict(self, horizon: int):
        return recursive_point_forecast(self.model_, self.series_, horizon)


class ChronosBaseline(ForecastBaseline):
    """Zero-shot Chronos baseline with optional custom/mock predictor support."""

    def __init__(
        self,
        model_or_factory: Any = None,
        predict_fn: Optional[Callable] = None,
        model_id: str = "amazon/chronos-bolt-small",
        device_map: str = "cpu",
        torch_dtype: Any = None,
        prediction_kwargs: Optional[dict] = None,
    ):
        self.model_or_factory = model_or_factory
        self.predict_fn = predict_fn
        self.chronos_kwargs = {
            "model_id": model_id,
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "prediction_kwargs": prediction_kwargs,
        }

    def fit(self, series: Any):
        self.series_ = ensure_series(series)
        self.model_ = make_chronos_predictor(
            model_or_factory=self.model_or_factory,
            predict_fn=self.predict_fn,
            **self.chronos_kwargs,
        )
        return self

    def predict(self, horizon: int):
        return chronos_predict_fixed_horizon(self.model_, self.series_, horizon)

    def trainable_parameter_count(self) -> int:
        return 0


BASELINE_ALIASES = {
    "KRR": "KernelRidge",
    "GPR": "GaussianProcess",
    "AutoReg": "AR",
    "Seasonal Naive": "SeasonalNaive",
    "ChronosZeroShot": "Chronos",
}


def canonical_baseline_name(model_name: str) -> str:
    return BASELINE_ALIASES.get(model_name, model_name)


BASELINE_CLASSES = {
    "SeasonalNaive": SeasonalNaiveBaseline,
    "ETS": ETSBaseline,
    "ARIMA": ARIMABaseline,
    "AR": AutoRegBaseline,
    "KernelRidge": KernelRidgeBaseline,
    "KernelRidgeRegressor": KernelRidgeBaseline,
    "GaussianProcess": GaussianProcessBaseline,
    "LSTM": lambda **p: KerasDirectBaseline("LSTM", **p),
    "CNN": lambda **p: KerasDirectBaseline("CNN", **p),
    "NLinear": lambda **p: TorchLinearBaseline("NLinear", **p),
    "DLinear": lambda **p: TorchLinearBaseline("DLinear", **p),
    "ARKAN": ARKANBaseline,
    "Chronos": ChronosBaseline,
}


def build_baseline(model_name: str, params: Optional[dict] = None) -> ForecastBaseline:
    params = dict(params or {})
    model_name = canonical_baseline_name(model_name)
    if model_name not in BASELINE_CLASSES:
        raise ValueError(f"Unknown baseline {model_name!r}.")
    return BASELINE_CLASSES[model_name](**params)


def fit_predict_baseline(model_name: str, train_series: Any, horizon: int, params: dict):
    model = build_baseline(model_name, params)
    if isinstance(model, (KernelRidgeBaseline, GaussianProcessBaseline, KerasDirectBaseline)):
        model.fit(train_series, horizon=horizon)
    else:
        model.fit(train_series)
    return model, model.predict(horizon)


def suggest_baseline_params(trial, model_name: str, train_length: int, horizon: int) -> dict:
    model_name = canonical_baseline_name(model_name)
    max_window = max(4, min(48, train_length - horizon - 2))
    if model_name == "SeasonalNaive":
        return {"seasonal_period": trial.suggest_categorical("seasonal_period", [1, 4, 12])}
    if model_name == "ETS":
        return {
            "error": "add",
            "trend": trial.suggest_categorical("trend", [None, "add"]),
            "seasonal": trial.suggest_categorical("seasonal", [None, "add"]),
            "seasonal_periods": 12,
            "damped_trend": trial.suggest_categorical("damped_trend", [False, True]),
        }
    if model_name == "ARIMA":
        return {
            "p": trial.suggest_int("p", 0, 6),
            "d": trial.suggest_int("d", 0, 2),
            "q": trial.suggest_int("q", 0, 6),
            "trend": trial.suggest_categorical("trend", ["n", "c", "t", "ct"]),
        }
    if model_name == "AR":
        return {
            "lags": trial.suggest_int("lags", 1, max(1, min(36, train_length // 3))),
            "trend": trial.suggest_categorical("trend", ["n", "c", "t", "ct"]),
            "seasonal": trial.suggest_categorical("seasonal", [False, True]),
            "period": 12,
        }
    if model_name in {"KernelRidge", "KernelRidgeRegressor"}:
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
        params = {
            "input_window": trial.suggest_int("input_window", 4, max_window),
            "alpha": trial.suggest_float("alpha", 1e-4, 50.0, log=True),
            "kernel": kernel,
        }
        if kernel in {"rbf", "poly"}:
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)
            params["coef0"] = trial.suggest_float("coef0", 0.0, 2.0)
        return params
    if model_name == "GaussianProcess":
        return {
            "input_window": trial.suggest_int("input_window", 4, min(24, max_window)),
            "kernel_type": trial.suggest_categorical("kernel_type", ["rbf", "matern32", "matern52"]),
            "length_scale": trial.suggest_float("length_scale", 1e-2, 20.0, log=True),
            "constant_value": trial.suggest_float("constant_value", 0.1, 10.0, log=True),
            "noise_level": trial.suggest_float("noise_level", 1e-6, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-10, 1e-2, log=True),
            "normalize_y": trial.suggest_categorical("normalize_y", [True, False]),
        }
    if model_name == "LSTM":
        layers = trial.suggest_categorical("n_layers", [1, 2])
        return {
            "input_window": trial.suggest_int("input_window", 4, max_window),
            "n_layers": layers,
            "units_1": trial.suggest_categorical("units_1", [16, 32, 64]),
            "units_2": trial.suggest_categorical("units_2", [8, 16, 32]) if layers == 2 else 0,
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "dense_units": trial.suggest_categorical("dense_units", [16, 32, 64]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "epochs": trial.suggest_int("epochs", 40, 160),
        }
    if model_name == "CNN":
        return {
            "input_window": trial.suggest_int("input_window", 4, max_window),
            "filters": trial.suggest_categorical("filters", [16, 32, 64]),
            "kernel_size": trial.suggest_int("kernel_size", 2, 6),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "dense_units": trial.suggest_categorical("dense_units", [16, 32, 64]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "epochs": trial.suggest_int("epochs", 40, 160),
        }
    if model_name in {"NLinear", "DLinear"}:
        params = {
            "window_size": trial.suggest_int("window_size", 4, max_window),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 50, 200),
            "patience": trial.suggest_int("patience", 8, 25),
        }
        if model_name == "DLinear":
            params["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])
        return params
    if model_name == "ARKAN":
        layers = trial.suggest_categorical("n_hidden_layers", [1, 2])
        return {
            "window_size": trial.suggest_int("window_size", 4, min(24, max_window)),
            "h1": trial.suggest_categorical("h1", [4, 8, 16]),
            "h2": trial.suggest_categorical("h2", [4, 8]) if layers == 2 else 4,
            "n_hidden_layers": layers,
            "grid": trial.suggest_categorical("grid", [3, 5]),
            "k": trial.suggest_categorical("k", [2, 3]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 40, 120),
            "patience": trial.suggest_int("patience", 8, 20),
        }
    raise ValueError(f"No search space is defined for {model_name!r}.")


@dataclass
class BaselineExperimentResult:
    model_name: str
    best_params: dict
    validation_score: float
    model: ForecastBaseline
    forecast: Any
    metrics: dict
    study: Any


def optimize_baseline(
    model_name: str,
    train_series: Any,
    *,
    val_size: int = 12,
    n_trials: int = 50,
    seed: int = 42,
    objective_metric: str = "mape",
):
    import optuna

    full = ensure_series(train_series)
    if len(full) <= val_size + 15:
        val_size = max(3, len(full) // 4)
    train, validation = split_train_val(full, val_size)

    def objective(trial):
        params = suggest_baseline_params(trial, model_name, len(train), len(validation))
        try:
            _, forecast = fit_predict_baseline(model_name, train, len(validation), params)
            metrics = evaluate_forecast(validation, parse_forecast_output(forecast).pred)
            score = metrics[objective_metric]
            return float(score) if np.isfinite(score) else float("inf")
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    return study


def evaluate_baseline(
    model_name: str,
    train_series: Any,
    test_series: Any,
    params: dict,
    *,
    seasonal_period: int = 12,
):
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    start = time.perf_counter()
    model, forecast = fit_predict_baseline(model_name, train, len(test), params)
    elapsed = time.perf_counter() - start
    out = parse_forecast_output(forecast)
    metrics = evaluate_forecast(
        test,
        out.pred,
        y_train=train,
        lower=out.lower,
        upper=out.upper,
        seasonal_period=seasonal_period,
        elapsed_seconds=elapsed,
        trainable_params=count_trainable_parameters(model),
    )
    return {"model": model, "forecast": forecast, "metrics": metrics}


def tune_and_test_baseline(
    model_name: str,
    train_series: Any,
    test_series: Any,
    *,
    val_size: int = 12,
    n_trials: int = 50,
    seed: int = 42,
):
    study = optimize_baseline(
        model_name,
        train_series,
        val_size=val_size,
        n_trials=n_trials,
        seed=seed,
    )
    evaluation = evaluate_baseline(model_name, train_series, test_series, study.best_params)
    return BaselineExperimentResult(
        model_name=model_name,
        best_params=study.best_params,
        validation_score=float(study.best_value),
        model=evaluation["model"],
        forecast=evaluation["forecast"],
        metrics=evaluation["metrics"],
        study=study,
    )


def run_baseline_suite(
    train_series: Any,
    test_series: Any,
    trials_by_model: Optional[dict[str, int]] = None,
    *,
    val_size: int = 12,
    seed: int = 42,
):
    trials_by_model = trials_by_model or {
        "SeasonalNaive": 3,
        "ETS": 30,
        "ARIMA": 50,
        "AR": 40,
        "KernelRidge": 40,
        "GaussianProcess": 30,
        "LSTM": 20,
        "CNN": 20,
        "NLinear": 20,
        "DLinear": 20,
        "ARKAN": 15,
        "Chronos": 1,
    }
    results, rows = {}, []
    for name, trials in trials_by_model.items():
        result = tune_and_test_baseline(
            name,
            train_series,
            test_series,
            val_size=val_size,
            n_trials=trials,
            seed=seed,
        )
        results[name] = result
        rows.append({"model": name, "val_score": result.validation_score, **result.metrics, "best_params": result.best_params})
    return results, pd.DataFrame(rows).sort_values(["rmse", "mape_percent"])
