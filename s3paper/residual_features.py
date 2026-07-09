"""Causal residual feature transformers for S3 adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import FrozenRobustScaler


def _clean(values: Any) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class EchoStateResidualTransformer:
    reservoir_size: int = 50
    spectral_radius: float = 0.9
    leak_rate: float = 0.5
    ar_lags: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        self.reservoir_size = int(self.reservoir_size)
        self.spectral_radius = float(min(max(self.spectral_radius, 1e-6), 0.999))
        self.leak_rate = float(np.clip(self.leak_rate, 1e-6, 1.0))
        self.ar_lags = int(self.ar_lags)
        self.scaler_ = FrozenRobustScaler()
        self.feature_names_: list[str] = []
        self.feature_groups_: list[str] = []
        self._init_weights()

    def _init_weights(self) -> None:
        rng = np.random.default_rng(int(self.seed))
        self.W_in_ = rng.uniform(-1.0, 1.0, size=(self.reservoir_size, 2))
        raw = rng.normal(0.0, 1.0, size=(self.reservoir_size, self.reservoir_size))
        radius = float(np.max(np.abs(np.linalg.eigvals(raw))))
        if not np.isfinite(radius) or radius <= 0.0:
            radius = 1.0
        self.W_res_ = raw * (self.spectral_radius / radius)

    @property
    def achieved_spectral_radius_(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.W_res_))))

    def fit(self, residuals: Any) -> "EchoStateResidualTransformer":
        self.scaler_.fit(_clean(residuals))
        self._set_names()
        return self

    def fit_transform(self, residuals: Any) -> np.ndarray:
        self.fit(residuals)
        return self.transform(residuals)

    def transform(self, residuals: Any) -> np.ndarray:
        e = _clean(residuals)
        scaled = self.scaler_.transform(e).reshape(-1)
        states = np.zeros((len(e), self.reservoir_size), dtype=float)
        hidden = np.zeros(self.reservoir_size, dtype=float)
        for t, value in enumerate(scaled):
            augmented = np.asarray([1.0, value], dtype=float)
            proposal = np.tanh(self.W_in_ @ augmented + self.W_res_ @ hidden)
            hidden = (1.0 - self.leak_rate) * hidden + self.leak_rate * proposal
            states[t] = hidden
        ar = np.zeros((len(e), self.ar_lags), dtype=float)
        for lag in range(1, self.ar_lags + 1):
            ar[lag:, lag - 1] = e[:-lag]
        return np.hstack([states, ar])

    def _set_names(self) -> None:
        self.feature_names_ = [f"esn_state_{i}" for i in range(self.reservoir_size)] + [
            f"ar_lag_{lag}" for lag in range(1, self.ar_lags + 1)
        ]
        self.feature_groups_ = ["reservoir"] * self.reservoir_size + ["ar"] * self.ar_lags


@dataclass
class FastSketchResidualTransformer:
    ar_lags: int = 3
    ema_spans: tuple[int, ...] = (2, 4, 8, 12)
    conv_scales: tuple[int, ...] = (2, 3, 4, 6)
    seasonal_period: int = 12
    use_calendar: bool = True

    def __post_init__(self) -> None:
        self.scaler_ = FrozenRobustScaler()
        self.feature_names_: list[str] = []
        self.feature_groups_: list[str] = []

    def fit(self, residuals: Any) -> "FastSketchResidualTransformer":
        raw = self._raw_features(_clean(residuals))
        self.scaler_.fit(raw)
        return self

    def fit_transform(self, residuals: Any) -> np.ndarray:
        self.fit(residuals)
        return self.transform(residuals)

    def transform(self, residuals: Any) -> np.ndarray:
        raw = self._raw_features(_clean(residuals))
        return self.scaler_.transform(raw)

    @staticmethod
    def _causal_conv_last(x: np.ndarray, t: int, scale: int, kind: str) -> float:
        start = max(0, t - scale + 1)
        window = x[start : t + 1]
        if len(window) == 0:
            return 0.0
        if kind == "mean":
            return float(np.mean(window))
        if kind == "slope":
            return float(window[-1] - window[0]) if len(window) > 1 else 0.0
        if kind == "contrast":
            if len(window) < 2:
                return 0.0
            mid = len(window) // 2
            return float(np.mean(window[mid:]) - np.mean(window[:mid]))
        if kind == "energy":
            return float(np.mean(np.abs(window)))
        return 0.0

    def _append(self, features: list[np.ndarray], names: list[str], groups: list[str], col: Any, name: str, group: str) -> None:
        features.append(np.asarray(col, dtype=float).reshape(-1))
        names.append(name)
        groups.append(group)

    def _raw_features(self, e: np.ndarray) -> np.ndarray:
        n = len(e)
        features: list[np.ndarray] = []
        names: list[str] = []
        groups: list[str] = []

        for lag in range(1, int(self.ar_lags) + 1):
            col = np.zeros(n)
            col[lag:] = e[:-lag]
            self._append(features, names, groups, col, f"ar_lag_{lag}", "ar")

        for span in self.ema_spans:
            col = pd.Series(e).ewm(span=int(span), adjust=False).mean().to_numpy()
            self._append(features, names, groups, col, f"ema_resid_{span}", "ema_residual")

        abs_e = np.abs(e)
        for span in self.ema_spans:
            col = pd.Series(abs_e).ewm(span=int(span), adjust=False).mean().to_numpy()
            self._append(features, names, groups, col, f"ema_abs_{span}", "volatility")

        for span in self.ema_spans:
            span = int(span)
            col = np.zeros(n)
            if n > span:
                col[span:] = e[span:] - e[:-span]
            self._append(features, names, groups, col, f"diff_{span}", "momentum")

        for scale in self.conv_scales:
            for kind in ("mean", "slope", "contrast", "energy"):
                col = np.asarray([self._causal_conv_last(e, t, int(scale), kind) for t in range(n)])
                self._append(features, names, groups, col, f"conv_{kind}_{scale}", "causal_conv")

        m = int(self.seasonal_period)
        if m > 1:
            lag = np.zeros(n)
            gap = np.zeros(n)
            if n > m:
                lag[m:] = e[:-m]
                gap[m:] = e[m:] - e[:-m]
            self._append(features, names, groups, lag, f"seasonal_lag_{m}", "seasonal")
            self._append(features, names, groups, gap, f"seasonal_gap_{m}", "seasonal")

        if bool(self.use_calendar) and m > 1:
            t = np.arange(n)
            self._append(features, names, groups, np.sin(2.0 * np.pi * t / m), f"sin_period_{m}", "calendar")
            self._append(features, names, groups, np.cos(2.0 * np.pi * t / m), f"cos_period_{m}", "calendar")

        self.feature_names_ = names
        self.feature_groups_ = groups
        return np.column_stack(features) if features else np.zeros((n, 1))
