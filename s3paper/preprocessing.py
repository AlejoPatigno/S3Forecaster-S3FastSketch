"""Invertible transformations fitted on training data only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class InvertibleTransformer:
    def fit(self, train_data: Any) -> "InvertibleTransformer":
        return self

    def transform(self, new_data: Any) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, transformed_data: Any) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, train_data: Any) -> np.ndarray:
        self.fit(train_data)
        return self.transform(train_data)


class IdentityTransformer(InvertibleTransformer):
    def transform(self, new_data: Any) -> np.ndarray:
        return np.asarray(new_data, dtype=float)

    def inverse_transform(self, transformed_data: Any) -> np.ndarray:
        return np.asarray(transformed_data, dtype=float)


@dataclass
class PositiveLogTransformer(InvertibleTransformer):
    eps: float = 1e-8
    offset_: float = 0.0

    def fit(self, train_data: Any) -> "PositiveLogTransformer":
        values = np.asarray(train_data, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("PositiveLogTransformer requires finite training data.")
        minimum = float(np.min(finite))
        self.offset_ = max(0.0, self.eps - minimum)
        return self

    def transform(self, new_data: Any) -> np.ndarray:
        values = np.asarray(new_data, dtype=float)
        shifted = values + self.offset_
        if np.any(shifted <= 0):
            raise ValueError("Log transform received values outside the fitted positive domain.")
        return np.log(shifted)

    def inverse_transform(self, transformed_data: Any) -> np.ndarray:
        return np.exp(np.asarray(transformed_data, dtype=float)) - self.offset_


@dataclass
class FrozenRobustScaler:
    """Column-wise robust scaler with explicit fit/transform separation."""

    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, train_data: Any) -> "FrozenRobustScaler":
        x = np.asarray(train_data, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.center_ = np.nanmedian(x, axis=0)
        q75 = np.nanpercentile(x, 75, axis=0)
        q25 = np.nanpercentile(x, 25, axis=0)
        scale = (q75 - q25) / 1.35
        scale = np.where(np.isfinite(scale) & (np.abs(scale) > 1e-8), scale, 1.0)
        self.scale_ = scale
        return self

    def transform(self, new_data: Any) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("FrozenRobustScaler must be fitted before transform().")
        x = np.asarray(new_data, dtype=float)
        original_ndim = x.ndim
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        out = (x - self.center_) / self.scale_
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.reshape(-1) if original_ndim == 1 and len(self.center_) == 1 else out

    def fit_transform(self, train_data: Any) -> np.ndarray:
        self.fit(train_data)
        return self.transform(train_data)
