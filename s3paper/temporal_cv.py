"""Temporal cross-validation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import ensure_series


def make_expanding_window_folds(
    series: Any,
    *,
    n_folds: int = 3,
    initial_train_size: int | None = None,
    validation_size: int = 6,
    step_size: int | None = None,
) -> list[tuple[object, object]]:
    y = ensure_series(series)
    validation_size = max(1, int(validation_size))
    n_folds = max(1, int(n_folds))
    step = validation_size if step_size is None else max(1, int(step_size))
    if initial_train_size is None:
        initial_train_size = max(validation_size * 2, len(y) - validation_size - step * (n_folds - 1))
    initial_train_size = int(initial_train_size)
    folds = []
    for fold in range(n_folds):
        train_end = initial_train_size + fold * step
        val_end = train_end + validation_size
        if train_end <= 1 or val_end > len(y):
            continue
        folds.append((y.iloc[:train_end], y.iloc[train_end:val_end]))
    return folds


def aggregate_scores(values: list[float], method: str = "median", trim_fraction: float = 0.10) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("inf")
    if method == "median":
        return float(np.median(arr))
    if method == "trimmed_mean":
        trim = int(np.floor(len(arr) * float(trim_fraction)))
        arr = np.sort(arr)
        if trim > 0 and len(arr) > 2 * trim:
            arr = arr[trim:-trim]
        return float(np.mean(arr))
    raise ValueError("aggregation must be 'median' or 'trimmed_mean'.")
