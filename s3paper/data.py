"""Small utilities for consistent preparation of the paper datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

import numpy as np
import pandas as pd

from .utils import ensure_series


@dataclass(frozen=True)
class SeriesSplit:
    """Normalized train/test pair and its transformation metadata."""

    series_id: str
    train: pd.Series
    test: pd.Series
    transformation: str = "identity"


def apply_log_if_positive(
    train_series: Any,
    test_series: Any,
    *,
    enabled: bool = True,
) -> tuple[pd.Series, pd.Series, str]:
    """Apply the same log transformation used by the experimental notebooks."""

    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    can_log = bool(enabled and (train > 0).all() and (test > 0).all())
    if not can_log:
        return train, test, "identity"
    return (
        pd.Series(np.log(train.to_numpy()), index=train.index, name=train.name),
        pd.Series(np.log(test.to_numpy()), index=test.index, name=test.name),
        "log",
    )


def inverse_transform(values: Any, transformation: str) -> np.ndarray:
    """Return forecasts to the original scale."""

    array = np.asarray(values, dtype=float)
    if transformation == "identity":
        return array
    if transformation == "log":
        return np.exp(array)
    raise ValueError(f"Unsupported transformation={transformation!r}.")


def common_series_ids(
    train_series_map: Mapping,
    test_series_map: Mapping,
    *,
    series_ids: Optional[list] = None,
    max_series: Optional[int] = None,
) -> list:
    ids = [series_id for series_id in train_series_map if series_id in test_series_map]
    if series_ids is not None:
        allowed = set(series_ids)
        ids = [series_id for series_id in ids if series_id in allowed]
    if max_series is not None:
        ids = ids[: int(max_series)]
    return ids


def iter_series_splits(
    train_series_map: Mapping,
    test_series_map: Mapping,
    *,
    series_ids: Optional[list] = None,
    max_series: Optional[int] = None,
    log_if_positive: bool = False,
) -> Iterator[SeriesSplit]:
    """Yield consistently prepared series pairs from dataset dictionaries."""

    for series_id in common_series_ids(
        train_series_map,
        test_series_map,
        series_ids=series_ids,
        max_series=max_series,
    ):
        train, test, transformation = apply_log_if_positive(
            train_series_map[series_id],
            test_series_map[series_id],
            enabled=log_if_positive,
        )
        yield SeriesSplit(str(series_id), train, test, transformation)
