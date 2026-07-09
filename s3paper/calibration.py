"""Chronological internal calibration splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_series


@dataclass(frozen=True)
class InternalCalibrationSplit:
    readout_train: pd.Series
    adapter_calibration: pd.Series
    conformal_calibration: pd.Series

    def report(self) -> dict[str, Any]:
        def bounds(block: pd.Series, prefix: str) -> dict[str, Any]:
            return {
                f"{prefix}_start": None if block.empty else block.index[0],
                f"{prefix}_end": None if block.empty else block.index[-1],
            }

        out = {
            "n_readout_samples": int(len(self.readout_train)),
            "n_adapter_calibration": int(len(self.adapter_calibration)),
            "n_conformal_calibration": int(len(self.conformal_calibration)),
        }
        out.update(bounds(self.readout_train, "readout"))
        out.update(bounds(self.adapter_calibration, "adapter"))
        out.update(bounds(self.conformal_calibration, "conformal"))
        return out


def split_internal_calibration(
    series: Any,
    *,
    readout_ratio: float = 0.60,
    adapter_ratio: float = 0.20,
    minimum_readout: int = 12,
    minimum_adapter: int = 4,
    minimum_conformal: int = 4,
) -> InternalCalibrationSplit:
    y = ensure_series(series)
    n = len(y)
    required = int(minimum_readout) + int(minimum_adapter) + int(minimum_conformal)
    if n < required:
        raise ValueError(f"insufficient_data_for_internal_split: n={n}, required={required}")

    readout_end = int(np.clip(round(n * float(readout_ratio)), minimum_readout, n - minimum_adapter - minimum_conformal))
    adapter_end = int(
        np.clip(
            readout_end + round(n * float(adapter_ratio)),
            readout_end + minimum_adapter,
            n - minimum_conformal,
        )
    )
    return InternalCalibrationSplit(
        readout_train=y.iloc[:readout_end],
        adapter_calibration=y.iloc[readout_end:adapter_end],
        conformal_calibration=y.iloc[adapter_end:],
    )
