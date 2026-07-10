"""Sequential adaptive conformal intervals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def finalize_symmetric_interval(
    point_prediction: float,
    raw_interval: tuple[float, float],
    *,
    interval_scale: float = 1.0,
    minimum_width: float = 0.0,
) -> tuple[float, float, float]:
    point = float(point_prediction)
    lower_raw, upper_raw = map(float, raw_interval)
    if not np.isfinite(point):
        raise ValueError(f"point_prediction must be finite, got {point_prediction}")
    if not (np.isfinite(lower_raw) and np.isfinite(upper_raw)):
        raise ValueError(f"raw_interval bounds must be finite, got {raw_interval}")

    if float(interval_scale) <= 0.0:
        raise ValueError(f"interval_scale must be > 0, got {interval_scale}")
    if float(minimum_width) < 0.0:
        raise ValueError(f"minimum_width must be >= 0, got {minimum_width}")

    raw_half_width = max(point - lower_raw, upper_raw - point, 0.0)
    final_half_width = max(
        float(interval_scale) * raw_half_width,
        float(minimum_width),
    )

    lower = point - final_half_width
    upper = point + final_half_width

    if lower > point or upper < point:
        raise ValueError("Invalid interval generated")
    if not np.isclose(upper - point, point - lower, atol=1e-5):
        raise ValueError("Interval is not symmetric")

    return lower, upper, final_half_width


def finite_sample_quantile(scores: Any, alpha: float) -> float:
    """Return the finite-sample conformal order statistic.

    Uses the conservative split-conformal rank ceil((n + 1) * (1 - alpha)).
    If the rank exceeds n, the largest observed score is returned.
    """

    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    rank = int(np.ceil((values.size + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), values.size)
    return float(np.sort(values)[rank - 1])


@dataclass
class SequentialACI:
    target_miscoverage: float = 0.10
    step_size: float = 0.05
    min_alpha: float = 0.01
    max_alpha: float = 0.50
    scores: list[float] = field(default_factory=list)
    alpha_t: float | None = None

    def __post_init__(self) -> None:
        if self.alpha_t is None:
            self.alpha_t = float(self.target_miscoverage)

    def fit_calibration(self, y_true: Any, y_pred: Any) -> "SequentialACI":
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        if len(yt) != len(yp):
            raise ValueError("y_true and y_pred must have matching lengths.")
        self.scores = []
        self.alpha_t = float(self.target_miscoverage)
        for truth, pred in zip(yt, yp):
            interval = self.interval(float(pred))
            self.update(float(truth), float(pred), interval=interval)
        return self

    def interval(self, point_prediction: float) -> tuple[float, float]:
        width = finite_sample_quantile(self.scores, float(self.alpha_t))
        point = float(point_prediction)
        return point - width, point + width

    def update(
        self,
        y_true: float,
        point_prediction: float,
        *,
        interval: tuple[float, float] | None = None,
    ) -> "SequentialACI":
        if interval is None:
            interval = self.interval(point_prediction)
        lower, upper = interval
        miss = int(float(y_true) < lower or float(y_true) > upper)
        self.alpha_t = float(
            np.clip(
                float(self.alpha_t) + self.step_size * (self.target_miscoverage - miss),
                self.min_alpha,
                self.max_alpha,
            )
        )
        self.scores.append(abs(float(y_true) - float(point_prediction)))
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "target_miscoverage": float(self.target_miscoverage),
            "step_size": float(self.step_size),
            "min_alpha": float(self.min_alpha),
            "max_alpha": float(self.max_alpha),
            "scores": list(map(float, self.scores)),
            "alpha_t": float(self.alpha_t),
        }

    def load_state_dict(self, state: dict[str, Any]) -> "SequentialACI":
        self.target_miscoverage = float(state["target_miscoverage"])
        self.step_size = float(state["step_size"])
        self.min_alpha = float(state.get("min_alpha", 0.01))
        self.max_alpha = float(state.get("max_alpha", 0.50))
        self.scores = list(map(float, state.get("scores", [])))
        self.alpha_t = float(state.get("alpha_t", self.target_miscoverage))
        return self
