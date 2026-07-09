"""Calibration-only adapter selection rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def residual_predictability_score(y_true_residual: Any, y_pred_residual: Any) -> float:
    truth = np.asarray(y_true_residual, dtype=float).reshape(-1)
    pred = np.asarray(y_pred_residual, dtype=float).reshape(-1)
    if len(truth) != len(pred):
        raise ValueError("Residual truth and prediction arrays must have equal length.")
    denominator = float(np.sum((truth - np.mean(truth)) ** 2) + 1e-12)
    numerator = float(np.sum((truth - pred) ** 2))
    return float(1.0 - numerator / denominator)


@dataclass
class AdapterSelectionRule:
    """Conservative calibration-only activation rule.

    Defaults are permissive enough to preserve legacy smoke-test behavior while
    still recording every decision. Paper runs can tighten these thresholds in
    configuration without changing model code.
    """

    r2_threshold: float = -np.inf
    minimum_improvement: float = -np.inf
    complexity_penalty: float = 0.0

    def decide(
        self,
        *,
        r2_res_cal: float,
        delta_cal: float,
        prior_loss: float,
        adapted_loss: float,
        reason_prefix: str = "calibration_rule",
    ) -> dict[str, Any]:
        checks = {
            "r2_res_cal": float(r2_res_cal),
            "delta_cal": float(delta_cal),
            "prior_loss": float(prior_loss),
            "adapted_loss": float(adapted_loss),
            "complexity_penalty": float(self.complexity_penalty),
            "r2_threshold": float(self.r2_threshold),
            "minimum_improvement": float(self.minimum_improvement),
        }
        pass_r2 = checks["r2_res_cal"] > checks["r2_threshold"]
        pass_delta = checks["delta_cal"] > checks["minimum_improvement"]
        pass_penalty = (
            checks["adapted_loss"] + checks["complexity_penalty"]
            < checks["prior_loss"]
            or not np.isfinite(checks["prior_loss"])
        )
        activated = bool(pass_r2 and pass_delta and pass_penalty)
        failed = [
            name
            for name, ok in (
                ("r2_threshold", pass_r2),
                ("minimum_improvement", pass_delta),
                ("complexity_penalty", pass_penalty),
            )
            if not ok
        ]
        checks.update(
            {
                "activated": activated,
                "reason": f"{reason_prefix}:activated"
                if activated
                else f"{reason_prefix}:failed:{','.join(failed)}",
            }
        )
        return checks


def select_family(calibration_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Choose prior-only, S3, or FastSketch from calibration-only reports."""

    candidates = []
    for family, report in calibration_reports.items():
        loss = float(report.get("adapted_loss", report.get("prior_loss", np.inf)))
        if family == "prior_only":
            loss = float(report.get("prior_loss", loss))
        candidates.append((loss, family, report))
    if not candidates:
        return {"family": "prior_only", "reason": "no_candidates"}
    loss, family, report = min(candidates, key=lambda item: item[0])
    return {
        "family": family,
        "selected_loss": float(loss),
        "reason": "minimum_calibration_loss",
        "report": dict(report),
    }
