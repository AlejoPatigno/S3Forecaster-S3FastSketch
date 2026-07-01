"""High-level orchestration utilities for the experiments reported in the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .metrics import paper_metric_row
from .s3_fastsketch_experiment import evaluate_fastsketch
from .s3_forecaster_experiment import evaluate_s3_forecaster


def evaluate_proposed_models(
    train_series: Any,
    test_series: Any,
    *,
    s3_params: dict,
    fastsketch_params: dict,
    s3_uq_params: Optional[dict] = None,
    fastsketch_uq_params: Optional[dict] = None,
    seasonal_period: int = 12,
    alpha: float = 0.10,
):
    """Evaluate both proposed approaches under one common protocol."""

    s3 = evaluate_s3_forecaster(
        train_series,
        test_series,
        s3_params,
        uq_params=s3_uq_params,
        seasonal_period=seasonal_period,
        alpha=alpha,
    )
    fast = evaluate_fastsketch(
        train_series,
        test_series,
        fastsketch_params,
        uq_params=fastsketch_uq_params,
        seasonal_period=seasonal_period,
        alpha=alpha,
    )
    table = pd.DataFrame(
        [
            paper_metric_row("S3-Forecaster", s3["metrics"]),
            paper_metric_row("S3-FastSketch", fast["metrics"]),
        ]
    )
    return {
        "table": table,
        "S3-Forecaster": s3,
        "S3-FastSketch": fast,
    }


def save_experiment_tables(
    output_directory: str | Path,
    **named_tables: pd.DataFrame,
) -> dict[str, Path]:
    """Persist publication tables with stable filenames."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, table in named_tables.items():
        if table is None:
            continue
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths
