"""Evaluate the frozen baseline settings without hyperparameter search."""

import pandas as pd

from .baselines import evaluate_baseline
from .fixed_baseline_aliases import normalize_fixed_baseline
from .metrics import paper_metric_row


def evaluate_fixed_baselines(
    train_series,
    test_series,
    baseline_params,
    seasonal_period=12,
    include_models=None,
    skip_optional_failures=True,
):
    allowed = set(include_models) if include_models is not None else None
    evaluations, rows, failures = {}, [], []

    for display_name, raw_params in baseline_params.items():
        if allowed is not None and display_name not in allowed:
            continue
        model_name, params = normalize_fixed_baseline(display_name, raw_params)
        try:
            result = evaluate_baseline(
                model_name,
                train_series,
                test_series,
                params,
                seasonal_period=seasonal_period,
            )
            evaluations[display_name] = result
            row = paper_metric_row(display_name, result["metrics"])
            row.update({"status": "ok", "error": None})
        except Exception as exc:
            if not skip_optional_failures:
                raise
            row = paper_metric_row(display_name, {})
            row.update({"status": "skipped", "error": str(exc)})
            failures.append({"model": display_name, "error": str(exc)})
        rows.append(row)

    return {
        "evaluations": evaluations,
        "table": pd.DataFrame(rows),
        "failures": pd.DataFrame(failures),
    }
