"""Multi-series statistical analysis derived from the canonical result store."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, t


def _iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def bootstrap_ci(
    values: Any,
    *,
    statistic=np.mean,
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(n_boot)):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats.append(float(statistic(sample)))
    alpha = (1.0 - confidence) / 2.0
    return tuple(np.quantile(stats, [alpha, 1.0 - alpha]).astype(float))


def summarize_metric_by_model(
    per_series_metrics: pd.DataFrame,
    *,
    metric: str = "mase",
    higher_is_better: bool = False,
    n_boot: int = 1000,
) -> pd.DataFrame:
    rows = []
    valid = per_series_metrics.copy()
    valid = valid[np.isfinite(valid[metric])]
    rank_ascending = not higher_is_better
    ranks = (
        valid.pivot_table(index=["dataset", "series_id"], columns="model", values=metric, aggfunc="first")
        .rank(axis=1, method="average", ascending=rank_ascending)
        .stack()
        .rename("rank")
        .reset_index()
    )
    valid = valid.merge(ranks, on=["dataset", "series_id", "model"], how="left")
    for (dataset, model), group in valid.groupby(["dataset", "model"], dropna=False):
        values = group[metric].astype(float)
        low, high = bootstrap_ci(values, n_boot=n_boot)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "metric": metric,
                "successful_series": int(values.notna().sum()),
                "failures": int((per_series_metrics["dataset"].eq(dataset) & per_series_metrics["model"].eq(model)).sum() - values.notna().sum()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "iqr": _iqr(values),
                "ci95_low": low,
                "ci95_high": high,
                "average_rank": float(group["rank"].mean()),
            }
        )
    return pd.DataFrame(rows)


def friedman_test(per_series_metrics: pd.DataFrame, *, metric: str = "mase") -> dict[str, float]:
    pivot = per_series_metrics.pivot_table(index=["dataset", "series_id"], columns="model", values=metric, aggfunc="first")
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        return {"statistic": np.nan, "p_value": np.nan, "n_series": int(pivot.shape[0]), "n_models": int(pivot.shape[1]), "status": "insufficient_data"}
    statistic, p_value = friedmanchisquare(*[pivot[column].to_numpy(dtype=float) for column in pivot.columns])
    return {"statistic": float(statistic), "p_value": float(p_value), "n_series": int(pivot.shape[0]), "n_models": int(pivot.shape[1]), "status": "ok"}


def holm_posthoc(per_series_metrics: pd.DataFrame, *, metric: str = "mase", baseline: str | None = None) -> pd.DataFrame:
    pivot = per_series_metrics.pivot_table(index=["dataset", "series_id"], columns="model", values=metric, aggfunc="first")
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return pd.DataFrame(columns=["model_a", "model_b", "mean_difference", "p_value", "p_holm", "status"])
    pairs = [(baseline, model) for model in pivot.columns if model != baseline] if baseline in pivot.columns else list(combinations(pivot.columns, 2))
    rows = []
    for a, b in pairs:
        diff = pivot[a].to_numpy(dtype=float) - pivot[b].to_numpy(dtype=float)
        se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan
        if not np.isfinite(se) or se <= 0:
            p_value = 1.0 if np.allclose(diff, 0) else 0.0
        else:
            statistic = float(np.mean(diff) / se)
            p_value = float(2.0 * t.sf(abs(statistic), df=len(diff) - 1))
        rows.append({"model_a": a, "model_b": b, "mean_difference": float(np.mean(diff)), "p_value": p_value})
    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    m = len(out)
    adjusted = []
    running = 0.0
    for rank, p_value in enumerate(out["p_value"], start=1):
        corrected = min(1.0, (m - rank + 1) * float(p_value))
        running = max(running, corrected)
        adjusted.append(running)
    out["p_holm"] = adjusted
    out["status"] = "ok"
    return out


def paired_bootstrap_difference(
    per_series_metrics: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    metric: str = "mase",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    pivot = per_series_metrics.pivot_table(index=["dataset", "series_id"], columns="model", values=metric, aggfunc="first")
    pair = pivot[[model_a, model_b]].dropna()
    diff = pair[model_a].to_numpy(dtype=float) - pair[model_b].to_numpy(dtype=float)
    low, high = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    return {"model_a": model_a, "model_b": model_b, "metric": metric, "n_pairs": int(len(diff)), "mean_difference": float(np.mean(diff)) if len(diff) else np.nan, "ci95_low": low, "ci95_high": high}


def diebold_mariano_hln(loss_a: Any, loss_b: Any, *, horizon: int = 1) -> dict[str, float | str]:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("loss_a and loss_b must have the same length.")
    d = a - b
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8:
        return {"statistic": np.nan, "p_value": np.nan, "n": int(n), "status": "low_power"}
    variance = float(np.var(d, ddof=1))
    if variance <= 1e-12:
        return {"statistic": 0.0, "p_value": 1.0, "n": int(n), "status": "low_power" if n < 24 else "ok"}
    dm = float(np.mean(d) / np.sqrt(variance / n))
    h = max(1, int(horizon))
    correction = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0.0))
    statistic = dm * correction
    p_value = float(2.0 * t.sf(abs(statistic), df=n - 1))
    return {"statistic": float(statistic), "p_value": p_value, "n": int(n), "status": "low_power" if n < 24 else "ok"}


def coverage_breakdowns(result_store: pd.DataFrame) -> dict[str, pd.DataFrame]:
    store = result_store.copy()
    covered = (store["y_true"] >= store["lower"]) & (store["y_true"] <= store["upper"])
    store["covered"] = covered.astype(float)
    store["abs_error"] = np.abs(store["y_true"] - store["y_pred"])
    store["width"] = store["upper"] - store["lower"]
    pooled = store.groupby(["dataset", "model"], as_index=False)["covered"].mean().rename(columns={"covered": "pooled_marginal_coverage"})
    per_series = store.groupby(["dataset", "series_id", "model"], as_index=False)["covered"].mean().rename(columns={"covered": "series_coverage"})
    store["volatility_quartile"] = store.groupby(["dataset", "series_id"])["abs_error"].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=min(4, len(s)), labels=False, duplicates="drop")
    )
    by_volatility = store.groupby(["dataset", "model", "volatility_quartile"], as_index=False)["covered"].mean()
    return {"pooled": pooled, "per_series": per_series, "by_volatility_quartile": by_volatility}
