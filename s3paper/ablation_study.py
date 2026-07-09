"""Production-backed ablation studies for S3 models."""

from __future__ import annotations

import copy
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .metrics import evaluate_forecast, seasonal_naive_scale
from .priors import split_model_prior_params
from .rolling_protocol import evaluate_rolling_model
from .s3_fastsketch import S3FastSketchForecaster
from .s3_forecaster import S3Forecaster
from .utils import count_trainable_parameters, ensure_series


ABLATION_SHORT_LABELS = {
    "prior only": "Prior only",
    "no reservoir": "No reservoir",
    "no AR lags": "No AR",
    "no gate": "No gate",
    "static conformal": "Static conformal",
    "no adapter selector": "No selector",
    "full model": "Full",
    "Full": "Full",
    "Foundation only": "Prior only",
    "Without foundation": "No foundation",
    "Without AR lags": "No AR",
    "Without EMA residual": "No EMA",
    "Without volatility": "No volatility",
    "Without momentum": "No momentum",
    "Without causal convolutions": "No causal conv.",
    "Without seasonal residual": "No seasonal",
    "Without calendar features": "No calendar",
    "Without calendar": "No calendar",
    "Without multiscale sketch": "No sketch core",
    "Without bounded shrinkage": "No shrinkage",
    "Static conformal (without ACI)": "Static conformal",
}


S3_ABLATION_LABELS = {
    "prior_only": "prior only",
    "no_reservoir": "no reservoir",
    "no_ar_lags": "no AR lags",
    "no_gate": "no gate",
    "static_conformal": "static conformal",
    "no_adapter_selector": "no adapter selector",
    "full_model": "full model",
    "base_only": "prior only",
    "base_residual": "no gate",
    "base_residual_gate": "static conformal",
    "full_aci": "full model",
    "full_static": "static conformal",
}


FASTSKETCH_ABLATIONS = {
    "full model": {},
    "prior only": {"foundation_only": True},
    "no AR": {"disabled_groups": {"ar"}},
    "no EMA residual": {"disabled_groups": {"ema_residual"}},
    "no volatility": {"disabled_groups": {"volatility"}},
    "no momentum": {"disabled_groups": {"momentum"}},
    "no causal filters": {"disabled_groups": {"causal_conv"}},
    "no seasonal features": {"disabled_groups": {"seasonal"}},
    "no calendar": {"disabled_groups": {"calendar"}},
    "Without multiscale sketch": {
        "disabled_groups": {"ema_residual", "volatility", "momentum", "causal_conv"}
    },
    "no contraction": {"use_shrinkage": False},
    "static conformal": {"aci_step_size": 0.0},
    "no adapter selector": {"use_adapter_selector": False},
}


def shorten_ablation_label(label: Any, max_chars: int = 22) -> str:
    text = ABLATION_SHORT_LABELS.get(str(label), str(label))
    replacements = {
        "Without ": "No ",
        " residual": "",
        " features": "",
        "convolutions": "conv.",
        "adapter": "adapt.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "."


def _legacy_split_alias(params: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(params or {})
    if "oob_split_ratio" in out and "calibration_split_ratio" not in out:
        out["calibration_split_ratio"] = out.pop("oob_split_ratio")
    return out


def _normalize_fastsketch_tuple_params(params: dict[str, Any]) -> dict[str, Any]:
    out = _legacy_split_alias(params)
    for key in ("ema_spans", "conv_scales"):
        if isinstance(out.get(key), str):
            out[key] = tuple(int(part) for part in out[key].split(",") if part)
    return out


def _s3_mode_flags(mode: str) -> dict[str, Any]:
    if mode in {"prior_only", "base_only"}:
        return {"use_residual_adapter": False}
    if mode == "base_residual":
        return {"use_volatility_gate": False, "aci_step_size": 0.0}
    if mode == "base_residual_gate":
        return {"aci_step_size": 0.0}
    if mode == "no_reservoir":
        return {"disabled_groups": {"reservoir"}}
    if mode == "no_ar_lags":
        return {"disabled_groups": {"ar"}}
    if mode == "no_gate":
        return {"use_volatility_gate": False}
    if mode in {"static_conformal", "full_static"}:
        return {"aci_step_size": 0.0}
    if mode == "no_adapter_selector":
        return {"use_adapter_selector": False}
    return {"use_residual_adapter": True, "use_volatility_gate": True, "use_aci": True}


def run_s3_ablation_study(
    train_series: Any,
    test_series: Any,
    best_params: dict,
    *,
    alpha: float = 0.10,
    seasonal_period: int = 12,
    modes: Optional[list[str]] = None,
):
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    modes = modes or list(S3_ABLATION_LABELS)
    rows, forecasts, models = [], {}, {}

    for mode in modes:
        params = _legacy_split_alias(best_params)
        params.update(_s3_mode_flags(mode))
        params, _, _ = split_model_prior_params(params)
        start = time.perf_counter()
        try:
            model = S3Forecaster(
                horizon=1,
                target_miscoverage=alpha,
                **params,
            )
            result = evaluate_rolling_model(
                model,
                train,
                test,
                seasonal_period=seasonal_period,
                alpha=alpha,
            )
            forecast = result["forecast"]
            metrics = dict(result["metrics"])
            metrics["elapsed_seconds"] = time.perf_counter() - start
            metrics["trainable_params"] = count_trainable_parameters(model)
            row = {
                "variant": S3_ABLATION_LABELS.get(mode, mode),
                "mode": mode,
                "gate_value": float(forecast["gate"].iloc[0]) if "gate" in forecast else np.nan,
                "alpha_final": float(getattr(model, "current_alpha_t", np.nan)),
                "status": "ok",
                "error": None,
                **metrics,
            }
            forecasts[mode], models[mode] = forecast, model
        except Exception as exc:
            row = {
                "variant": S3_ABLATION_LABELS.get(mode, mode),
                "mode": mode,
                "status": "failed",
                "error": str(exc),
            }
            forecasts[mode], models[mode] = None, None
        rows.append(row)
    return {"results": pd.DataFrame(rows), "forecasts": forecasts, "models": models}


def run_fastsketch_ablation(
    train_series: Any,
    test_series: Any,
    point_params: dict,
    *,
    uq_params: Optional[dict] = None,
    variants: Optional[dict] = None,
    series_id: str = "series",
    seasonal_period: int = 12,
    alpha: float = 0.10,
    verbose: bool = False,
):
    train = ensure_series(train_series, name="train")
    test = ensure_series(test_series, name="test")
    uq = dict(uq_params or {})
    variants = copy.deepcopy(variants or FASTSKETCH_ABLATIONS)

    allowed = {
        "foundation_window",
        "prior_name",
        "prior__window",
        "prior__seasonal_period",
        "prior__ets_trend",
        "prior__ets_damped",
        "prior__theta_period",
        "prior__chronos_model_id",
        "prior__timesfm_model_id",
        "ar_lags",
        "ema_spans",
        "conv_scales",
        "use_calendar",
        "ridge_alpha",
        "calibration_split_ratio",
        "shrinkage_max",
    }
    model_params = {
        key: value for key, value in _normalize_fastsketch_tuple_params(point_params).items() if key in allowed
    }
    min_width = float(uq.get("min_width_factor", 0.0)) * seasonal_naive_scale(
        train, seasonal_period=seasonal_period
    )

    rows, forecasts, models = [], {}, {}
    for variant_name, flags in variants.items():
        params = dict(model_params)
        params.update(flags)
        params, _, _ = split_model_prior_params(params)
        start = time.perf_counter()
        try:
            model = S3FastSketchForecaster(
                horizon=1,
                seasonal_period=seasonal_period,
                aci_target=float(uq.get("aci_target", alpha)),
                aci_step_size=float(uq.get("aci_step_size", 0.05)),
                min_train_samples=6,
                min_calib_samples=2,
                **params,
            )
            result = evaluate_rolling_model(
                model,
                train,
                test,
                seasonal_period=seasonal_period,
                alpha=alpha,
            )
            forecast = result["forecast"].copy()
            center = forecast["pred"].to_numpy(dtype=float)
            half_width = 0.5 * (forecast["upper"].to_numpy(dtype=float) - forecast["lower"].to_numpy(dtype=float))
            half_width = np.maximum(float(uq.get("interval_scale", 1.0)) * half_width, min_width)
            forecast["lower"] = center - half_width
            forecast["upper"] = center + half_width
            metrics = evaluate_forecast(
                test,
                forecast["pred"],
                y_train=train,
                lower=forecast["lower"],
                upper=forecast["upper"],
                alpha=alpha,
                seasonal_period=seasonal_period,
                elapsed_seconds=time.perf_counter() - start,
                trainable_params=0 if model.foundation_only else count_trainable_parameters(model),
            )
            row = {
                "series_id": series_id,
                "variant": variant_name,
                "n_features": 0 if model.foundation_only else model.fit_report_.get("n_features", np.nan),
                "gamma": tuple(np.asarray(model.gamma_, dtype=float)),
                "final_alpha_t": float(model.current_alpha_t),
                "status": "ok",
                "error": None,
                **metrics,
            }
            forecasts[variant_name], models[variant_name] = forecast, model
            if verbose:
                print(f"[OK] {variant_name}: MAPE={metrics['mape_percent']:.3f}%, MSIS={metrics['msis']:.4f}")
        except Exception as exc:
            row = {"series_id": series_id, "variant": variant_name, "status": "failed", "error": str(exc)}
            forecasts[variant_name], models[variant_name] = None, None
            if verbose:
                print(f"[FAILED] {variant_name}: {exc}")
        rows.append(row)

    results = pd.DataFrame(rows)
    full = results[(results["variant"].isin(["full model", "Full"])) & (results["status"] == "ok")]
    if len(full) == 1:
        baseline = full.iloc[0]
        for metric in ["mape_percent", "rmse", "coverage_error", "mean_width", "msis"]:
            if metric in results:
                results[f"delta_{metric}"] = results[metric] - baseline[metric]
    return {"results": results, "forecasts": forecasts, "models": models}


def run_fastsketch_ablation_dataset(
    train_series_map: dict,
    test_series_map: dict,
    point_params: dict,
    *,
    uq_params: Optional[dict] = None,
    variants: Optional[dict] = None,
    series_ids: Optional[list] = None,
    max_series: Optional[int] = None,
    log_if_positive: bool = True,
    seasonal_period: int = 12,
    alpha: float = 0.10,
):
    ids = [sid for sid in train_series_map if sid in test_series_map]
    if series_ids is not None:
        selected = set(series_ids)
        ids = [sid for sid in ids if sid in selected]
    if max_series is not None:
        ids = ids[: int(max_series)]

    details, transforms = [], []
    for sid in ids:
        train = ensure_series(train_series_map[sid])
        test = ensure_series(test_series_map[sid])
        transformed = bool(log_if_positive and (train > 0).all() and (test > 0).all())
        if transformed:
            train = pd.Series(np.log(train.to_numpy()), index=train.index, name=train.name)
            test = pd.Series(np.log(test.to_numpy()), index=test.index, name=test.name)
        out = run_fastsketch_ablation(
            train,
            test,
            point_params,
            uq_params=uq_params,
            variants=variants,
            series_id=str(sid),
            seasonal_period=seasonal_period,
            alpha=alpha,
        )
        details.append(out["results"])
        transforms.append(
            {
                "series_id": sid,
                "log_transformed": transformed,
                "n_train": len(train),
                "n_test": len(test),
            }
        )

    detailed = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    valid = detailed[detailed.get("status") == "ok"].copy() if len(detailed) else detailed
    metric_cols = [
        c
        for c in [
            "mape_percent",
            "smape_percent",
            "mae",
            "rmse",
            "r2",
            "ecp",
            "coverage_error",
            "mean_width",
            "interval_score",
            "msis",
            "n_features",
            "trainable_params",
        ]
        if c in valid
    ]
    if len(valid):
        aggregate = pd.concat(
            [
                valid.groupby("variant")[metric_cols].mean().add_suffix("_mean"),
                valid.groupby("variant")[metric_cols].median().add_suffix("_median"),
                valid.groupby("variant").size().rename("n_series"),
            ],
            axis=1,
        ).reset_index()
    else:
        aggregate = pd.DataFrame()
    return {"detailed": detailed, "aggregate": aggregate, "transformations": pd.DataFrame(transforms)}


def paired_ablation_wilcoxon(
    detailed_results: pd.DataFrame,
    metric: str = "mape_percent",
    baseline: str = "full model",
):
    valid = detailed_results[detailed_results["status"] == "ok"]
    pivot = valid.pivot_table(index="series_id", columns="variant", values=metric, aggfunc="first")
    rows = []
    if baseline not in pivot:
        return pd.DataFrame()
    for variant in pivot.columns:
        if variant == baseline:
            continue
        pair = pivot[[baseline, variant]].dropna()
        delta = pair[variant].to_numpy() - pair[baseline].to_numpy()
        if len(delta) == 0:
            statistic, p_value = np.nan, np.nan
        elif np.allclose(delta, 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(pair[variant], pair[baseline])
        rows.append(
            {
                "metric": metric,
                "variant": variant,
                "n_pairs": len(pair),
                "baseline_mean": pair[baseline].mean(),
                "ablation_mean": pair[variant].mean(),
                "mean_delta": np.mean(delta) if len(delta) else np.nan,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    finite = result["p_value"].notna()
    ordered = result.loc[finite].sort_values("p_value").index.tolist()
    adjusted = pd.Series(np.nan, index=result.index)
    running = 0.0
    m = len(ordered)
    for rank, idx in enumerate(ordered):
        corrected = min(1.0, (m - rank) * result.loc[idx, "p_value"])
        running = max(running, corrected)
        adjusted.loc[idx] = running
    result["p_holm"] = adjusted
    result["significant_0.05"] = result["p_holm"] < 0.05
    return result.sort_values("p_holm")


def plot_s3_ablation_forecasts(train_series, test_series, forecasts, last_history=24):
    import matplotlib.pyplot as plt

    train = ensure_series(train_series)
    test = ensure_series(test_series)
    n = len(forecasts)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (mode, forecast) in zip(axes, forecasts.items()):
        if forecast is None:
            continue
        ax.plot(train.index[-last_history:], train.to_numpy()[-last_history:], label="history")
        ax.plot(test.index, test.to_numpy(), label="real", color="green")
        ax.plot(forecast.index, forecast["pred"].to_numpy(), label="forecast", color="red", linestyle="--")
        if {"lower", "upper"}.issubset(forecast.columns):
            ax.fill_between(forecast.index, forecast["lower"].to_numpy(), forecast["upper"].to_numpy(), alpha=0.18)
        ax.set_title(S3_ABLATION_LABELS.get(mode, mode))
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_componentwise_ablation_bars(
    table: pd.DataFrame,
    metric: str,
    *,
    label_column: str = "variant",
    status_column: str = "status",
    figsize=(11, 6),
    font_size: int = 16,
    tick_size: int = 14,
    label_max_chars: int = 22,
    color: str = "#4C78A8",
    show_axis_labels: bool = True,
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    valid = table.copy()
    if status_column in valid.columns:
        valid = valid[valid[status_column] == "ok"]
    valid = valid[valid[metric].notna() & np.isfinite(valid[metric])].copy()
    valid["_label"] = valid[label_column].map(lambda value: shorten_ablation_label(value, max_chars=label_max_chars))
    valid = valid.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(valid["_label"], valid[metric].astype(float), color=color)
    if show_axis_labels:
        ax.set_xlabel(metric.replace("_", " ").upper(), fontsize=font_size)
        ax.set_ylabel("Component", fontsize=font_size)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax
