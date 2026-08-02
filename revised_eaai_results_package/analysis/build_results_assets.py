from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "source_data"
FIGURE_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "generated_tables"
OUTPUT_DIR = ROOT / "analysis_outputs"

DATASET_ORDER = ["CIF", "M3", "M4", "Tourism"]
S3_MODELS = ["S3FastSketchForecaster", "S3Forecaster"]
MODEL_LABELS = {
    "S3FastSketchForecaster": "S3-FastSketch",
    "S3Forecaster": "S3-Forecaster",
    "Chronos-Bolt-Small": "Chronos-Bolt-Small",
    "TimesFM-2.5-200M": "TimesFM-2.5-200M",
    "EasyTSF-DLinear": "DLinear",
    "EasyTSF-NLinear": "NLinear",
    "SeasonalNaive": "Seasonal Naive",
}
VARIANT_LABELS = {
    "no_adapter_selector": "No adapter selector",
    "no_ar_lags": "No AR lags",
    "no_calendar": "No calendar",
    "no_causal_conv": "No causal convolution",
    "no_ema_residual": "No EMA residual",
    "no_gate": "No volatility gate",
    "no_momentum": "No momentum",
    "no_multiscale_sketch": "No multiscale sketch",
    "no_reservoir": "No reservoir",
    "no_seasonal": "No seasonal features",
    "no_shrinkage": "No shrinkage",
    "no_volatility": "No volatility features",
    "prior_only": "Prior only",
    "static_conformal": "Static conformal",
}
COLORS = {
    "S3FastSketchForecaster": "#0072B2",
    "S3Forecaster": "#D55E00",
    "other": "#8A8A8A",
}


def display_model(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "benchmark": pd.read_csv(
            SOURCE_DIR / "tabla_baselines_S3Family_fase1_actualizada_rank_MAPE.csv"
        ),
        "efficiency": pd.read_csv(SOURCE_DIR / "data_efficiency_concatenado.csv"),
        "shock": pd.read_csv(
            SOURCE_DIR / "shock_non_shock_analysis_fase1_actualizado.csv"
        ),
        "ablation": pd.read_csv(
            SOURCE_DIR / "resultados_ablacion_actualizado_Tourism.csv"
        ),
    }


def validate_and_summarize(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    benchmark = data["benchmark"]
    efficiency = data["efficiency"]
    shock = data["shock"]
    ablation = data["ablation"]

    key_specs = {
        "benchmark": ["dataset", "model"],
        "efficiency": ["dataset", "model", "history_fraction"],
        "shock": ["dataset", "model", "regime"],
        "ablation": ["dataset", "model", "variant_key"],
    }
    duplicate_keys = {
        name: int(frame.duplicated(key_specs[name]).sum())
        for name, frame in data.items()
    }
    if any(duplicate_keys.values()):
        raise ValueError(f"Duplicate analytical keys detected: {duplicate_keys}")

    rank_mismatches: list[dict[str, object]] = []
    for dataset, group in benchmark.groupby("dataset"):
        expected = (
            group["mape_percent_median"]
            .rank(method="first", ascending=True)
            .astype(int)
        )
        mask = expected.to_numpy() != group["rank"].astype(int).to_numpy()
        for row_index in group.index[mask]:
            rank_mismatches.append(
                {
                    "dataset": dataset,
                    "model": benchmark.loc[row_index, "model"],
                    "reported_rank": int(benchmark.loc[row_index, "rank"]),
                    "computed_rank": int(expected.loc[row_index]),
                }
            )

    model_union = sorted(benchmark["model"].unique())
    missing_benchmark = {
        dataset: sorted(
            set(model_union)
            - set(benchmark.loc[benchmark["dataset"] == dataset, "model"])
        )
        for dataset in DATASET_ORDER
    }
    missing_efficiency = []
    for dataset in DATASET_ORDER:
        for model in S3_MODELS:
            available = set(
                efficiency.loc[
                    (efficiency["dataset"] == dataset)
                    & (efficiency["model"] == model),
                    "history_fraction",
                ].astype(float)
            )
            for fraction in (0.25, 0.5, 0.75, 1.0):
                if fraction not in available:
                    missing_efficiency.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "history_fraction": fraction,
                        }
                    )

    shock_counts = (
        shock.groupby(["dataset", "model"])["regime"].nunique().reset_index(name="n")
    )
    incomplete_shock_pairs = shock_counts.loc[shock_counts["n"] != 2].to_dict("records")

    mean_median_ratio = (
        efficiency["mape_percent_mean"]
        / efficiency["mape_percent_median"].replace(0, np.nan)
    )
    return {
        "row_counts": {name: int(len(frame)) for name, frame in data.items()},
        "duplicate_keys": duplicate_keys,
        "rank_mismatches": rank_mismatches,
        "missing_benchmark_models": missing_benchmark,
        "missing_efficiency_cells": missing_efficiency,
        "incomplete_shock_pairs": incomplete_shock_pairs,
        "ablation_null_relative_deltas": int(
            ablation["relative_delta_mape_percent_pct_median"].isna().sum()
        ),
        "efficiency_max_mean_to_median_mape_ratio": float(mean_median_ratio.max()),
        "datasets": DATASET_ORDER,
        "benchmark_models": model_union,
    }


def plot_benchmark(benchmark: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.6))
    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        group = (
            benchmark.loc[benchmark["dataset"] == dataset]
            .sort_values("mape_percent_median", ascending=True)
            .copy()
        )
        y = np.arange(len(group))
        colors = [COLORS.get(model, COLORS["other"]) for model in group["model"]]
        axis.barh(y, group["mape_percent_median"], color=colors, alpha=0.88)
        axis.set_yticks(y, [display_model(model) for model in group["model"]])
        axis.invert_yaxis()
        axis.set_title(f"{dataset} (available models: {len(group)})")
        axis.set_xlabel("Median MAPE (%)")
        axis.grid(axis="x", alpha=0.2)
        for yi, value in zip(y, group["mape_percent_median"]):
            axis.text(
                value,
                yi,
                f" {value:.2f}",
                va="center",
                ha="left",
                fontsize=7,
            )
    fig.suptitle("Phase-1 benchmark accuracy by dataset", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "benchmark_mape_phase1.pdf")
    fig.savefig(FIGURE_DIR / "benchmark_mape_phase1.png")
    plt.close(fig)


def plot_efficiency(efficiency: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True)
    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        group = efficiency.loc[efficiency["dataset"] == dataset].copy()
        for model in S3_MODELS:
            model_data = group.loc[group["model"] == model].sort_values(
                "history_fraction"
            )
            if model_data.empty:
                continue
            axis.plot(
                model_data["history_fraction"] * 100,
                model_data["mape_percent_median"],
                marker="o",
                linewidth=2,
                color=COLORS[model],
                label=display_model(model),
            )
        axis.set_title(dataset)
        axis.set_xticks([25, 50, 75, 100])
        axis.set_xlabel("Available training history (%)")
        axis.set_ylabel("Median MAPE (%)")
        axis.grid(alpha=0.25)
        if group["history_fraction"].nunique() == 1:
            axis.text(
                50,
                float(group["mape_percent_median"].mean()),
                "Only full-history\nresults supplied",
                ha="center",
                va="center",
                fontsize=8,
                color="#555555",
            )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if not handles:
        for axis in axes.flat:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Data efficiency of the S3 family", y=1.02, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_DIR / "data_efficiency_phase1.pdf")
    fig.savefig(FIGURE_DIR / "data_efficiency_phase1.png")
    plt.close(fig)


def shock_ratios(shock: pd.DataFrame) -> pd.DataFrame:
    wide = shock.pivot(
        index=["dataset", "model"],
        columns="regime",
        values="mape_percent_median",
    ).reset_index()
    wide = wide.dropna(subset=["shock", "non_shock"]).copy()
    wide["shock_ratio"] = wide["shock"] / wide["non_shock"].replace(0, np.nan)
    wide["shock_delta_pp"] = wide["shock"] - wide["non_shock"]
    return wide


def plot_shock(shock: pd.DataFrame) -> None:
    ratios = shock_ratios(shock)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.6))
    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        group = (
            ratios.loc[ratios["dataset"] == dataset]
            .sort_values("shock_ratio", ascending=True)
            .copy()
        )
        y = np.arange(len(group))
        colors = [COLORS.get(model, COLORS["other"]) for model in group["model"]]
        axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
        axis.scatter(group["shock_ratio"], y, c=colors, s=32, zorder=3)
        axis.set_yticks(y, [display_model(model) for model in group["model"]])
        axis.set_xlim(0.75, 5.05)
        axis.set_xlabel("Shock / non-shock median MAPE ratio")
        axis.set_title(f"{dataset} (paired models: {len(group)})")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Relative accuracy degradation under shock regimes", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shock_penalty_phase1.pdf")
    fig.savefig(FIGURE_DIR / "shock_penalty_phase1.png")
    plt.close(fig)


def plot_ablation(ablation: pd.DataFrame) -> None:
    value_column = "aggregate_relative_mape_change"
    full = (
        ablation.loc[ablation["variant_key"] == "full"]
        .set_index(["dataset", "model"])["mape_percent_median"]
    )
    available = ablation.loc[ablation["variant_key"] != "full"].copy()
    available[value_column] = available.apply(
        lambda row: 100.0
        * (row["mape_percent_median"] / full.loc[(row["dataset"], row["model"])] - 1.0),
        axis=1,
    )
    model_variants = {
        model: [
            variant
            for variant in VARIANT_LABELS
            if variant in set(available.loc[available["model"] == model, "variant_key"])
        ]
        for model in S3_MODELS
    }
    all_values = available[value_column].to_numpy(dtype=float)
    bound = max(5.0, float(np.nanmax(np.abs(all_values))))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 6.2),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    image = None
    for axis, model in zip(axes, S3_MODELS):
        variants = model_variants[model]
        matrix = (
            available.loc[available["model"] == model]
            .pivot(index="variant_key", columns="dataset", values=value_column)
            .reindex(index=variants, columns=DATASET_ORDER)
        )
        image = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
        axis.set_xticks(np.arange(len(DATASET_ORDER)), DATASET_ORDER)
        axis.set_yticks(
            np.arange(len(variants)),
            [VARIANT_LABELS.get(variant, variant) for variant in variants],
        )
        axis.set_title(display_model(model))
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                if pd.notna(value):
                    axis.text(
                        column,
                        row,
                        f"{value:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.03)
        colorbar.set_label("Relative change in aggregated median MAPE (%)")
    fig.suptitle("Ablation sensitivity of the S3 family", y=0.98, fontsize=12)
    fig.subplots_adjust(left=0.18, right=0.92, top=0.90, bottom=0.08, wspace=0.45)
    fig.savefig(FIGURE_DIR / "ablation_sensitivity_phase1.pdf")
    fig.savefig(FIGURE_DIR / "ablation_sensitivity_phase1.png")
    plt.close(fig)


def write_benchmark_table(benchmark: pd.DataFrame) -> None:
    rows = []
    for dataset in DATASET_ORDER:
        group = benchmark.loc[benchmark["dataset"] == dataset].copy()
        best = group.sort_values("mape_percent_median").iloc[0]
        non_s3 = group.loc[~group["model"].isin(S3_MODELS)].sort_values(
            "mape_percent_median"
        )
        best_non_s3 = non_s3.iloc[0] if not non_s3.empty else None
        row = [latex_escape(dataset)]
        for model in S3_MODELS:
            match = group.loc[group["model"] == model]
            if match.empty:
                row.extend(["--", "--"])
            else:
                record = match.iloc[0]
                row.extend(
                    [
                        str(int(record["rank"])),
                        fmt(record["mape_percent_median"]),
                    ]
                )
        row.extend(
            [
                latex_escape(display_model(best["model"])),
                fmt(best["mape_percent_median"]),
                (
                    latex_escape(display_model(best_non_s3["model"]))
                    if best_non_s3 is not None
                    else "--"
                ),
                fmt(
                    best_non_s3["mape_percent_median"]
                    if best_non_s3 is not None
                    else None
                ),
            ]
        )
        rows.append(" & ".join(row) + r" \\")

    text = r"""\begin{table*}[t]
\centering
\caption{Phase-1 benchmark summary based on median MAPE. Only completed model--dataset evaluations are ranked.}
\label{tab:phase1_benchmark}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrlrlr}
\toprule
& \multicolumn{2}{c}{S3-FastSketch} & \multicolumn{2}{c}{S3-Forecaster}
& \multicolumn{2}{c}{Overall best} & \multicolumn{2}{c}{Best non-S3} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Dataset & Rank & MAPE (\%) & Rank & MAPE (\%) & Model & MAPE (\%) & Model & MAPE (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\footnotesize\emph{Note:} Ranks are conditional on the available phase-1 results and must not be interpreted as final rankings until the missing model--dataset runs are completed.
\end{minipage}
\end{table*}
"""
    (TABLE_DIR / "benchmark_summary_phase1.tex").write_text(text, encoding="utf-8")


def write_efficiency_table(efficiency: pd.DataFrame) -> None:
    rows = []
    for dataset in DATASET_ORDER:
        for model in S3_MODELS:
            group = efficiency.loc[
                (efficiency["dataset"] == dataset) & (efficiency["model"] == model)
            ].set_index("history_fraction")
            values = []
            for fraction in (0.25, 0.5, 0.75, 1.0):
                if fraction not in group.index:
                    values.append("--")
                else:
                    values.append(
                        f"{fmt(group.loc[fraction, 'mape_percent_median'])}"
                        f" ({int(group.loc[fraction, 'n_series'])})"
                    )
            rows.append(
                " & ".join(
                    [latex_escape(dataset), latex_escape(display_model(model)), *values]
                )
                + r" \\"
            )
    text = r"""\begin{table}[t]
\centering
\caption{Median MAPE (\%) as a function of the available training history.}
\label{tab:data_efficiency_phase1}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrr}
\toprule
Dataset & Model & 25\% ($n$) & 50\% ($n$) & 75\% ($n$) & 100\% ($n$) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{minipage}{0.96\linewidth}
\footnotesize\emph{Note:} Each cell reports median MAPE followed by the number of successfully evaluated series in parentheses. Cohort size changes across fractions and, in some cases, across models; the trajectories are therefore descriptive rather than fixed-cohort learning curves. ``--'' denotes an unavailable run.
\end{minipage}
\end{table}
"""
    (TABLE_DIR / "data_efficiency_phase1.tex").write_text(text, encoding="utf-8")


def write_shock_table(shock: pd.DataFrame) -> None:
    ratios = shock_ratios(shock)
    rows = []
    for dataset in DATASET_ORDER:
        for model in S3_MODELS:
            match = ratios.loc[
                (ratios["dataset"] == dataset) & (ratios["model"] == model)
            ]
            if match.empty:
                values = ["--", "--", "--"]
            else:
                record = match.iloc[0]
                counts = shock.loc[
                    (shock["dataset"] == dataset) & (shock["model"] == model)
                ].set_index("regime")["n_series"]
                values = [
                    f"{fmt(record['non_shock'])} ({int(counts.loc['non_shock'])})",
                    f"{fmt(record['shock'])} ({int(counts.loc['shock'])})",
                    fmt(record["shock_ratio"]),
                ]
            rows.append(
                " & ".join(
                    [latex_escape(dataset), latex_escape(display_model(model)), *values]
                )
                + r" \\"
            )
    text = r"""\begin{table}[t]
\centering
\caption{Shock versus non-shock median MAPE for the S3 family.}
\label{tab:shock_phase1}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrr}
\toprule
Dataset & Model & Non-shock (\%, $n$) & Shock (\%, $n$) & Ratio \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{minipage}{0.96\linewidth}
\footnotesize\emph{Note:} The ratio is shock MAPE divided by non-shock MAPE. Values above one indicate degradation during shock observations.
\end{minipage}
\end{table}
"""
    (TABLE_DIR / "shock_summary_phase1.tex").write_text(text, encoding="utf-8")


def write_ablation_table(ablation: pd.DataFrame) -> None:
    value_column = "aggregate_relative_mape_change"
    full = (
        ablation.loc[ablation["variant_key"] == "full"]
        .set_index(["dataset", "model"])["mape_percent_median"]
    )
    candidates = ablation.loc[ablation["variant_key"] != "full"].copy()
    candidates[value_column] = candidates.apply(
        lambda row: 100.0
        * (row["mape_percent_median"] / full.loc[(row["dataset"], row["model"])] - 1.0),
        axis=1,
    )
    rows = []
    for dataset in DATASET_ORDER:
        for model in S3_MODELS:
            group = candidates.loc[
                (candidates["dataset"] == dataset) & (candidates["model"] == model)
            ].sort_values(value_column)
            if group.empty:
                row = [dataset, display_model(model), "--", "--", "--", "--"]
            else:
                best = group.iloc[0]
                worst = group.iloc[-1]
                row = [
                    dataset,
                    display_model(model),
                    VARIANT_LABELS.get(best["variant_key"], best["variant"]),
                    fmt(best[value_column], 1),
                    VARIANT_LABELS.get(worst["variant_key"], worst["variant"]),
                    fmt(worst[value_column], 1),
                ]
            rows.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    text = r"""\begin{table*}[t]
\centering
\caption{Largest favorable and adverse paired median MAPE changes in the phase-1 ablation export.}
\label{tab:ablation_extremes_phase1}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllrlr}
\toprule
Dataset & Model & Most favorable removal & Change (\%) & Most adverse removal & Change (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\footnotesize\emph{Note:} Changes compare each variant's aggregated median MAPE with the corresponding full-model median. Negative values favor the ablated variant; positive values favor the full model. This descriptive summary does not establish statistical significance.
\end{minipage}
\end{table*}
"""
    (TABLE_DIR / "ablation_extremes_phase1.tex").write_text(text, encoding="utf-8")


def write_narrative_values(data: dict[str, pd.DataFrame]) -> None:
    benchmark = data["benchmark"]
    efficiency = data["efficiency"]
    ratios = shock_ratios(data["shock"])
    ablation = data["ablation"]
    payload: dict[str, object] = {}

    payload["benchmark"] = {}
    for dataset in DATASET_ORDER:
        group = benchmark.loc[benchmark["dataset"] == dataset].sort_values(
            "mape_percent_median"
        )
        payload["benchmark"][dataset] = [
            {
                "rank": int(row["rank"]),
                "model": display_model(row["model"]),
                "mape_percent_median": float(row["mape_percent_median"]),
                "mape_percent_iqr": float(row["mape_percent_iqr"]),
                "mase_median": float(row["mase_median"]),
                "ecp_mean": float(row["ecp_mean"]),
                "elapsed_seconds_median": float(row["elapsed_seconds_median"]),
            }
            for _, row in group.iterrows()
        ]

    payload["data_efficiency"] = {}
    for (dataset, model), group in efficiency.groupby(["dataset", "model"]):
        payload["data_efficiency"][f"{dataset}|{display_model(model)}"] = {
            str(float(row["history_fraction"])): float(row["mape_percent_median"])
            for _, row in group.sort_values("history_fraction").iterrows()
        }

    payload["shock"] = {
        f"{row['dataset']}|{display_model(row['model'])}": {
            "non_shock": float(row["non_shock"]),
            "shock": float(row["shock"]),
            "ratio": float(row["shock_ratio"]),
            "delta_pp": float(row["shock_delta_pp"]),
        }
        for _, row in ratios.iterrows()
    }

    value_column = "aggregate_relative_mape_change"
    full = (
        ablation.loc[ablation["variant_key"] == "full"]
        .set_index(["dataset", "model"])["mape_percent_median"]
    )
    ablation = ablation.copy()
    ablation[value_column] = ablation.apply(
        lambda row: 100.0
        * (row["mape_percent_median"] / full.loc[(row["dataset"], row["model"])] - 1.0),
        axis=1,
    )
    payload["ablation"] = {}
    for (dataset, model), group in ablation.loc[
        ablation["variant_key"] != "full"
    ].groupby(["dataset", "model"]):
        ordered = group.sort_values(value_column)
        payload["ablation"][f"{dataset}|{display_model(model)}"] = {
            "most_favorable": {
                "variant": VARIANT_LABELS.get(
                    ordered.iloc[0]["variant_key"], ordered.iloc[0]["variant"]
                ),
                "relative_change_percent": float(ordered.iloc[0][value_column]),
            },
            "most_adverse": {
                "variant": VARIANT_LABELS.get(
                    ordered.iloc[-1]["variant_key"], ordered.iloc[-1]["variant"]
                ),
                "relative_change_percent": float(ordered.iloc[-1][value_column]),
            },
        }
    (OUTPUT_DIR / "narrative_values.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    data = load_data()
    quality = validate_and_summarize(data)
    (OUTPUT_DIR / "data_quality_summary.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    plot_benchmark(data["benchmark"])
    plot_efficiency(data["efficiency"])
    plot_shock(data["shock"])
    plot_ablation(data["ablation"])
    write_benchmark_table(data["benchmark"])
    write_efficiency_table(data["efficiency"])
    write_shock_table(data["shock"])
    write_ablation_table(data["ablation"])
    write_narrative_values(data)
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
