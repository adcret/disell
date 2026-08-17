#!/usr/bin/env python3
"""Minimal manuscript figures for the synthetic count-first benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

HERE = Path(__file__).resolve().parent
COUNT = HERE / "analysis" / "count_first_v1"
COMPARISON = HERE / "analysis" / "synthetic_method_comparison_v1"
ORDERING = HERE / "analysis" / "paired_ordering_v1"
OUT = HERE / "analysis" / "paper_minimal_v1"
PARAMETERS = (
    "local_threshold_deg",
    "global_threshold_deg",
    "footprint_tolerance",
    "footprint_radius_um",
    "min_cell_size",
    "kam_radius_um",
)
LABELS = {
    "local_threshold_deg": "local threshold (°)",
    "global_threshold_deg": "global threshold (°)",
    "footprint_tolerance": "neighbour fraction",
    "footprint_radius_um": "neighbour radius (µm)",
    "min_cell_size": "minimum cell size (voxels)",
    "kam_radius_um": "KAM radius (µm)",
}
LOG_X = {
    "local_threshold_deg",
    "global_threshold_deg",
    "footprint_radius_um",
    "kam_radius_um",
}


def save(fig, name: str) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(
            OUT / f"{name}.{extension}",
            dpi=600 if extension == "png" else None,
        )
    plt.close(fig)


def parameter_search_difficulty() -> None:
    report = json.loads((COUNT / "count_first_report.json").read_text())
    difficulty = report["parameter_difficulty"]
    labels = ["exact", "within 1%", "within 5%"]
    keys = ["exact_count", "within_1_percent", "within_5_percent"]
    values = [100 * difficulty[key]["fraction"] for key in keys]
    source = pd.DataFrame({"criterion": labels, "successful_parameter_sets_percent": values})
    source.to_csv(OUT / "parameter_search_difficulty_source.csv", index=False)

    fig, axis = plt.subplots(figsize=(8.8 / 2.54, 5.2 / 2.54))
    y = np.arange(len(labels))
    axis.barh(y, values, color="0.2", height=0.55)
    axis.set_yticks(y, labels)
    axis.set_xlabel("successful parameter sets (%)")
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlim(0.01, 5)
    for row, value in enumerate(values):
        axis.text(value * 1.12, row, f"{value:.3g}%", va="center")
    fig.subplots_adjust(left=0.25, right=0.90, top=0.94, bottom=0.22)
    save(fig, "parameter_search_difficulty")


def local_count_sensitivity() -> None:
    frame = pd.read_csv(COUNT / "local_oat" / "profiles.csv")
    fig, axes = plt.subplots(2, 3, figsize=(18.0 / 2.54, 9.8 / 2.54))
    for axis, name in zip(axes.ravel(), PARAMETERS):
        group = frame[frame.parameter == name].sort_values("value")
        axis.plot(group.value, group.n_cells_pred, "-o", color="k", lw=0.9, ms=2.5)
        axis.axhline(360, color="0.55", ls="--", lw=0.8)
        axis.set_xlabel(LABELS[name])
        if name in LOG_X:
            axis.set_xscale("log")
        values = group.value.to_numpy()
        tick_values = (
            [values[0], values[-1]]
            if name in LOG_X
            else [values[0], values[len(values) // 2], values[-1]]
        )
        axis.xaxis.set_major_locator(FixedLocator(tick_values))
        axis.xaxis.set_major_formatter(FixedFormatter([f"{value:.2g}" for value in tick_values]))
        axis.xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].set_ylabel("recovered cells")
    axes[1, 0].set_ylabel("recovered cells")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.13, wspace=0.30, hspace=0.42)
    frame.to_csv(OUT / "local_count_sensitivity_source.csv", index=False)
    save(fig, "local_count_sensitivity")


def paired_ordering_effect() -> None:
    summary = json.loads((ORDERING / "paired_ordering_summary.json").read_text())
    measures = [
        ("count error", "absolute_cell_count_error"),
        ("correct-cell F1", "identity_f1"),
    ]
    records = []
    for label, key in measures:
        result = summary["configuration_medians"][key]
        records.append({
            "measure": label,
            "ordered_better": result["ordered_better"],
            "equal": result["equal"],
            "ordered_worse": result["ordered_worse"],
        })
    source = pd.DataFrame(records)
    source.to_csv(OUT / "method_count_robustness_source.csv", index=False)

    fig, axis = plt.subplots(figsize=(8.8 / 2.54, 5.8 / 2.54))
    y = np.arange(len(source))
    left = np.zeros(len(source))
    for column, label, color in (
        ("ordered_better", "better", "0.15"),
        ("equal", "same", "0.60"),
        ("ordered_worse", "worse", "0.90"),
    ):
        values = source[column].to_numpy()
        axis.barh(
            y, values, left=left, height=0.52, color=color,
            edgecolor="0.15", linewidth=0.5, label=label,
        )
        for row, (start, value) in enumerate(zip(left, values)):
            if value:
                axis.text(start + value / 2, row, str(value), ha="center", va="center")
        left += values
    axis.set_yticks(y, source["measure"])
    axis.set_xlabel("parameter settings")
    axis.set_xlim(0, int(left.max()))
    axis.invert_yaxis()
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.80, bottom=0.23)
    save(fig, "method_count_robustness")


def method_identity_recovery() -> None:
    comparison = json.loads((COMPARISON / "comparison.json").read_text())
    labels = ["size ordered", "not ordered", "KAM"]
    keys = ["size ordered", "not size ordered", "KAM"]
    values = [
        comparison["distributions"][key]["identity_f1"]["median"]
        for key in keys
    ]
    pd.DataFrame({"method": labels, "identity_f1": values}).to_csv(
        OUT / "method_identity_recovery_source.csv", index=False
    )
    fig, axis = plt.subplots(figsize=(8.8 / 2.54, 5.8 / 2.54))
    axis.bar(range(3), values, color="0.2", width=0.58)
    axis.set_xticks(range(3), labels)
    axis.set_ylabel("correct-cell F1")
    axis.set_ylim(0, 1)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.96, bottom=0.22)
    save(fig, "method_identity_recovery")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parameter_search_difficulty()
    local_count_sensitivity()
    paired_ordering_effect()
    method_identity_recovery()
    print(f"minimal synthetic figures written to {OUT}")


if __name__ == "__main__":
    main()
