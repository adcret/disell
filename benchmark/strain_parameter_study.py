#!/usr/bin/env python3
"""How the best flood-fill parameters move as the microstructure evolves.

The question
------------
As tensile strain rises, cells get smaller and neighbouring cells become more
misoriented (Zelenika et al., Sci Rep 15, 8655 (2025); see
``strain_phantoms.py``).  Both changes push the segmentation in opposite
directions: finer cells demand a smaller neighbourhood and a smaller minimum
size, while stronger misorientation makes interfaces easier to detect and so
permits a *larger* local threshold.  This script measures which wins, by
searching each strain phantom independently under the same 1.5 um radius cap
and then comparing the optima.

Each strain gets its own full search, so the answer is "the best parameters for
this microstructure", not "the primary phantom's parameters transferred".

Usage::

    python strain_parameter_study.py run     --workers 14
    python strain_parameter_study.py report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PHANTOM_DIR = HERE / "phantoms"
RESULT_ROOT = HERE / "runs/strain"

#: The parameters whose movement with strain is the study's output.
TRACKED = ("footprint_radius_um", "footprint_voxels", "neighbour_requirement",
           "footprint_tolerance", "local_threshold_deg", "global_threshold_deg",
           "min_cell_size", "kam_radius_um")

STAGES = ("markers", "kam", "merge", "baseline", "finalise")


def result_dir(strain_key: str) -> Path:
    return RESULT_ROOT / strain_key


def phantom_path(strain_key: str, realization: int = 0) -> Path:
    return PHANTOM_DIR / f"strain_{strain_key}_r{realization}.npz"


def run_stage(strain_key: str, stage: str, workers: int, reduced: bool = True) -> None:
    command = [
        sys.executable, str(HERE / "capped_search.py"), stage,
        "--workers", str(workers),
        "--phantom", str(phantom_path(strain_key)),
        "--out", str(result_dir(strain_key)),
    ]
    if reduced and stage == "markers":
        command.append("--reduced")
    print(f"\n=== strain {strain_key}: {stage} ===", flush=True)
    subprocess.run(command, check=True)


def best_per_arm(strain_key: str, policy: str | None = None) -> dict[str, dict]:
    """The winning configuration for each arm at one strain, under one policy.

    Defaults to the fusion-averse policy: with a merge step downstream,
    fragmentation is recoverable but fusion is not, so hitting the cell count
    exactly is no longer the objective.  See ``capped_search.POLICIES``.
    """

    import capped_search as cs

    key = cs.POLICIES[policy or cs.STRICT_RECOVERY]
    rows = [r for r in cs.read_rows(result_dir(strain_key) / "final.jsonl")
            if r.get("status") == "ok"]
    winners: dict[str, dict] = {}
    for row in rows:
        arm = row.get("arm") or "flood fill"
        if arm not in winners or key(row) < key(winners[arm]):
            winners[arm] = row
    return winners


def collect(policy: str | None = None):
    """One row per strain per arm, with the microstructure it was tuned on."""

    import capped_search as cs
    import pandas as pd
    import strain_phantoms

    policy = policy or cs.STRICT_RECOVERY
    records = []
    for strain_key, (strain, mean_d, sd_d, chi_k, chi_sigma) in \
            strain_phantoms.STRAIN_TREND.items():
        meta_path = PHANTOM_DIR / f"strain_{strain_key}_r0.json"
        realised = {}
        if meta_path.exists():
            realised = json.loads(meta_path.read_text()).get("realised", {})
        for arm, row in best_per_arm(strain_key, policy).items():
            record = {
                "policy": policy,
                "strain_percent": strain,
                "strain_key": strain_key,
                "arm": arm,
                "extrapolated": strain_key in strain_phantoms.EXTRAPOLATED,
                "target_cell_diameter_um": mean_d,
                "chi_sigma_deg": chi_sigma,
                "realised_cells": realised.get("n_cells"),
                "realised_misorientation_deg":
                    realised.get("neighbour_misorientation_mean_deg"),
                "n_cells_pred": row.get("n_cells_pred"),
                "cell_count_error": row.get("cell_count_error"),
                "identity_f1": row.get("identity_f1"),
                "identity_recall": row.get("identity_recall"),
                "vi_merge_bits": row.get("vi_merge_bits"),
                "vi_split_bits": row.get("vi_split_bits"),
                "ari": row.get("ari"),
                "boundary_assd_um": row.get("boundary_assd_um"),
            }
            for name in TRACKED:
                record[name] = row.get(name)
            # Normalised forms.  If a parameter must simply track the cell size
            # then its ratio to the cell diameter is constant across strain, and
            # the practical rule is "scale with cell size" rather than a table
            # of numbers.  A ratio that itself drifts means something beyond
            # geometry is changing -- most plausibly the rising misorientation.
            diameter = mean_d
            voxel_um3 = 0.16
            cell_voxels = (np.pi * diameter ** 3 / 6.0) / voxel_um3
            if row.get("footprint_radius_um"):
                record["footprint_radius_per_diameter"] = (
                    row["footprint_radius_um"] / diameter)
            if row.get("kam_radius_um"):
                record["kam_radius_per_diameter"] = row["kam_radius_um"] / diameter
            if row.get("min_cell_size"):
                record["min_cell_size_per_cell_volume"] = (
                    row["min_cell_size"] / cell_voxels)
            if row.get("local_threshold_deg"):
                record["local_threshold_per_chi_sigma"] = (
                    row["local_threshold_deg"] / chi_sigma)
            for name in ("percentile", "merge_size_voxels", "merge_threshold_deg"):
                if name in row:
                    record[name] = row[name]
            records.append(record)
    return pd.DataFrame(records).sort_values(["arm", "strain_percent"])


def plot(frame, out: Path):
    """Optimal parameters against strain, one panel per parameter."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flood = frame[frame["arm"].str.startswith("flood fill")]
    if flood.empty:
        print("no flood-fill winners to plot")
        return None

    panels = [
        ("footprint_radius_um", "footprint radius (um)"),
        ("local_threshold_deg", "local threshold (deg)"),
        ("min_cell_size", "min cell size (voxels)"),
        ("kam_radius_um", "KAM radius (um)"),
        ("recovered_at_90", "cells recovered at 90 %"),
        ("contamination", "contamination at the optimum"),
        # Normalised: a flat line here is a transferable rule.
        ("footprint_radius_per_diameter", "footprint radius / cell diameter"),
        ("min_cell_size_per_cell_volume", "min cell size / cell volume"),
        ("local_threshold_per_chi_sigma", "local threshold / chi sigma"),
    ]
    panels = [(c, t) for c, t in panels if c in flood]
    rows_n = int(np.ceil(len(panels) / 3))
    fig, axes = plt.subplots(rows_n, 3, figsize=(14, 3.7 * rows_n), squeeze=False)
    axes = axes
    for ax, (column, title) in zip(axes.ravel(), panels):
        for arm, group in flood.groupby("arm"):
            group = group.sort_values("strain_percent")
            ax.plot(group["strain_percent"], group[column], "o-", label=arm)
        # Mark the extrapolated point so it is never read as measured.
        extra = flood[flood["extrapolated"]]
        if not extra.empty:
            ax.scatter(extra["strain_percent"], extra[column], s=140,
                       facecolors="none", edgecolors="0.4", linewidths=1.4, zorder=5)
        ax.set_xlabel("strain (%)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Best flood-fill parameters against strain, under the 1.5 um radius cap\n"
        "(circled point is extrapolated beyond the paper's measured 0.6-4.6 % range)",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "report"])
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--strains", nargs="*", default=None)
    parser.add_argument("--stages", nargs="*", default=list(STAGES))
    args = parser.parse_args()

    import strain_phantoms

    keys = args.strains or list(strain_phantoms.STRAIN_TREND)

    if args.action == "run":
        missing = [k for k in keys if not phantom_path(k).exists()]
        if missing:
            raise SystemExit(
                f"missing phantoms for {missing}; run `python strain_phantoms.py build`"
            )
        for strain_key in keys:
            for stage in args.stages:
                run_stage(strain_key, stage, args.workers)
        print("\nall strain searches complete")

    import capped_search as cs
    import pandas as pd

    frames = []
    for policy in (cs.STRICT_RECOVERY, cs.COUNT_FIRST, cs.FUSION_AVERSE):
        part = collect(policy)
        if not part.empty:
            frames.append(part)
    if not frames:
        print("no finalised results yet")
        return
    everything = pd.concat(frames, ignore_index=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    everything.to_csv(RESULT_ROOT / "strain_optima.csv", index=False)
    frame = frames[0]          # the strict-recovery policy is the headline
    plot(frame, RESULT_ROOT / "strain_optima.png")

    columns = ["strain_percent", "arm", "target_cell_diameter_um", "chi_sigma_deg",
               "footprint_radius_um", "local_threshold_deg", "min_cell_size",
               "kam_radius_um", "n_cells_pred", "cell_count_error",
               "identity_recall", "vi_merge_bits", "identity_f1", "ari"]
    print(f"\nBest parameters per strain under the 1.5 um cap "
          f"({cs.STRICT_RECOVERY}):\n")
    print(frame[[c for c in columns if c in frame]].to_string(index=False))
    print(f"\nwritten: {RESULT_ROOT/'strain_optima.csv'} and strain_optima.png")


if __name__ == "__main__":
    main()
