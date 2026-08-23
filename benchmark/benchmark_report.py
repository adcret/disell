#!/usr/bin/env python3
"""Assemble the flood-fill versus KAM benchmark into one report.

Pulls together the four pieces of evidence produced by the other scripts:

* ``runs/primary/``          both arms searched to their own optimum on the
                               primary phantom, under the 1.5 um radius cap;
* ``runs/strain/``   the same, per strain, giving the parameter trend;
* ``analysis/``  the mechanism -- faint walls and the 2D/3D split;
* ``analysis/``  whether the parameters can be chosen without
                               ground truth.

Writes ``analysis/`` with a markdown summary, the figures,
and the machine-readable numbers behind them.

Everything here is measured on synthetic phantoms whose labels are known.  That
is the point: the experimental volumes have no ground truth, so the argument for
using this segmentation on them has to be made where truth exists and then
carried over by the mechanism, not by the score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "analysis"
PRIMARY = HERE / "runs/primary"
STRAIN_ROOT = HERE / "runs/strain"
#: The refined sweeps supersede the coarse search for sections 1, 4 and 5.
#: They resolve each arm on its own terms: KAM over 4-60 in half-point
#: percentile steps and 0.4-2.475 um kernels (the 0.9 um floor was flood-fill
#: reasoning and never applied to KAM), the flood fill about its own optimum,
#: and the flood fill *after its merge step*, which is the arm as deployed.
REFINED = HERE / "runs/refined"

#: Reported in this order throughout.
ARMS = (("flood", "flood fill"), ("merged", "flood fill + merge"),
        ("kam", "KAM threshold"))

#: True cell counts, per phantom.
TRUTH = {"primary": 2534, "2p4": 1481, "3p5": 1622, "4p6": 1924, "6p2": 2525}

#: Strain series only; the primary phantom is reported on its own.
STRAIN_KEYS = ("2p4", "3p5", "4p6", "6p2")

#: For section 5, alongside each strain's optimum.
STRAIN_CONTEXT = {
    "2p4": {"strain_percent": 2.4, "cell_diameter_um": 5.02, "chi_sigma_deg": 0.14},
    "3p5": {"strain_percent": 3.5, "cell_diameter_um": 4.87, "chi_sigma_deg": 0.22},
    "4p6": {"strain_percent": 4.6, "cell_diameter_um": 4.59, "chi_sigma_deg": 0.28},
    "6p2": {"strain_percent": 6.2, "cell_diameter_um": 4.20, "chi_sigma_deg": 0.36},
}
WHY = HERE / "analysis" / "why_flood_fill.json"
GENERAL = HERE / "analysis" / "generalisation.json"
ECONOMY = HERE / "analysis" / "parameter_economy.json"
REPLICATES = HERE / "analysis" / "replicates.json"
SEEDS = HERE / "analysis" / "seed_ablation.json"
DIMREP = HERE / "analysis" / "dimension_replicates.json"


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def refined_rows(stem: str, label: str) -> list:
    """Scored, admissible rows for one arm on one phantom."""

    import capped_search as cs

    return [r for r in cs.read_rows(REFINED / f"{stem}_{label}.jsonl")
            if r.get("status") == "ok" and not cs._disqualified(r)]


def refined_best(stem: str, label: str):
    import capped_search as cs

    rows = refined_rows(stem, label)
    return min(rows, key=cs.strict_recovery_key) if rows else None


def refined_winners() -> dict:
    """Each arm at its own optimum on the primary phantom, refined grid."""

    winners = {}
    for stem, name in ARMS:
        best = refined_best(stem, "primary")
        if best is not None:
            winners[name] = best
    return winners


def refined_breadth() -> dict:
    """How much of each arm's grid is near that arm's own best.

    Pooled over the four strain phantoms, each measured against its own best,
    so a phantom on which every arm does badly cannot flatter anybody.
    """

    out = {}
    for stem, name in ARMS:
        total = 0
        within = {95: 0, 90: 0, 80: 0}
        best_sum = 0
        best_rates = []
        for label in STRAIN_KEYS:
            rows = refined_rows(stem, label)
            if not rows:
                continue
            best = max(r["recovered_at_90"] for r in rows)
            best_sum += best
            best_rates.append(best / TRUTH[label])
            total += len(rows)
            for pct in within:
                within[pct] += sum(1 for r in rows
                                   if r["recovered_at_90"] >= pct / 100 * best)
        if total:
            out[name] = {"n_configurations": total,
                         "best_recovered_at_90": best_sum,
                         "mean_best_rate": float(np.mean(best_rates)),
                         **{f"fraction_within_{p}pct_of_best": within[p] / total
                            for p in within}}
    return out


#: The one axis each arm is actually tuned on.
TUNED_AXIS = {"flood": "local_threshold_deg", "merged": "local_threshold_deg",
              "kam": "percentile"}


def refined_tuning_profile() -> dict:
    """What one global value of each arm's tuned axis costs across strain.

    Section 6 asks what freezing an axis costs when every *other* axis is
    re-tuned per volume.  This asks the blunter question the deployment story
    needs: fix the one axis anybody would sweep at a single value for all
    strains, and see what is lost against tuning it per volume.
    """

    out = {}
    for stem, name in ARMS:
        axis = TUNED_AXIS[stem]
        curves = {}
        for label in STRAIN_KEYS:
            best = {}
            for row in refined_rows(stem, label):
                value = row.get(axis)
                if value is None:
                    continue
                best[value] = max(best.get(value, 0), row["recovered_at_90"])
            if best:
                curves[label] = {v: c / TRUTH[label] for v, c in best.items()}
        if len(curves) != len(STRAIN_KEYS):
            continue
        shared = set.intersection(*(set(c) for c in curves.values()))
        if not shared:
            continue
        own = [max(curves[k].values()) for k in STRAIN_KEYS]
        global_value = max(shared,
                           key=lambda v: np.mean([curves[k][v] for k in STRAIN_KEYS]))
        fixed = [curves[k][global_value] for k in STRAIN_KEYS]
        out[name] = {
            "axis": axis, "curves": curves,
            "per_phantom_optimum": {k: max(curves[k], key=curves[k].get)
                                    for k in STRAIN_KEYS},
            "best_single_value": global_value,
            "mean_tuned": float(np.mean(own)),
            "mean_fixed": float(np.mean(fixed)),
            "mean_cost": float(np.mean(own) - np.mean(fixed)),
            "worst_cost": float(max(o - f for o, f in zip(own, fixed))),
            "relative_cost": float((np.mean(own) - np.mean(fixed)) / np.mean(own)),
        }
    return out


def refined_strain_table():
    """Each arm's optimum on each strain phantom, from the refined grids."""

    import pandas as pd

    rows = []
    for label in STRAIN_KEYS:
        for stem, name in ARMS:
            best = refined_best(stem, label)
            if best is None:
                continue
            rows.append({**STRAIN_CONTEXT[label], "arm": name,
                         "footprint_radius_um": best.get("footprint_radius_um"),
                         "local_threshold_deg": best.get("local_threshold_deg"),
                         "percentile": best.get("percentile"),
                         "min_cell_size": best.get("min_cell_size"),
                         "kam_radius_um": best.get("kam_radius_um"),
                         "n_cells_pred": best.get("n_cells_pred"),
                         "recovered_at_90": best.get("recovered_at_90"),
                         "rate": best["recovered_at_90"] / TRUTH[label],
                         "contamination": best.get("contamination"),
                         "ari": best.get("ari")})
    frame = pd.DataFrame(rows)
    return None if frame.empty else frame


def primary_winners() -> dict:
    import capped_search as cs

    rows = [r for r in cs.read_rows(PRIMARY / "final.jsonl")
            if r.get("status") == "ok"]
    winners: dict[str, dict] = {}
    for row in rows:
        arm = row.get("arm") or "flood fill"
        if arm not in winners or cs.strict_recovery_key(row) < cs.strict_recovery_key(winners[arm]):
            winners[arm] = row
    return winners


def figure_arms(winners: dict, path: Path):
    """Recovery and contamination, per arm, at each arm's own optimum."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not winners:
        return
    names = list(winners)
    recovered = [winners[n].get("recovered_at_90") or 0 for n in names]
    contamination = [winners[n].get("contamination") or 0 for n in names]
    fused = [winners[n].get("fused_true_cells") or 0 for n in names]
    n_true = winners[names[0]].get("n_cells_true") or 1

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, values, title in (
        (axes[0], recovered, f"true cells recovered at 90 % (of {n_true})"),
        (axes[1], contamination, "contamination (voxels in the wrong cell)"),
        (axes[2], fused, "fused true cells"),
    ):
        bars = ax.bar(range(len(names)), values,
                      color=["#4C78A8", "#72B7B2", "#E45756"][:len(names)])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Each arm at its own optimum, on its own refined grid", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def figure_dimension(why: dict, path: Path):
    """The 2D+link versus 3D comparison, which is the 3D argument."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dimension = (why or {}).get("dimension")
    if not dimension:
        return
    arms = [k for k in dimension if isinstance(dimension[k], dict)
            and "3D" in dimension[k]]
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    x = np.arange(len(arms))
    two = [dimension[a]["2D+link"]["recovered_at_90"] for a in arms]
    three = [dimension[a]["3D"]["recovered_at_90"] for a in arms]
    ax.bar(x - width / 2, two, width, label="2D slice-wise + z linking",
           color="#B0B0B0")
    ax.bar(x + width / 2, three, width, label="3D volumetric", color="#4C78A8")
    for i, (a, b) in enumerate(zip(two, three)):
        ax.text(i - width / 2, a, str(a), ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, b, str(b), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("true cells recovered at 90 %")
    ax.set_title("What each method gains from the third dimension", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def strain_table():
    import strain_parameter_study as sps

    try:
        frame = sps.collect()
    except Exception:                                # noqa: BLE001
        return None
    return None if frame.empty else frame


def markdown(winners, why, general, strain, economy, replicates, seeds, dimrep,
             tuning=None) -> str:
    _bre = (general or {}).get("breadth", {}).get("arms") or {}

    def _pct(arm: str, within: int) -> str:
        """Section 4's own number, so the prose cannot drift from the table."""

        entry = _bre.get(arm) or {}
        return f"{entry.get(f'fraction_within_{within}pct_of_best', 0):.2%}"

    def _ceiling(arm: str) -> str:
        return f"{(_bre.get(arm) or {}).get('mean_best_rate', 0):.2f}"

    lines = ["# Flood fill versus KAM thresholding on 3D DFXM volumes", ""]
    lines += [
        "All numbers are measured against synthetic phantoms with known labels.",
        "The experimental volumes have no ground truth, so the case for using",
        "this segmentation on them rests on the mechanism established here, not",
        "on a score transferred from them.", "",
        "Recovery is counted strictly: a true cell counts only when a predicted",
        "cell is at least 90 % pure **and** at least 90 % complete for it.",
        "Contamination is the fraction of labelled voxels sitting in a cell",
        "other than their own; splitting a cell costs nothing, fusing two costs",
        "in proportion to how wrong it is.", "",
    ]

    primary_rep = (replicates or {}).get("primary") or {}

    if winners:
        n_true = winners[list(winners)[0]].get("n_cells_true")
        lines += ["## 1. Each arm at its own optimum", "",
                  f"Primary phantom, {n_true} cells. Each arm is searched on "
                  "its own refined grid: the flood fill over footprints of "
                  "1.0-1.45 um, KAM over kernels of 0.4-2.475 um and "
                  "percentiles from 4 to 60 in half-point steps. The earlier "
                  "0.9 um radius floor was flood-fill reasoning -- a smaller "
                  "kernel has no out-of-plane reach at 1.0 um z spacing -- and "
                  "was never appropriate to KAM; given the range back, KAM "
                  "still peaks at 1.08 um, so the floor cost it nothing. The "
                  "flood fill is scored **after its merge step**, which is the "
                  "arm as deployed. The parameter column identifies the exact "
                  "winning configuration.", "",
                  "| arm | parameters | cells | recovered@90 | rate | contamination | fused | ARI |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, row in winners.items():
            if row.get("percentile") is not None:
                params = (f"pct={row.get('percentile')}, "
                          f"r={row.get('kam_radius_um')} um, "
                          f"min={row.get('min_cell_size')}, "
                          f"conn={row.get('connectivity')}")
            else:
                params = (f"r={row.get('footprint_radius_um')} um, "
                          f"tol={row.get('footprint_tolerance')}, "
                          f"local={row.get('local_threshold_deg')}, "
                          f"min={row.get('min_cell_size')}")
            lines.append(
                f"| {name} | {params} | {row.get('n_cells_pred')} | "
                f"{row.get('recovered_at_90')} | "
                f"{(row.get('recovery_rate_at_90') or 0):.3f} | "
                f"{(row.get('contamination') or 0):.4f} | "
                f"{row.get('fused_true_cells')} | {(row.get('ari') or 0):.3f} |")
        lines.append("")

    if primary_rep.get("summary"):
        lines += ["### 1b. The same table on independent volumes", "",
                  "Section 1 is one phantom. These are independent",
                  "realisations of it under the section 7 protocol -- flood",
                  "fill blind on the defaults, KAM re-tuned on each volume.", "",
                  "| arm | n | recovered@90 rate | ARI |",
                  "|---|---|---|---|"]
        for arm, entry in primary_rep["summary"].items():
            ari = ("--" if entry.get("ari_mean") is None
                   else f"{entry['ari_mean']:.4f} +- {entry['ari_sd']:.4f}")
            lines.append(f"| {arm} | {entry['n']} | "
                         f"{entry['mean']:.4f} +- {entry['sd']:.4f} | {ari} |")
        lines.append("")
        gaps = primary_rep.get("paired") or {}
        if gaps:
            lines += ["Paired within each volume: " + "; ".join(
                f"{arm} {g['mean']:+.4f} +- {g['sd']:.4f} "
                f"(smallest {g['min']:+.4f})"
                for arm, g in gaps.items()) + ".", ""]

    dimension = (why or {}).get("dimension")
    if dimension:
        lines += ["## 2. The advantage is three-dimensional", "",
                  "Slice-wise segmentation followed by linking labels through z",
                  "is what one does without a volumetric algorithm. It is the",
                  "honest 2D baseline: an unlinked stack scores zero by",
                  "construction, since no single slice holds 90 % of a cell that",
                  "spans several layers. These 3D values are a separate",
                  "mechanism run, not the section 1 primary winner: the flood-fill",
                  "run uses its own configuration and recovers 2010 cells.", "",
                  "| arm | 2D + link | 3D | gain |", "|---|---|---|---|"]
        for arm, entry in dimension.items():
            if not isinstance(entry, dict) or "3D" not in entry:
                continue
            lines.append(
                f"| {arm} | {entry['2D+link']['recovered_at_90']} | "
                f"{entry['3D']['recovered_at_90']} | "
                f"+{entry['gain_from_3d_cells']} |")
        lines.append("")

    if dimrep and dimrep.get("summary"):
        lines += ["### 2b. The same claim on independent volumes", "",
                  "The 2D baseline is a construction rather than a search, so",
                  "a single number gives no sense of how much of it belongs to",
                  "the one volume it was measured on. Repeated on independent",
                  "realisations of the primary phantom, at the study defaults:",
                  "",
                  "| arm | n | 2D+link recovered | 3D recovered | gain |",
                  "|---|---|---|---|---|"]
        for arm, entry in dimrep["summary"].items():
            lines.append(
                f"| {arm} | {entry['n']} | "
                f"{entry['recovered_2d']['mean']:.1f} "
                f"+- {entry['recovered_2d']['sd']:.1f} | "
                f"{entry['recovered_3d']['mean']:.1f} "
                f"+- {entry['recovered_3d']['sd']:.1f} | "
                f"{entry['gain_cells']['mean']:+.1f} "
                f"+- {entry['gain_cells']['sd']:.1f} |")
        lines.append("")

    walls = (why or {}).get("walls")
    if walls:
        lines += ["## 3. Where each method fails", "",
                  "The phantom broadens a fraction of its wall area, so a wider",
                  "wall is a fainter one with a weaker KAM ridge. If a method",
                  "fails because it needs a closed ridge, the interfaces it",
                  "fuses should have wider walls than those it keeps.", "",
                  "| arm | fused interfaces | wall width fused | intact | ratio | p |",
                  "|---|---|---|---|---|---|"]
        for arm, entry in walls.get("arms", {}).items():
            if not entry.get("fused_interfaces"):
                continue
            lines.append(
                f"| {arm} | {entry['fused_interfaces']} | "
                f"{entry['median_width_fused_um']:.3f} um | "
                f"{entry['median_width_intact_um']:.3f} um | "
                f"{entry['width_ratio_fused_over_intact']:.2f} | "
                f"{entry['p_value_fused_walls_are_wider']:.1e} |")
        lines.append("")

        # Interfaces sharing a cell are not independent, so the edge-level p
        # above overstates its own confidence.  Resampling cells is the
        # honest interval, and it is what the claim should rest on.
        boot = {a: e["cluster_bootstrap"] for a, e in walls.get("arms", {}).items()
                if e.get("cluster_bootstrap", {}).get("n_boot")}
        if boot:
            any_entry = next(iter(boot.values()))
            lines += [
                "Interfaces that share a cell are not independent draws, so the",
                "edge-level test above overstates its confidence. Resampling",
                f"cells instead ({any_entry['n_boot']} bootstrap replicates over",
                f"{any_entry['n_cluster_cells']} cells) gives the interval the",
                "claim actually rests on:", "",
                "| arm | width ratio | 95 % CI | median difference | 95 % CI | excludes parity |",
                "|---|---|---|---|---|---|"]
            for arm, entry in boot.items():
                lo, mid, hi = entry["ratio_median_ci95"]
                dlo, dmid, dhi = entry["difference_median_um_ci95"]
                lines.append(
                    f"| {arm} | {mid:.3f} | [{lo:.3f}, {hi:.3f}] | "
                    f"{dmid * 1000:.1f} nm | [{dlo * 1000:.1f}, {dhi * 1000:.1f}] nm | "
                    f"{'yes' if lo > 1.0 else 'no'} |")
            lines += ["",
                      "The differential result survives the correction: KAM's",
                      "interval clears parity, the flood fill's straddles it.", ""]

    interior = (why or {}).get("interior")
    if interior:
        lines += ["## 3b. Why KAM fails: the low-KAM interior collapses", "",
                  "A KAM kernel of radius r raises KAM within r of any wall, so a",
                  "low-KAM component can only form in the inner core of a cell.",
                  "As cells shrink that core vanishes, and cells with no core at",
                  "all must share a component with a neighbour -- under-segmentation",
                  "that nothing downstream can undo. This is measurable on",
                  "experimental data, because it needs only the kernel radius and",
                  "the cell size, not ground truth.", "",
                  f"Kernel radius {interior['kernel_radius_um']} um throughout; "
                  "only the cell size changes.", "",
                  "| phantom | cell diameter | r / d | median interior fraction | cells with no interior |",
                  "|---|---|---|---|---|"]
        for name, e in sorted(interior["phantoms"].items(),
                              key=lambda kv: -kv[1]["mean_cell_diameter_um"]):
            lines.append(
                f"| {name} | {e['mean_cell_diameter_um']:.2f} um | "
                f"{e['kernel_over_diameter']:.2f} | "
                f"{e['median_interior_fraction']:.3f} | "
                f"{e['fraction_of_cells_with_no_interior']:.1%} |")
        lines.append("")

    bre = (general or {}).get("breadth", {}).get("arms")
    if bre:
        lines += ["## 4. Can the parameters be chosen without ground truth?", "",
                  "Each arm is measured against its **own** best, so this",
                  "compares tunability rather than accuracy. A method whose",
                  "accuracy collapses a step away from an exactly-tuned point",
                  "cannot be tuned on experimental data, however high that point",
                  "scores. Measured on the refined grids, pooled over the",
                  "four strain phantoms, each against its own best there.", "",
                  "| arm | configurations | mean best rate | within 95 % | within 90 % | within 80 % |",
                  "|---|---|---|---|---|---|"]
        for arm, entry in bre.items():
            lines.append(
                f"| {arm} | {entry['n_configurations']} | "
                f"{entry.get('mean_best_rate', 0):.4f} | "
                f"{entry.get('fraction_within_95pct_of_best', 0):.2%} | "
                f"{entry.get('fraction_within_90pct_of_best', 0):.2%} | "
                f"{entry.get('fraction_within_80pct_of_best', 0):.2%} |")
        lines.append("")

    rows = (general or {}).get("transfer", {}).get("rows")
    if rows:
        import pandas as _pd

        frame = _pd.DataFrame([r for r in rows if r.get("status") == "ok"])
        if not frame.empty:
            lines += ["## 4b. Do the parameters transfer across strain?", "",
                      "Each strain's own optimum applied to every other strain.",
                      "Rows are where the parameters came from, columns where they",
                      "were applied; the diagonal is the matched case.", ""]
            for arm, group in frame.groupby("arm"):
                matrix = group.pivot_table(index="source_strain",
                                           columns="target_strain",
                                           values="recovery_rate_at_90")
                lines += [f"**{arm}**", "",
                          "| from \\ to | " + " | ".join(matrix.columns) + " |",
                          "|" + "---|" * (len(matrix.columns) + 1)]
                for source, row in matrix.iterrows():
                    lines.append(f"| {source} | " +
                                 " | ".join(f"{v:.3f}" for v in row) + " |")
                lines.append("")

    if strain is not None:
        lines += ["## 5. How the parameters move with strain", "",
                  "Cell size falls and misorientation rises with strain",
                  "(Zelenika et al., Sci Rep 15, 8655 (2025)). 6.2 % is",
                  "extrapolated beyond the paper's measured 0.6-4.6 % range.", "",
                  "Read the local threshold column against section 5b: on the",
                  "refined grid, and with the flood fill scored after its merge,",
                  "it barely moves.", ""]
        columns = [c for c in ["strain_percent", "arm", "cell_diameter_um",
                               "chi_sigma_deg", "footprint_radius_um",
                               "local_threshold_deg", "percentile",
                               "min_cell_size", "kam_radius_um", "n_cells_pred",
                               "recovered_at_90", "rate", "ari"] if c in strain]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "---|" * len(columns))
        for _, row in strain.iterrows():
            lines.append("| " + " | ".join(
                f"{row[c]:.4g}" if isinstance(row[c], float) else str(row[c])
                for c in columns) + " |")
        lines.append("")

    if tuning:
        lines += ["### 5b. What one global value of that knob costs", "",
                  "Each arm has one axis anybody would actually sweep -- the",
                  "local threshold for the flood fill, the percentile for KAM.",
                  "Fixing it at a single value for every strain, against tuning",
                  "it per volume:", "",
                  "| arm | axis | per-strain optima | best single value | tuned | fixed | mean cost | worst |",
                  "|---|---|---|---|---|---|---|---|"]
        for arm, entry in tuning.items():
            optima = ", ".join(f"{v:g}" for v in entry["per_phantom_optimum"].values())
            lines.append(
                f"| {arm} | {entry['axis']} | {optima} | "
                f"{entry['best_single_value']:g} | {entry['mean_tuned']:.4f} | "
                f"{entry['mean_fixed']:.4f} | {entry['mean_cost']:.4f} | "
                f"{entry['worst_cost']:.4f} |")
        lines += ["",
                  "Two things to read here, and only one of them separates the",
                  "arms.", "",
                  "**The merge is what makes the threshold transferable.** After",
                  "it, one global threshold serves all four strains for 0.0010",
                  "mean and 0.0040 worst; the same fixed value costs the unmerged",
                  "arm six times as much, and its optimum still drifts upward",
                  "with strain. The arm as deployed needs no per-volume tuning at",
                  "all.", "",
                  "**An earlier reading of this was wrong and is worth recording.**",
                  "Measured on the unmerged arm under the retired admissibility",
                  "rule, the optimum threshold rose monotonically with strain and",
                  "fitted a clean power law in the misorientation width. That",
                  "trend was largely manufactured by the rule: it disqualified",
                  "over-segmenting configurations before the merge could act, and",
                  "so pushed the threshold up in proportion to how much each",
                  "phantom over-segmented. Assessed after the merge, the optimum",
                  "is the same value at 2.4, 3.5 and 4.6 % and rises only at",
                  "6.2 %.", "",
                  "**This does not separate the arms**, for the same reason",
                  "section 6 gives: KAM fixes its percentile for 2.1 % of its own",
                  "ceiling, against 0.15 % for the merged arm. Both arms tolerate",
                  "one frozen number. What separates them is section 4 and the",
                  "ceilings themselves.", ""]

    if economy:
        lines += ["## 6. How many parameters actually have to be chosen", "",
                  "A method with seven parameters and a method with one are not",
                  "equally usable on data without ground truth, even at equal",
                  "accuracy. Each axis below is frozen at a single global value",
                  "while every other axis is re-tuned per strain; the cost is",
                  "the loss in recovery rate at tau = 0.9, averaged over the",
                  "four strain phantoms.", "",
                  "The two flood-fill tables are read off the staged search's",
                  "finalists; the KAM one is read off its complete 9,108-point",
                  "grid, which is scored in full. The KAM figures are thus the",
                  "better measured of the three, which matters because they are",
                  "the ones that complicate the story. `best fixed` below means",
                  "one axis fixed while all other axes are re-tuned; it is not",
                  "the joint fallback default used by the blind replicate runs.", ""]
        for arm in ("flood fill", "flood fill + merge", "KAM threshold"):
            marg = (economy.get(arm) or {}).get("marginal") or {}
            if not marg:
                continue
            lines += [f"**{arm}**", "",
                      "| axis | values searched | best fixed (marginal) | mean cost | worst |",
                      "|---|---|---|---|---|"]
            for axis, row in sorted(marg.items(), key=lambda kv: kv[1]["mean_cost"]):
                lines.append(
                    f"| {axis} | {row['values_searched']} | {row['best_fixed']:g} | "
                    f"{row['mean_cost']:.4f} | {row['worst_cost']:.4f} |")
            lines.append("")

        lines += ["Only the local threshold is worth a sweep. Two axes cost",
                  "nothing at all in the unmerged arm and four in the merged",
                  "one, and the footprint radius -- the axis the cap was",
                  "imposed on -- costs 0.006 anywhere on its plateau.", "",
                  "**This does not separate the arms, and should not be read",
                  "as though it did.** Asked the same question, the KAM arm is",
                  "just as insensitive: its worst axis costs 0.0016. Measured",
                  "against each arm's own ceiling the three are the same to",
                  "within a percentage point -- 1.4 % for flood fill, 2.0 %",
                  "merged, 1.9 % for KAM. Every method here reaches its own",
                  "optimum with any single axis pinned, because the axes",
                  "compensate for one another.", "",
                  "What section 6 establishes is therefore a statement about",
                  "flood fill on its own -- it can be deployed on data with no",
                  "ground truth by choosing one number -- and not a comparative",
                  "claim. The comparison is carried by section 4, where an",
                  f"arbitrary configuration is near-optimal {_pct('flood fill + merge', 90)} "
                  "of the time for",
                  f"the merged arm and {_pct('KAM threshold', 90)} of the time for KAM, and by the",
                  f"accuracy itself: these two ceilings are {_ceiling('flood fill + merge')} "
                  f"and {_ceiling('KAM threshold')}. The",
                  "KAM optimum is reachable but not findable.", "",
                  "Freezing axes one at a time does not license freezing them",
                  "together, and the staged search cannot answer that: its",
                  "finalist funnel keeps only what leads at each stage, so the",
                  "fully frozen combination never appears in its results. It was",
                  "therefore run directly (`capped_search.py --frozen`), pinning",
                  "every axis and sweeping the threshold alone.", ""]
        lines += ["| arm | strain | full search | frozen 1-D sweep | cost |",
                  "|---|---|---|---|---|"]
        for arm in ("flood fill", "flood fill + merge"):
            j = (economy.get(arm) or {}).get("joint") or {}
            for row in j.get("per_strain", []):
                lines.append(
                    f"| {arm} | {row['strain_key']} | {row['full_search']:.4f} | "
                    f"{row['frozen_sweep']:.4f} | {row['cost']:+.4f} |")
        lines.append("")
        summary = []
        for arm in ("flood fill", "flood fill + merge"):
            j = (economy.get(arm) or {}).get("joint") or {}
            if j:
                summary.append(f"{arm} {j['mean_cost']:.4f} mean / "
                               f"{j['worst_cost']:.4f} worst")
        if summary:
            lines += ["Cost of the whole reduction -- six axes to one in the",
                      "unmerged arm, seven to one in the merged one: "
                      + "; ".join(summary) + ".", ""]
        lines += ["At 4.6 % strain the frozen sweep is *better* than the full",
                  "search. That is not a rounding artefact and not a point in",
                  "the defaults' favour so much as a caveat on the search: the",
                  "staged funnel discards configurations that a direct sweep",
                  "still finds. Reported optima are therefore lower bounds.", "",
                  "Two cliffs bound the plateau. A footprint radius of 0.9 um",
                  "costs 0.64 -- the effective floor is 1.08 um, *inside* the",
                  "searched range rather than below it -- and a min_cell_size of",
                  "40 or more costs 0.15 at 6.2 % strain, 0.59 at 80.", ""]

    if replicates:
        strains = ["2p4", "3p5", "4p6", "6p2"]
        arms = ["flood fill", "flood fill + merge", "KAM threshold"]
        proto = replicates.get("protocol") or {}
        n_rep = proto.get("replicates_per_strain")
        lines += ["## 7. Does the comparison survive a different roll of the dice?", "",
                  "Sections 1-6 rest on one phantom realisation per strain",
                  f"searched with one seed. Here each strain gets {n_rep}",
                  "further independent realisations and the comparison is",
                  "repeated on every one.", "",
                  "The protocol is deliberately asymmetric, and against the",
                  "conclusion. The flood-fill arm runs **blind**: every axis",
                  "pinned to the fallback defaults in `capped_search.py`, chosen on",
                  "r0 and never re-derived, so these phantoms are out of",
                  "sample. The KAM arm is **re-tuned on every replicate** over",
                  "828 configurations. If flood fill still wins, the gap cannot",
                  "be explained by unequal tuning effort.", ""]

        ver = replicates.get("verification") or []
        if ver:
            lines += ["The KAM arm's 828-point grid first has to be shown",
                      "sufficient. On r0, against the full 9,108-point grid:", "",
                      "| strain | full grid | reduced grid | difference |",
                      "|---|---|---|---|"]
            for row in ver:
                lines.append(f"| {row['strain_key']} | {row['full_grid']:.4f} | "
                             f"{row['reduced_grid']:.4f} | {row['difference']:+.4f} |")
            lines.append("")

        summary = replicates.get("summary") or {}
        if summary:
            lines += ["Recovery rate at tau = 0.9, mean +- sd over replicates:", "",
                      "| arm | " + " | ".join(strains) + " |",
                      "|---|" + "---|" * len(strains)]
            for arm in arms:
                cells = []
                for strain in strains:
                    e = (summary.get(arm) or {}).get(strain)
                    cells.append(f"{e['mean']:.4f} +- {e['sd']:.4f}" if e else "--")
                lines.append(f"| {arm} | " + " | ".join(cells) + " |")
            lines.append("")

        if summary:
            blind_rows = []
            for arm in ("flood fill", "flood fill + merge"):
                for strain in strains:
                    e = (summary.get(arm) or {}).get(strain) or {}
                    if e.get("blind_mean") is None:
                        continue
                    blind_rows.append((arm, strain, e))
            if blind_rows:
                lines += ["Those runs still choose one number per phantom, the",
                          "local threshold. Fixing that too -- the fallback",
                          "default in `capped_search.py`, with no tuning on the",
                          "replicate whatsoever --",
                          "costs this much:", "",
                          "| arm | strain | threshold swept | threshold fixed | cost |",
                          "|---|---|---|---|---|"]
                for arm, strain, e in blind_rows:
                    lines.append(
                        f"| {arm} | {strain} | {e['mean']:.4f} +- {e['sd']:.4f} | "
                        f"{e['blind_mean']:.4f} +- {e['blind_sd']:.4f} | "
                        f"{e['mean'] - e['blind_mean']:.4f} |")
                lines += ["",
                          "The cost is small on average but not evenly spread,",
                          "and the two arms need the threshold at opposite ends",
                          "of the range: unmerged it matters most at low strain",
                          "(0.066 at 2.4 %), merged at high strain (0.037 at",
                          "6.2 %). Sweeping one number is cheap, so sweep it --",
                          "but even wholly untuned the worst blind mean, 0.405,",
                          "is four times the best KAM arm managed with per-",
                          "replicate tuning.", ""]

        pair = replicates.get("paired") or {}
        rendered = False
        for arm in ("flood fill", "flood fill + merge"):
            entry = pair.get(arm) or {}
            if not entry.get("n_pairs"):
                continue
            if not rendered:
                lines += ["Each replicate pairs the two arms on the *same*",
                          "phantom, so the difference is measured within a",
                          "realisation rather than across the noise between",
                          "them:", "",
                          "| arm minus KAM | pairs | mean gap | sd | smallest gap | all positive | paired t | p |",
                          "|---|---|---|---|---|---|---|---|"]
                rendered = True
            lines.append(
                f"| {arm} | {entry['n_pairs']} | {entry['mean']:+.4f} | "
                f"{entry['sd']:.4f} | {entry['min']:+.4f} | "
                f"{'yes' if entry.get('all_positive') else 'no'} | "
                + (f"{entry['t']:.1f} | {entry['p']:.1e} |"
                   if entry.get('p') is not None else "-- | -- |"))
        if rendered:
            lines.append("")

        oos = replicates.get("out_of_sample") or {}
        rows = oos.get("flood fill + merge") or oos.get("flood fill") or []
        if rows:
            lines += ["The defaults were chosen on r0, so r0 could in",
                      "principle be flattered by them. It is not: the frozen",
                      "run on r0 sits inside the spread of the replicates it",
                      "never saw.", "",
                      "| strain | r0, full search | r0, frozen | replicates (blind) |",
                      "|---|---|---|---|"]
            for row in rows:
                lines.append(
                    f"| {row['strain_key']} | {row['r0_full_search']:.4f} | "
                    f"{row['r0_frozen']:.4f} | {row['replicates_mean']:.4f} "
                    f"+- {row['replicates_sd']:.4f} |")
            lines.append("")

    if seeds:
        decomp = seeds.get("decomposition") or {}
        rendered = False
        for arm in ("flood fill", "flood fill + merge"):
            rows = decomp.get(arm) or []
            if not rows:
                continue
            if not rendered:
                lines += ["### 7b. Is that spread the microstructure or the algorithm?", "",
                          "The replicate spread confounds two things: the",
                          "microstructure changing between realisations, and the",
                          "flood fill's own stochasticity -- it grows regions from",
                          "seed points in an order that depends on a random seed.",
                          "Holding the phantom fixed at r0 and sweeping only the",
                          "seed separates them. The seed contributes nothing:", "",
                          "| arm | strain | seeds | mean | seed sd | seed range | replicate sd | seed / replicate |",
                          "|---|---|---|---|---|---|---|---|"]
                rendered = True
            for row in rows:
                rep = ("--" if row.get("replicate_sd") is None
                       else f"{row['replicate_sd']:.4f}")
                ratio = ("--" if row.get("ratio_seed_to_replicate") is None
                         else f"{row['ratio_seed_to_replicate']:.2f}")
                lines.append(
                    f"| {arm} | {row['strain_key']} | {row['n_seeds']} | "
                    f"{row['seed_mean']:.4f} | {row['seed_sd']:.4f} | "
                    f"{row['seed_range']:.4f} | {rep} | {ratio} |")
        if rendered:
            lines.append("")

        det = seeds.get("determinism") or []
        if det:
            unanimous = all(abs(r["ari_against_seed_0"] - 1.0) < 1e-12 for r in det)
            distinct = all(not r["label_arrays_identical"] for r in det)
            lines += ["A quantity that never moves is equally consistent with",
                      "the seed never reaching the algorithm, so that is checked",
                      "directly rather than assumed. The seed does reach it and",
                      "does change the output -- the label *numbering* differs on",
                      "every seed, unseeded runs included. What does not change",
                      "is the partition those labels describe:",
                      "`flood_fill_dfxm_two_stage` collects its seeds",
                      "deterministically and then sorts them by size.", "",
                      "**That invariance is conditional, and the condition is the",
                      "global threshold.** It is not a blanket property of the",
                      "algorithm. The global test compares each candidate voxel",
                      "against the region's *running* mean, whose value depends on",
                      "the order voxels were added, so wherever that test actually",
                      "binds the partition becomes order-dependent. Sweeping it on",
                      "the primary phantom with everything else fixed: off, 2.0,",
                      "0.873 and 0.30 deg all give a spread of 0 cells over five",
                      "seeds, while 0.10 deg gives a spread of 5. The defaults used",
                      "throughout this study disable the global threshold, which is",
                      "why the measured spread is zero.", "",
                      "The earlier synthetic benchmark, which enabled it at a tight",
                      "local threshold, correspondingly reported 358-362 cells over",
                      "five seed orders -- reproduced here exactly. Both",
                      "observations are consistent once the condition is stated.", ""]
            lines += [f"Across {len(det)} comparisons on four phantoms "
                      f"(seeds 1-3 and one seeded from `std::random_device`): "
                      f"ARI against seed 0 is exactly 1 in "
                      f"{'every' if unanimous else 'most'} case, cell counts "
                      f"match exactly, and the raw label arrays differ in "
                      f"{'every' if distinct else 'most'} case.", "",
                      "So the error bars in section 7 are microstructure",
                      "variance, not the algorithm reshuffling its own output.",
                      "The single exception is the merged arm at 3.5 %, whose",
                      "seed sd is 0.0003 -- the merge step walks cells in label",
                      "order, so ties between equal-sized candidates can break",
                      "differently. It is two orders of magnitude below the",
                      "replicate spread.", ""]

    lines += ["## Provenance", "",
              "- Phantoms: `strain_phantoms.py`, from the measured DFXM trend.",
              "- Search: `capped_search.py`, both arms to their own optimum.",
              "- Mechanism: `why_flood_fill.py`. Generalisation: `generalisation.py`.",
              "- Seed ablation: `seed_ablation.py`, results in",
              "  `runs/seeds/`.",
              "- Replicates: `replicates.py`, results in",
              "  `runs/replicates/`; the KAM grid check is in",
              "  `runs/kamverify/`.",
              "- Parameter economy: `parameter_economy.py`; the frozen searches",
              "  are in `runs/frozen/` and the defaults they price",
              "  are `capped_search.DEFAULTS`.",
              "- Selection policy and its alternatives are recorded in",
              "  `runs/primary/capped_report.json`.",
              "* Sections 1, 4, 5 and 5b come from the refined sweeps in",
              "  `runs/refined/` -- `refine.py kam|flood|merged`. Section 6 and",
              "  the transfer matrix in 4b are still read off the coarse staged",
              "  search, whose axes the refined grids do not reproduce.", ""]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    winners = refined_winners() or primary_winners()
    why = load_json(WHY)
    general = load_json(GENERAL)
    economy = load_json(ECONOMY)
    replicates = load_json(REPLICATES)
    seeds = load_json(SEEDS)
    dimrep = load_json(DIMREP)
    strain = refined_strain_table()
    if strain is None:
        strain = strain_table()
    refined_bre = refined_breadth()
    if refined_bre:
        general = dict(general or {})
        general["breadth"] = dict(general.get("breadth") or {})
        general["breadth"]["coarse_arms"] = general["breadth"].get("arms")
        general["breadth"]["arms"] = refined_bre

    figure_arms(winners, OUT / "arms_at_optimum.png")
    figure_dimension(why, OUT / "two_versus_three_d.png")
    if strain is not None:
        strain.to_csv(OUT / "strain_optima.csv", index=False)

    tuning = refined_tuning_profile()
    text = markdown(winners, why, general, strain, economy, replicates, seeds,
                    dimrep, tuning)
    (OUT / "BENCHMARK.md").write_text(text + "\n")
    (OUT / "benchmark.json").write_text(json.dumps(
        {"winners": winners, "why": why, "generalisation": general,
         "refined_tuning_profile": tuning,
         "parameter_economy": economy, "replicates": replicates,
         "seed_ablation": seeds, "dimension_replicates": dimrep},
        indent=2, sort_keys=True, default=float) + "\n")
    print(text)
    print(f"\nwritten: {OUT/'BENCHMARK.md'}")


if __name__ == "__main__":
    main()
