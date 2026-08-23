#!/usr/bin/env python3
"""Does the comparison survive a different roll of the dice?

Every number in the study so far comes from one phantom realisation per strain
(``_r0``) searched with ``seed: 0``.  That is a single sample, so it carries no
uncertainty, and a reader has no way to tell a real 15-fold gap between the
arms from a lucky one.  This builds independent realisations of each strain and
repeats the comparison on each.

The protocol is deliberately **asymmetric**, and against the conclusion:

``flood fill``   Runs blind.  Every axis is pinned to ``capped_search.DEFAULTS``
                 -- chosen on r0 and never re-derived -- so the replicates are
                 genuinely out of sample.  Two variants are reported: sweeping
                 the local threshold, and fixing that too, which is a run with
                 no tuning on the replicate whatsoever.

``KAM``          Is re-tuned on every replicate over 828 configurations, its
                 percentile axis whole.

So the KAM arm is handed per-replicate tuning that the flood-fill arm is
denied.  If flood fill still wins, the result cannot be explained by unequal
tuning effort -- which is the objection a single-realisation table invites.

Usage::

    python replicates.py verify  --workers 14   # reduced KAM grid vs the full one
    python replicates.py run     --workers 14
    python replicates.py report
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

OUT = HERE / "analysis"
ROOT = HERE / "runs/replicates"
FULL_ROOT = HERE / "runs/strain"
VERIFY_ROOT = HERE / "runs/kamverify"
FULL_REPLICATE_VERIFY_ROOT = HERE / "runs/kamverify_replicates"
PHANTOM_DIR = HERE / "phantoms"

STRAINS = ("2p4", "3p5", "4p6", "6p2")
REPLICATES = (1, 2, 3, 4, 5)

#: Sections 1-4 are all measured on the primary phantom, which is a single
#: volume.  These are independent realisations of it, run under the same
#: protocol, so the headline table carries an uncertainty rather than resting
#: on one roll.
PRIMARY_NAME = "primary_6p2_consistent"
PRIMARY_REPLICATES = (1, 2, 3, 4, 5)
ARMS = ("flood fill", "flood fill + merge", "KAM threshold")
STAGES = ("markers", "merge", "baseline", "finalise")


def rows_for(directory: Path, arm: str) -> list[dict]:
    import capped_search as cs

    return [r for r in cs.read_rows(directory / "final.jsonl")
            if r.get("status") == "ok"
            and (r.get("arm") or "flood fill") == arm
            and not cs._disqualified(r)]


def best(directory: Path, arm: str, fixed_threshold: bool = False) -> dict | None:
    """The winner for one arm in one result directory."""

    import capped_search as cs

    # KAM's complete cheap-score grid is stored in baseline.jsonl.  The
    # final.jsonl file is only a finalist subset, so it cannot establish a
    # full-grid KAM winner when baseline.jsonl is available.
    source = directory / "baseline.jsonl" if arm == "KAM threshold" and \
        (directory / "baseline.jsonl").exists() else directory / "final.jsonl"
    rows = [r for r in cs.read_rows(source)
            if r.get("status") == "ok"
            and (r.get("arm") or "KAM threshold") == arm
            and not cs._disqualified(r)]
    if fixed_threshold:
        target = cs.DEFAULT_LOCAL_THRESHOLD_DEG.get(arm)
        rows = [r for r in rows if r.get("local_threshold_deg") == target]
    return min(rows, key=cs.strict_recovery_key) if rows else None


# ------------------------------------------------------------------- the runs

def run(workers: int) -> None:
    for strain in STRAINS:
        for rep in REPLICATES:
            phantom = PHANTOM_DIR / f"strain_{strain}_r{rep}.npz"
            if not phantom.exists():
                print(f"skipping {strain} r{rep}: no phantom", flush=True)
                continue
            out = ROOT / f"{strain}_r{rep}"
            for stage in STAGES:
                print(f"\n=== {strain} r{rep}: {stage} ===", flush=True)
                subprocess.run(
                    [sys.executable, str(HERE / "capped_search.py"), stage,
                     "--frozen", "--workers", str(workers),
                     "--phantom", str(phantom), "--out", str(out)],
                    check=True)


def run_primary(workers: int) -> None:
    for rep in PRIMARY_REPLICATES:
        phantom = PHANTOM_DIR / f"{PRIMARY_NAME}_r{rep}.npz"
        if not phantom.exists():
            print(f"skipping primary r{rep}: no phantom", flush=True)
            continue
        out = ROOT / f"primary_r{rep}"
        for stage in STAGES:
            print(f"\n=== primary r{rep}: {stage} ===", flush=True)
            subprocess.run(
                [sys.executable, str(HERE / "capped_search.py"), stage,
                 "--frozen", "--workers", str(workers),
                 "--phantom", str(phantom), "--out", str(out)],
                check=True)


def primary_block() -> dict:
    """The headline table's arms, repeated on independent primary volumes."""

    records = []
    for rep in PRIMARY_REPLICATES:
        directory = ROOT / f"primary_r{rep}"
        if not (directory / "final.jsonl").exists():
            continue
        for arm in ARMS:
            row = best(directory, arm)
            if not row:
                continue
            records.append({
                "replicate": rep, "arm": arm,
                "recovery_rate_at_90": row["recovery_rate_at_90"],
                "recovered_at_90": row.get("recovered_at_90"),
                "n_cells_true": row.get("n_cells_true"),
                "n_cells_pred": row.get("n_cells_pred"),
                "contamination": row.get("contamination"),
                "ari": row.get("ari"),
            })
    if not records:
        return {}

    summary, gaps = {}, {}
    for arm in ARMS:
        values = [r["recovery_rate_at_90"] for r in records if r["arm"] == arm]
        aris = [r["ari"] for r in records if r["arm"] == arm and r.get("ari")]
        if not values:
            continue
        summary[arm] = {
            "n": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ari_mean": float(np.mean(aris)) if aris else None,
            "ari_sd": (float(np.std(aris, ddof=1))
                       if len(aris) > 1 else 0.0) if aris else None,
        }
    for arm in ("flood fill", "flood fill + merge"):
        paired_gaps = []
        for rep in PRIMARY_REPLICATES:
            a = [r for r in records if r["arm"] == arm and r["replicate"] == rep]
            b = [r for r in records
                 if r["arm"] == "KAM threshold" and r["replicate"] == rep]
            if a and b:
                paired_gaps.append(a[0]["recovery_rate_at_90"]
                                   - b[0]["recovery_rate_at_90"])
        if paired_gaps:
            gaps[arm] = {
                "n": len(paired_gaps),
                "mean": float(np.mean(paired_gaps)),
                "sd": (float(np.std(paired_gaps, ddof=1))
                       if len(paired_gaps) > 1 else 0.0),
                "min": float(min(paired_gaps)),
            }
    return {"records": records, "summary": summary, "paired": gaps}


def verify(workers: int) -> None:
    """The reduced KAM grid must reach the full grid's answer, on r0."""

    for strain in STRAINS:
        phantom = PHANTOM_DIR / f"strain_{strain}_r0.npz"
        for stage in ("baseline", "finalise"):
            print(f"\n=== verify {strain}: {stage} ===", flush=True)
            subprocess.run(
                [sys.executable, str(HERE / "capped_search.py"), stage,
                 "--frozen", "--workers", str(workers),
                 "--phantom", str(phantom), "--out", str(VERIFY_ROOT / strain)],
                check=True)


def verify_full_replicates(workers: int) -> None:
    """Run the complete KAM grid on every strain replicate.

    Results go into a new store; the existing reduced-grid outputs remain
    untouched and can still be compared with the full grid afterward.
    """

    for strain in STRAINS:
        for rep in REPLICATES:
            phantom = PHANTOM_DIR / f"strain_{strain}_r{rep}.npz"
            if not phantom.exists():
                continue
            out = FULL_REPLICATE_VERIFY_ROOT / f"{strain}_r{rep}"
            print(f"\n=== full KAM verify {strain} r{rep} ===", flush=True)
            subprocess.run(
                [sys.executable, str(HERE / "capped_search.py"), "baseline",
                 "--workers", str(workers), "--phantom", str(phantom),
                 "--out", str(out)], check=True)


# ------------------------------------------------------------------ the tables

def verification() -> list[dict]:
    out = []
    for strain in STRAINS:
        full = best(FULL_ROOT / strain, "KAM threshold")
        cut = best(VERIFY_ROOT / strain, "KAM threshold")
        if not full or not cut:
            continue
        out.append({
            "strain_key": strain,
            "full_grid": full["recovery_rate_at_90"],
            "reduced_grid": cut["recovery_rate_at_90"],
            "difference": cut["recovery_rate_at_90"] - full["recovery_rate_at_90"],
        })
    return out


def replicate_grid_verification() -> list[dict]:
    """Compare the reduced and full KAM grids on all available replicates."""

    import capped_search as cs

    out = []
    for strain in STRAINS:
        for rep in REPLICATES:
            full = best(FULL_REPLICATE_VERIFY_ROOT / f"{strain}_r{rep}",
                        "KAM threshold")
            reduced = best(ROOT / f"{strain}_r{rep}", "KAM threshold")
            full_rows = cs.read_rows(
                FULL_REPLICATE_VERIFY_ROOT / f"{strain}_r{rep}" /
                "baseline.jsonl")
            # A cancelled or resumed run must never be mistaken for a full
            # grid.  The exact grid size is fixed by capped_search's baseline
            # axes and is checked here before any winner is used.
            if len(full_rows) != 9108 or not full or not reduced:
                continue
            out.append({
                "strain_key": strain,
                "replicate": rep,
                "full_grid": full["recovery_rate_at_90"],
                "reduced_grid": reduced["recovery_rate_at_90"],
                "difference": reduced["recovery_rate_at_90"] -
                              full["recovery_rate_at_90"],
            })
    return out


def collect() -> list[dict]:
    """One record per strain, replicate and arm."""

    records = []
    for strain in STRAINS:
        for rep in REPLICATES:
            directory = ROOT / f"{strain}_r{rep}"
            if not (directory / "final.jsonl").exists():
                continue
            for arm in ARMS:
                row = best(directory, arm)
                if not row:
                    continue
                record = {
                    "strain_key": strain, "replicate": rep, "arm": arm,
                    "recovery_rate_at_90": row["recovery_rate_at_90"],
                    "recovered_at_90": row.get("recovered_at_90"),
                    "n_cells_true": row.get("n_cells_true"),
                    "contamination": row.get("contamination"),
                    "ari": row.get("ari"),
                    "local_threshold_deg": row.get("local_threshold_deg"),
                }
                if arm != "KAM threshold":
                    blind = best(directory, arm, fixed_threshold=True)
                    record["blind_rate_at_90"] = (
                        blind["recovery_rate_at_90"] if blind else None)
                records.append(record)
    return records


def summarise(records: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        per_strain = {}
        for strain in STRAINS:
            values = [r["recovery_rate_at_90"] for r in records
                      if r["arm"] == arm and r["strain_key"] == strain]
            if not values:
                continue
            blind = [r["blind_rate_at_90"] for r in records
                     if r["arm"] == arm and r["strain_key"] == strain
                     and r.get("blind_rate_at_90") is not None]
            entry = {
                "n": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(min(values)), "max": float(max(values)),
            }
            if blind:
                entry["blind_mean"] = float(np.mean(blind))
                entry["blind_sd"] = (float(np.std(blind, ddof=1))
                                     if len(blind) > 1 else 0.0)
            per_strain[strain] = entry
        out[arm] = per_strain
    return out


def paired(records: list[dict]) -> dict:
    """Flood fill minus KAM, on the same phantom -- so the pairing is exact."""

    out = {}
    for arm in ("flood fill", "flood fill + merge"):
        gaps, per_strain = [], {}
        for strain in STRAINS:
            local = []
            for rep in REPLICATES:
                a = [r for r in records if r["arm"] == arm
                     and r["strain_key"] == strain and r["replicate"] == rep]
                b = [r for r in records if r["arm"] == "KAM threshold"
                     and r["strain_key"] == strain and r["replicate"] == rep]
                if a and b:
                    gap = a[0]["recovery_rate_at_90"] - b[0]["recovery_rate_at_90"]
                    local.append(gap)
                    gaps.append(gap)
            if local:
                per_strain[strain] = {
                    "n": len(local), "mean": float(np.mean(local)),
                    "sd": float(np.std(local, ddof=1)) if len(local) > 1 else 0.0,
                    "min": float(min(local)),
                }
        entry = {"per_strain": per_strain, "n_pairs": len(gaps)}
        if gaps:
            entry["mean"] = float(np.mean(gaps))
            entry["sd"] = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
            entry["min"] = float(min(gaps))
            entry["all_positive"] = bool(all(g > 0 for g in gaps))
            try:
                from scipy import stats

                t, p = stats.ttest_1samp(gaps, 0.0)
                entry["t"] = float(t)
                entry["p"] = float(p)
                entry["wilcoxon_p"] = float(
                    stats.wilcoxon(gaps).pvalue) if len(gaps) > 5 else None
            except Exception:
                pass
            # Strain is a block.  Retain the replicate-level values above for
            # descriptive reporting, but also summarize the four block means
            # so the result is not represented only by a pooled 20-pair test.
            block_means = [value["mean"] for value in per_strain.values()]
            entry["block_means"] = {
                strain: value["mean"]
                for strain, value in per_strain.items()
            }
            entry["n_blocks"] = len(block_means)
            if block_means:
                entry["block_mean"] = float(np.mean(block_means))
                entry["block_sd"] = (float(np.std(block_means, ddof=1))
                                      if len(block_means) > 1 else 0.0)
            if len(block_means) > 1:
                try:
                    from scipy import stats
                    t_block, p_block = stats.ttest_1samp(block_means, 0.0)
                    entry["block_t"] = float(t_block)
                    entry["block_p"] = float(p_block)
                except Exception:
                    pass
        out[arm] = entry
    return out


def out_of_sample(records: list[dict]) -> dict:
    """The defaults were chosen on r0.  Is r0 flattered relative to r1-r5?"""

    import capped_search as cs

    out = {}
    for arm in ("flood fill", "flood fill + merge"):
        rows = []
        for strain in STRAINS:
            tuned = best(FULL_ROOT / strain, arm)
            frozen_r0 = best(HERE / "runs/frozen" / strain, arm)
            values = [r["recovery_rate_at_90"] for r in records
                      if r["arm"] == arm and r["strain_key"] == strain]
            if not (tuned and frozen_r0 and values):
                continue
            rows.append({
                "strain_key": strain,
                "r0_full_search": tuned["recovery_rate_at_90"],
                "r0_frozen": frozen_r0["recovery_rate_at_90"],
                "replicates_mean": float(np.mean(values)),
                "replicates_sd": (float(np.std(values, ddof=1))
                                  if len(values) > 1 else 0.0),
                "n": len(values),
            })
        out[arm] = rows
    return out


def show(report: dict) -> None:
    ver = report.get("verification") or []
    if ver:
        print("\n=== the reduced KAM grid against the full one, on r0 ===")
        print(f"  {'strain':>7s} {'full 9108':>10s} {'reduced 828':>12s} {'diff':>9s}")
        for row in ver:
            print(f"  {row['strain_key']:>7s} {row['full_grid']:>10.4f} "
                  f"{row['reduced_grid']:>12.4f} {row['difference']:>+9.4f}")

    summary = report.get("summary") or {}
    if summary:
        print("\n=== recovery rate at 0.9, mean +- sd over replicates ===")
        print(f"  {'arm':>19s} " + " ".join(f"{s:>15s}" for s in STRAINS))
        for arm in ARMS:
            cells = []
            for strain in STRAINS:
                e = summary.get(arm, {}).get(strain)
                cells.append(f"{e['mean']:.4f}+-{e['sd']:.4f}" if e else " " * 15)
            print(f"  {arm:>19s} " + " ".join(f"{c:>15s}" for c in cells))

    pair = report.get("paired") or {}
    for arm, entry in pair.items():
        if not entry.get("n_pairs"):
            continue
        print(f"\n=== {arm} minus KAM, paired on the same phantom ===")
        print(f"  n = {entry['n_pairs']} pairs, mean {entry['mean']:+.4f} "
              f"+- {entry['sd']:.4f}, smallest {entry['min']:+.4f}, "
              f"every pair positive: {entry['all_positive']}")
        if entry.get("p") is not None:
            print(f"  paired t = {entry['t']:.1f}, p = {entry['p']:.2e}"
                  + (f", wilcoxon p = {entry['wilcoxon_p']:.2e}"
                     if entry.get("wilcoxon_p") else ""))

    prim = report.get("primary") or {}
    if prim.get("summary"):
        print("\n=== primary phantom, independent realisations ===")
        print(f"  {'arm':>19s} {'n':>3s} {'rate@90':>17s} {'ARI':>17s}")
        for arm, e in prim["summary"].items():
            ari = ("--" if e.get("ari_mean") is None
                   else f"{e['ari_mean']:.4f}+-{e['ari_sd']:.4f}")
            print(f"  {arm:>19s} {e['n']:>3d} "
                  f"{e['mean']:.4f}+-{e['sd']:.4f}  {ari:>17s}")
        for arm, g in (prim.get("paired") or {}).items():
            print(f"  {arm} minus KAM: {g['mean']:+.4f} +- {g['sd']:.4f} "
                  f"(n={g['n']}, smallest {g['min']:+.4f})")

    oos = report.get("out_of_sample") or {}
    for arm, rows in oos.items():
        if not rows:
            continue
        print(f"\n=== {arm}: defaults chosen on r0, applied to r1-r5 ===")
        print(f"  {'strain':>7s} {'r0 search':>10s} {'r0 frozen':>10s} "
              f"{'replicates':>18s}")
        for row in rows:
            print(f"  {row['strain_key']:>7s} {row['r0_full_search']:>10.4f} "
                  f"{row['r0_frozen']:>10.4f} "
                  f"{row['replicates_mean']:>10.4f}+-{row['replicates_sd']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",
                        choices=["verify", "verify-full-replicates", "run",
                                 "primary", "report", "all"])
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    if args.action in ("verify", "all"):
        verify(args.workers)
    if args.action == "verify-full-replicates":
        verify_full_replicates(args.workers)
    if args.action in ("run", "all"):
        run(args.workers)
    if args.action in ("primary", "all"):
        run_primary(args.workers)

    if args.action in ("verify", "verify-full-replicates", "primary",
                       "report", "all"):
        records = collect()
        report = {
            "protocol": {
                "flood_fill": "blind: every axis at capped_search.DEFAULTS, "
                              "chosen on r0; threshold swept, and also fixed",
                "kam": "re-tuned per replicate over 828 configurations",
                "replicates_per_strain": len(REPLICATES),
            },
            "verification": verification(),
            "replicate_grid_verification": replicate_grid_verification(),
            "records": records,
            "summary": summarise(records),
            "paired": paired(records),
            "out_of_sample": out_of_sample(records),
            "primary": primary_block(),
        }
        show(report)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "replicates.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
        print(f"\nwritten: {OUT/'replicates.json'}")


if __name__ == "__main__":
    main()
