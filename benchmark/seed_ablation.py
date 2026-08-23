#!/usr/bin/env python3
"""How much of the spread is the algorithm, and how much is the phantom?

``replicates.py`` varies the phantom and reports a spread, but that spread
confounds two different things: the microstructure changing between
realisations, and the flood fill's own stochasticity -- it grows regions from
seed points in an order that depends on a random seed, and a different order
can land on a different partition of the same volume.

Holding the phantom fixed and sweeping only the seed separates them.  Every
other axis stays at ``capped_search.DEFAULTS``, so this is the same blind
configuration the replicates run, on the r0 phantoms whose replicate spread is
already known.

If the seed spread is the smaller of the two, the replicate error bars are
measuring the microstructure, which is what they are meant to measure.  If it
dominates, they are mostly measuring the algorithm shuffling its own output and
should be reported as such.

The measured answer is that the seed spread is *zero*, which is a strong enough
claim to need its own check -- a metric that never moves is equally consistent
with the seed never reaching the algorithm.  ``determinism`` rules that out
directly.  The seed does reach the RNG and does change the output: the label
*numbering* differs on every seed.  What does not change is the partition those
labels describe, because ``flood_fill_dfxm_two_stage`` collects its seeds
deterministically and then sorts them by size, leaving the RNG nothing to
decide.  Comparing two seeds with a permutation-invariant measure returns
exactly 1, while comparing the raw label arrays returns "different" -- and it is
the permutation-invariant reading that every metric in this study uses.

Usage::

    python seed_ablation.py determinism
    python seed_ablation.py run --workers 14 --seeds 0 1 2 3 4
    python seed_ablation.py report
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
ROOT = HERE / "runs/seeds"
PHANTOM_DIR = HERE / "phantoms"
REPLICATE_JSON = HERE / "analysis" / "replicates_v1" / "replicates.json"

STRAINS = ("2p4", "3p5", "4p6", "6p2")
ARMS = ("flood fill", "flood fill + merge")
STAGES = ("markers", "merge", "finalise")


def run(workers: int, seeds: list[int]) -> None:
    for strain in STRAINS:
        phantom = PHANTOM_DIR / f"strain_{strain}_r0.npz"
        if not phantom.exists():
            print(f"skipping {strain}: no phantom", flush=True)
            continue
        out = ROOT / strain
        for stage in STAGES:
            command = [sys.executable, str(HERE / "capped_search.py"), stage,
                       "--frozen", "--workers", str(workers),
                       "--phantom", str(phantom), "--out", str(out)]
            if stage == "markers":
                command += ["--seeds", *[str(s) for s in seeds]]
            print(f"\n=== seeds {strain}: {stage} ===", flush=True)
            subprocess.run(command, check=True)


def determinism(seeds=(1, 2, 3, None)) -> list[dict]:
    """Does the seed reach the algorithm, and does the partition move?"""

    import disell
    import pipelines
    from sklearn.metrics import adjusted_rand_score

    import capped_search as cs

    rows = []
    for strain in STRAINS:
        phantom = PHANTOM_DIR / f"strain_{strain}_r0.npz"
        if not phantom.exists():
            continue
        with np.load(phantom) as data:
            # The capped benchmark segments the latent field.  ``field`` is
            # the separately blurred measurement field and using it here would
            # test determinism on a different input than the headline runs.
            field = np.ascontiguousarray(data["latent"], dtype=np.float32)
            spacing = tuple(float(v) for v in data["spacing"])
        mask = np.ascontiguousarray(np.ones(field.shape[:3], dtype=np.uint8))
        footprint = np.ascontiguousarray(
            pipelines.isotropic_footprint(spacing, cs.DEFAULTS["footprint_radius_um"]),
            dtype=bool)

        def run_one(seed):
            result, _ = disell.flood_fill_dfxm_two_stage(
                field, footprint=footprint,
                local_misorientation_threshold=float(
                    cs.DEFAULT_LOCAL_THRESHOLD_DEG["flood fill"]),
                global_threshold=float(cs.DEFAULTS["global_threshold_deg"]),
                footprint_tolerance=float(cs.DEFAULTS["footprint_tolerance"]),
                mask=mask.copy(), max_iterations=cs.MAX_SEED_ATTEMPTS,
                min_grain_size=int(cs.DEFAULT_MIN_CELL_SIZE["flood fill"]),
                recycle_small_grains=False,
                stagnation_tolerance=cs.STAGNATION_TOLERANCE,
                random_seed=seed)
            return np.asarray(result["segmentation"], dtype=np.int32).ravel()

        reference = run_one(0)
        for seed in seeds:
            other = run_one(seed)
            rows.append({
                "strain_key": strain,
                "seed": "random_device" if seed is None else seed,
                "ari_against_seed_0": float(adjusted_rand_score(reference, other)),
                "labels_reference": int(len(np.unique(reference)) - 1),
                "labels_other": int(len(np.unique(other)) - 1),
                "label_arrays_identical": bool(np.array_equal(reference, other)),
            })
        print(f"  {strain}: " + ", ".join(
            f"seed {r['seed']} ARI {r['ari_against_seed_0']:.10f}"
            f" (labels identical: {r['label_arrays_identical']})"
            for r in rows if r["strain_key"] == strain), flush=True)
    return rows


def collect() -> list[dict]:
    """Best configuration per seed, per arm -- the seed is not tuned over."""

    import capped_search as cs

    records = []
    for strain in STRAINS:
        path = ROOT / strain / "final.jsonl"
        if not path.exists():
            continue
        rows = [r for r in cs.read_rows(path)
                if r.get("status") == "ok" and not cs._disqualified(r)]
        for arm in ARMS:
            arm_rows = [r for r in rows if (r.get("arm") or "flood fill") == arm]
            for seed in sorted({r.get("seed", 0) for r in arm_rows}):
                subset = [r for r in arm_rows if r.get("seed", 0) == seed]
                if not subset:
                    continue
                winner = min(subset, key=cs.strict_recovery_key)
                records.append({
                    "strain_key": strain, "arm": arm, "seed": int(seed),
                    "recovery_rate_at_90": winner["recovery_rate_at_90"],
                    "ari": winner.get("ari"),
                    "local_threshold_deg": winner.get("local_threshold_deg"),
                })
    return records


def decompose(records: list[dict]) -> dict:
    """Seed spread beside the replicate spread, strain by strain."""

    replicate_sd = {}
    if REPLICATE_JSON.exists():
        payload = json.loads(REPLICATE_JSON.read_text())
        for arm, per_strain in (payload.get("summary") or {}).items():
            for strain, entry in per_strain.items():
                replicate_sd[(arm, strain)] = entry.get("sd")

    out = {}
    for arm in ARMS:
        rows = []
        for strain in STRAINS:
            values = [r["recovery_rate_at_90"] for r in records
                      if r["arm"] == arm and r["strain_key"] == strain]
            if len(values) < 2:
                continue
            seed_sd = float(np.std(values, ddof=1))
            rep_sd = replicate_sd.get((arm, strain))
            rows.append({
                "strain_key": strain,
                "n_seeds": len(values),
                "seed_mean": float(np.mean(values)),
                "seed_sd": seed_sd,
                "seed_range": float(max(values) - min(values)),
                "replicate_sd": rep_sd,
                "ratio_seed_to_replicate": (
                    float(seed_sd / rep_sd) if rep_sd else None),
            })
        out[arm] = rows
    return out


def show(report: dict) -> None:
    for arm, rows in (report.get("decomposition") or {}).items():
        if not rows:
            continue
        print(f"\n=== {arm}: seed spread vs replicate spread ===")
        print(f"  {'strain':>7s} {'seeds':>6s} {'mean':>8s} {'seed sd':>9s} "
              f"{'range':>8s} {'replicate sd':>13s} {'ratio':>7s}")
        for row in rows:
            rep = ("--" if row["replicate_sd"] is None
                   else f"{row['replicate_sd']:.4f}")
            ratio = ("--" if row["ratio_seed_to_replicate"] is None
                     else f"{row['ratio_seed_to_replicate']:.2f}")
            print(f"  {row['strain_key']:>7s} {row['n_seeds']:>6d} "
                  f"{row['seed_mean']:>8.4f} {row['seed_sd']:>9.4f} "
                  f"{row['seed_range']:>8.4f} {rep:>13s} {ratio:>7s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",
                        choices=["run", "report", "determinism", "all"])
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    if args.action in ("run", "all"):
        run(args.workers, args.seeds)

    checks = []
    if args.action in ("determinism", "all"):
        print("=== does the seed reach the algorithm? ===")
        checks = determinism()

    if args.action in ("report", "determinism", "all"):
        records = collect()
        report = {"records": records, "decomposition": decompose(records)}
        if checks:
            report["determinism"] = checks
        elif (OUT / "seed_ablation.json").exists():
            previous = json.loads((OUT / "seed_ablation.json").read_text())
            if previous.get("determinism"):
                report["determinism"] = previous["determinism"]
        show(report)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "seed_ablation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
        print(f"\nwritten: {OUT/'seed_ablation.json'}")


if __name__ == "__main__":
    main()
