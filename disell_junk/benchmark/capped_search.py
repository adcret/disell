#!/usr/bin/env python3
"""Systematic search for both segmentation arms under the 1.5 um radius cap.

Why this supersedes the earlier Sobol/LHS oracle search
-------------------------------------------------------
Capping both the flood-fill neighbourhood and the KAM kernel at 1.5 um makes
the radius axes *discrete*: at (1.0, 0.4, 0.4) um spacing only 13 distinct
footprints exist below the cap (5 to 87 voxels).  ``footprint_tolerance`` is
likewise discrete -- it maps onto an integer neighbour requirement.  Sampling
those axes continuously, as the old search did, wastes most of its budget
re-evaluating identical footprints.  This search enumerates them exhaustively
instead and spends the budget on the genuinely continuous axes.

Staging
-------
``markers``   flood-fill marker search with the KAM radius pinned.  Scored on
              cell count, ARI and VI -- the cheap metrics -- because the
              selection policy is count-first and identity only breaks ties.
``kam``       for the best marker configurations, sweep all 13 KAM radii.  The
              KAM radius moves boundaries, not cell counts.
``merge``     for the best *small-min_cell_size* configurations, sweep the
              orientation-gated merge (see ``merge_cells``).
``baseline``  the KAM-threshold arm, searched over its own parameters so the
              comparison is between two arms at their own optima.
``finalise``  full scoring, including identity F1 and boundary ASSD, of every
              stage's finalists, then the ranked report.

Every stage appends to a JSONL store keyed by a configuration hash and skips
work already present, so a run can be interrupted and resumed.

Usage::

    python capped_search.py markers  --workers 14
    python capped_search.py kam      --workers 14
    python capped_search.py merge    --workers 14
    python capped_search.py baseline --workers 14
    python capped_search.py finalise --workers 14
    python capped_search.py report
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MAX_RADIUS_UM = 2.0
MIN_RADIUS_UM = 0.9
WATERSHED_CONNECTIVITY = 1
MAX_SEED_ATTEMPTS = 700_000
STAGNATION_TOLERANCE = 2_000

# --------------------------------------------------------------- search grids

#: Fraction of the neighbourhood that must agree.  Converted to the integer
#: neighbour requirement per footprint and de-duplicated, so a small footprint
#: contributes fewer distinct tolerances than a large one.
TOLERANCE_FRACTIONS = (0.04, 0.10, 0.20, 0.30, 0.45, 0.60)

#: Log-spaced.  The capped neighbourhood pushes the optimum well above the
#: unconstrained winner's 0.0068, so the grid is centred higher.
LOCAL_THRESHOLDS_DEG = tuple(np.round(np.logspace(np.log10(0.003), np.log10(0.06), 14), 6))

#: -1 disables the global threshold.
GLOBAL_THRESHOLDS_DEG = (-1.0, 0.30, 0.873, 2.0)

# Deliberately small.  A large min_cell_size buys the cell count by deleting
# genuinely small cells, and with a merge step downstream that trade is not
# worth making: it is better to keep every fragment and let the merge fold the
# spurious ones into their neighbours.  The largest values from the earlier
# count-first grid (160-400) are gone because they exist only to hit a count.
# The small values are the ones of interest: with a merge step downstream it is
# better to keep every fragment than to buy the cell count by deleting real
# cells.  80 and 160 are retained only so the *unmerged* arm has a fair chance
# in the comparison -- without them every no-merge configuration would be
# disqualified for over-segmentation by construction, which would rig the
# arm-versus-arm result rather than measure it.
# Scaled to the primary phantom: 98,304 um^3 over ~2,525 cells is ~243 voxels
# per mean cell, so these run from ~1 % to ~50 % of one.  An earlier grid ran
# up to 400, which is larger than a whole cell at this size and would delete
# almost everything.  80 and 120 exist only to give the *unmerged* arm a
# fair chance; the small values are the ones of interest.
MIN_CELL_SIZES = (3, 5, 10, 20, 40, 80, 120)

# The merge now does the work that min_cell_size used to, so its grid is the
# one that needs resolution.
# Up to about two mean cells: beyond that a merge is fusing real cells rather
# than reclaiming fragments.
MERGE_SIZE_VOXELS = (20, 40, 80, 140, 220, 350, 500)
MERGE_THRESHOLDS_DEG = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40)

#: KAM-threshold arm.
BASELINE_PERCENTILES = tuple(np.round(np.arange(20.0, 76.0, 2.5), 2))
BASELINE_MIN_SIZES = (3, 5, 10, 20, 40, 80)
BASELINE_CONNECTIVITIES = (1, 2, 3)

#: The KAM arm's grid for replicate runs, deliberately *asymmetric*.
#:
#: On the replicates the flood-fill arm runs blind -- every axis pinned to the
#: defaults chosen on r0, one threshold swept -- while the KAM arm is re-tuned
#: on each replicate over this grid.  The asymmetry is the point: it loads the
#: comparison against the conclusion, so a flood-fill win survives the
#: objection that it simply received more tuning effort.
#:
#: The percentile axis is kept whole because it is the KAM arm's real knob.
#: The other three are cut to a neighbourhood of where the full 9,108-point
#: grid put the optimum at every strain on r0 -- radius 1.08 um, min size 3,
#: connectivity 1-2 -- with room on both sides.  ``replicates.py verify``
#: checks on r0 that this reduction reaches the full grid's answer.
BASELINE_REPLICATE_RADII_UM = (1.0, 1.08, 1.15, 1.2)
BASELINE_REPLICATE_MIN_SIZES = (3, 5, 10)
BASELINE_REPLICATE_CONNECTIVITIES = (1, 2, 3)

#: The KAM radius pinned during the marker stage; refined later in the kam stage.
PINNED_KAM_RADIUS_UM = 1.2

FINALIST_COUNT = 250

# ------------------------------------------------------------------- defaults
#
# Measured on all four strain phantoms (2026-08-20): freeze one axis at a
# single global value, re-tune every other axis per strain, and record what it
# costs in recovery rate at tau = 0.9.  Only ``local_threshold_deg`` is worth
# a sweep -- every other axis costs at most 0.006, and three cost nothing at
# all.  The measurement lives in ``parameter_economy.py``; the numbers below
# are the mean cost over the four strains, for the flood-fill arm.
#
#     footprint_tolerance    0.0000   0.04 wins at every strain
#     global_threshold_deg   0.0000   inert: both grid values tie everywhere
#     merge_size_voxels      0.0000
#     kam_radius_um          0.0008   22 values over 0.9-2.0 um are equivalent
#     merge_threshold_deg    0.0009
#     min_cell_size          0.0029   set by whether a merge runs, not by strain
#     footprint_radius_um    0.0063   flat plateau; see the cliff note below
#     local_threshold_deg    0.0089   the one real knob
#
# Two cliffs bound the plateau and must not be crossed:  a footprint radius of
# 0.9 um costs 0.64 -- the effective floor is 1.08 um, which is *inside* the
# searched range, not below it -- and a min_cell_size of 40 or more costs 0.15
# at 6.2 % strain and 0.59 at 80.
#
#: Everything a caller should not have to think about.  ``min_cell_size`` is
#: not here because it follows from whether a merge step runs: see
#: ``DEFAULT_MIN_CELL_SIZE``.
DEFAULTS = {
    "footprint_radius_um": 1.15,
    "footprint_tolerance": 0.04,
    "global_threshold_deg": -1.0,
    "kam_radius_um": PINNED_KAM_RADIUS_UM,
    "merge_size_voxels": 20,
    "merge_threshold_deg": 0.05,
}

#: Keep every fragment when a merge will tidy up, otherwise drop the smallest.
DEFAULT_MIN_CELL_SIZE = {"flood fill": 20, "flood fill + merge": 3}

#: Where the local threshold lands when it is not swept at all.  Sweeping it
#: is much the better choice -- with the rest of ``DEFAULTS`` frozen, a sweep
#: costs 0.007 against the full search while a fixed value costs 0.016-0.019 --
#: so these are a fallback, not a recommendation.
#:
#: **These two values predate the retirement of the unmerged surplus rule and
#: are kept only because the section 7 blind replicates were run on them.**
#: The gap between the arms was the rule's doing: with no merge downstream,
#: every over-segmenting threshold was disqualified before the merge could act,
#: which forced the unmerged arm high.  On the refined grids, scored after the
#: merge, both arms want the *same* value and one global value serves all four
#: strains for 0.0010 mean recovery rate (0.0040 worst) -- see BENCHMARK
#: section 5b and ``refine.py merged``.  Re-deriving these defaults means
#: re-running the blind replicates, since changing them silently would leave
#: section 7 reporting a protocol it no longer used.
DEFAULT_LOCAL_THRESHOLD_DEG = {"flood fill": 0.015055,
                               "flood fill + merge": 0.009495}

#: What the refined, post-merge search actually recommends.  Not yet wired into
#: the replicate protocol, for the reason above.
REFINED_GLOBAL_LOCAL_THRESHOLD_DEG = 0.010483


def neighbour_requirements(footprint_tolerance: float, n_neighbours: int) -> np.ndarray:
    """``ceil(tolerance * n)`` for every ``n`` from 1 to ``n_neighbours``.

    This vector is what the algorithm actually sees.  The full-footprint entry
    is the headline number, but the smaller counts are reached constantly:
    the C++ driver clears the mask over every region it accepts, so a voxel
    growing alongside an established cell is tested against a reduced count.

    Inlined from the retired ``oracle_core`` so this study carries no dependency
    on the superseded benchmark.
    """

    n = np.arange(1, int(n_neighbours) + 1, dtype=np.float64)
    return np.ceil(float(footprint_tolerance) * n - 1e-9).astype(np.int64)


# ------------------------------------------------------------ worker workspace

WORK = None


class Workspace:
    """Phantom plus a bounded KAM cache, loaded once per worker process."""

    def __init__(self, phantom_path: Path):
        with np.load(phantom_path) as data:
            self.labels = np.ascontiguousarray(data["labels"])
            self.field = np.ascontiguousarray(data["latent"])
            self.spacing = tuple(float(v) for v in data["spacing"])
        self.mask = self.labels > 0
        self.n_truth = int(self.labels.max())
        self._kam: dict = {}
        self._footprint: dict = {}

    def footprint(self, radius_um: float):
        import pipelines

        key = round(float(radius_um), 4)
        if key not in self._footprint:
            self._footprint[key] = pipelines.isotropic_footprint(self.spacing, key)
        return self._footprint[key]

    def kam(self, radius_um: float):
        import pipelines

        key = round(float(radius_um), 4)
        if key not in self._kam:
            # Must exceed the number of radius classes a single task sweeps
            # (22), or the kam stage evicts the field it is about to need again
            # and recomputes all of them for every task.  Each field is ~2.4 MB,
            # so 26 costs ~60 MB per worker.
            while len(self._kam) >= 26:
                self._kam.pop(next(iter(self._kam)))
            self._kam[key] = pipelines.masked_kam(
                self.field, self.mask, self.footprint(key)
            )
        return self._kam[key]


def init_worker(phantom_path: str):
    global WORK
    WORK = Workspace(Path(phantom_path))


# -------------------------------------------------------------------- scoring


#: The recovery criterion the study optimises: a true cell counts as recovered
#: only when a predicted cell holds at least this fraction of it and is at
#: least this pure.  Above 0.5 the pairing is automatically one-to-one, which
#: is what makes it cheap enough to score every configuration.
STRICT_TAU = 0.9


def score_cheap(truth, prediction) -> dict:
    """Counts, ARI, VI and strict recovery.  Skips only boundary and identity.

    Strict recovery is included here rather than in the expensive pass because
    at tau > 0.5 it needs no assignment problem -- see ``strict_recovery``.
    """

    from sklearn.metrics import adjusted_rand_score

    import bench_metrics
    import strict_recovery as sr

    total, split, merge = bench_metrics.variation_of_information(truth, prediction)
    n_pred = int(np.unique(prediction).size)
    n_truth = int(np.unique(truth).size)
    strict = sr.strict_recovery(truth, prediction, tau=STRICT_TAU)
    return {
        "recovered_at_90": strict["recovered_at_tau"],
        "recovery_rate_at_90": strict["recovery_rate_at_tau"],
        "recovered_purity_only_at_90": strict["recovered_purity_only"],
        "fused_true_cells": strict["fused_true_cells"],
        "strict_split_true_cells": strict["split_true_cells"],
        "fused_true_cell_fraction_at_90": strict["fused_true_cell_fraction"],
        # How far the partition is from ground truth in voxels, not in counts.
        # Zero for pure over-segmentation; grows with how badly cells are fused.
        "contamination": strict["contamination"],
        "contaminated_voxels": strict["contaminated_voxels"],
        "fused_contamination": strict["fused_contamination"],
        "fused_predicted_cells": strict["fused_predicted_cells"],
        "n_cells_pred": n_pred,
        "n_cells_true": n_truth,
        "cell_count_error": n_pred - n_truth,
        "absolute_count_error": abs(n_pred - n_truth),
        "ari": float(adjusted_rand_score(truth.ravel(), prediction.ravel())),
        "vi_total_bits": total,
        "vi_split_bits": split,
        "vi_merge_bits": merge,
    }


def score_full(truth, prediction, field, spacing) -> dict:
    """Everything the selection policy needs, including identity F1."""

    import bench_metrics
    import object_orientation_metrics as oom

    out = score_cheap(truth, prediction)
    out["boundary_assd_um"] = bench_metrics.boundary_assd_um(truth, prediction, spacing)
    _, objects = oom.match_cells(
        truth, prediction, field, spacing,
        purity_threshold=0.6, completeness_threshold=0.6,
    )
    n_true = int(objects["n_true_cells"])
    n_pred = int(objects["n_predicted_cells"])
    recovered = int(objects["one_to_one_recovered_cells"])
    correct = int(objects["orientation_correct_cells_at_0p02deg"])
    denominator = max(n_true + n_pred, 1)
    out.update({
        "identity_f1": 2 * recovered / denominator,
        "identity_precision": recovered / max(n_pred, 1),
        "identity_recall": recovered / max(n_true, 1),
        "recovered_cells": recovered,
        "orientation_f1_0p02": 2 * correct / denominator,
        "split_true_cells": int(objects["split_true_cells"]),
        "merged_predicted_cells": int(objects["merged_predicted_cells"]),
        "median_orientation_error_deg": objects[
            "median_matched_mean_orientation_error_deg"
        ],
        # Fusion is the error that no downstream merge can undo, so it is
        # tracked separately from fragmentation, which merging does fix.
        "fused_true_cell_fraction": int(objects["merged_predicted_cells"]) / max(n_true, 1),
        "split_true_cell_fraction": int(objects["split_true_cells"]) / max(n_true, 1),
    })
    return out


# ------------------------------------------------------------------ policies

#: The original policy: hit the true cell count, then break ties on identity.
#: Correct when the partition is the final answer.
COUNT_FIRST = "count_first_identity_v1"

#: The policy for a pipeline that merges afterwards.  Fragmentation is
#: recoverable -- the merge step folds fragments into their closest neighbour --
#: but fusion is not: two true cells sharing one predicted cell can never be
#: separated again.  So this ranks on under-segmentation first and treats an
#: excess of cells as acceptable.  Cell count becomes a diagnostic, not a
#: selector.
FUSION_AVERSE = "fusion_averse_v1"


def count_first_key(row: dict):
    return (
        row["absolute_count_error"],
        -(row.get("identity_f1") or 0.0),
        -(row.get("orientation_f1_0p02") or 0.0),
        -(row.get("ari") or 0.0),
        row.get("vi_total_bits") or np.inf,
        row.get("boundary_assd_um") or np.inf,
    )


#: How far above the true cell count a candidate may sit and still be
#: considered recoverable.  Empirically the merge step roughly halves a cell
#: count, so beyond about four times the truth there is no realistic route
#: back.  This guard is what stops the fusion-averse policy from being won by
#: shattering: minimising fusion alone is trivially maximised by a partition
#: that fuses nothing because it splits everything.
OVERSEGMENTATION_LIMIT = 4.0


def fusion_averse_key(row: dict):
    """Rank by how many true cells came out *right*, not by how many came out.

    ``identity_recall`` is the fraction of true cells recovered one-to-one at
    60 % overlap in both directions.  Fragmenting a cell fails the completeness
    half of that test and fusing two fails the purity half, so recall cannot be
    gained by either kind of damage -- while it stays indifferent to spare
    fragments, which the merge step removes.  Fusion breaks ties, because it is
    the damage nothing downstream can undo.
    """

    n_true = row.get("n_cells_true") or 0
    n_pred = row.get("n_cells_pred") or 0
    beyond = n_pred > OVERSEGMENTATION_LIMIT * max(n_true, 1)
    return (
        bool(beyond),                       # unrecoverable shattering, ranked last
        -(row.get("identity_recall") or 0.0),
        row.get("vi_merge_bits") if row.get("vi_merge_bits") is not None else np.inf,
        -(row.get("ari") or 0.0),
        row.get("boundary_assd_um") or np.inf,
    )


def cheap_fusion_key(row: dict):
    """Rank finished partitions by the headline objective."""

    return strict_recovery_key(row)


def merge_candidate_key(row: dict):
    """Rank *pre-merge* partitions as inputs to the merge stage.

    These are deliberately over-segmented, so the headline policy's
    "a surplus requires a merge step" rule would disqualify every one of them
    -- the merge step is exactly what is about to be applied.  What matters in
    a candidate is that it carries little fusion (which merging cannot fix) and
    already resolves many cells cleanly; the surplus is the merge stage's
    problem.  Only gross shattering is excluded, because beyond a few times the
    true count there is no realistic route back.
    """

    n_true = max(row.get("n_cells_true") or 0, 1)
    n_pred = row.get("n_cells_pred") or 0
    return (
        n_pred > OVERSEGMENTATION_LIMIT * n_true,
        -(row.get("recovered_at_90") or 0),
        row.get("contamination")
        if row.get("contamination") is not None else np.inf,
        -(row.get("ari") or 0.0),
    )


#: How far above the true count the **final** partition may sit and still count
#: as "mild".  Over-segmentation is acceptable only because a merge step
#: follows, so the limit is applied to what the merge produces, not to the raw
#: marker stage.
MILD_OVERSEGMENTATION = 1.5

#: Retired 2026-08-20.  This required an *unmerged* partition to sit within
#: 1.05x the true count, on the reasoning that a surplus is only acceptable if
#: something downstream removes it.  That judged an intermediate product by a
#: final-product standard, and it did not merely filter the unmerged arm -- it
#: shaped it, forcing the arm's threshold high because every over-segmenting
#: value was disqualified before the merge could act (see
#: ``DEFAULT_LOCAL_THRESHOLD_DEG``).  Measured cost of the rule at the time it
#: was dropped: 0.0008-0.0109 recovery rate on the unmerged arm, and **exactly
#: zero on the merged arm** on all five phantoms, the merge already bringing
#: the count to 1.00-1.18x true.  The flood fill is assessed after merging, so
#: nothing the study reports depends on this.  Kept as a named constant only so
#: the retired rule stays legible in ``POLICIES`` provenance.
RETIRED_UNMERGED_SURPLUS_TOLERANCE = 1.05

#: The study's headline policy: fewest fused cells first, then the most true
#: cells recovered almost exactly (>= 90 % pure and >= 90 % complete).  Fusion
#: leads because it is the damage nothing downstream can undo -- a merge step
#: folds fragments back together, but no step separates two true cells that
#: share one predicted cell.  Cell count is a constraint, not a selector.
STRICT_RECOVERY = "fused_first_recovery_v2"


def _disqualified(row: dict) -> bool:
    """True when the partition is more than mildly over-segmented.

    One way to fail.  Shattering has to stay out -- a partition of one voxel
    per cell recovers nothing but costs no contamination, so the count is what
    keeps the policy honest -- but the limit belongs on the *final* partition
    and nowhere else.  The retired second test applied it to unmerged output
    too; see ``RETIRED_UNMERGED_SURPLUS_TOLERANCE``.
    """

    n_true = max(row.get("n_cells_true") or 0, 1)
    n_pred = row.get("n_cells_pred") or 0
    return n_pred > MILD_OVERSEGMENTATION * n_true


def strict_recovery_key(row: dict):
    """Most correct cells first; then how close the rest are to ground truth.

    Contamination is the second key rather than a count of fused cells, because
    counting cannot tell a predicted cell that straddles two true cells 50/50
    from one that overreaches by 5 %.  It also encodes the preference between
    the two failure modes directly: a cell that is merely split contributes
    nothing to it, while a fused one contributes in proportion to the voxels
    that sit in the wrong cell.
    """

    return (
        _disqualified(row),
        -(row.get("recovered_at_90") or 0),
        row.get("contamination")
        if row.get("contamination") is not None else np.inf,
        row.get("fused_true_cells")
        if row.get("fused_true_cells") is not None else np.inf,
        -(row.get("ari") or 0.0),
        row.get("boundary_assd_um") or np.inf,
    )


POLICIES = {
    STRICT_RECOVERY: strict_recovery_key,
    COUNT_FIRST: count_first_key,
    FUSION_AVERSE: fusion_averse_key,
}


# ------------------------------------------------------------------- segmenting


def flood_markers(work: Workspace, params: dict):
    """Flood-fill markers only.  Independent of the KAM radius."""

    import disell

    footprint = work.footprint(params["footprint_radius_um"])
    result, _ = disell.flood_fill_dfxm_two_stage(
        np.ascontiguousarray(work.field, dtype=np.float32),
        footprint=np.ascontiguousarray(footprint, dtype=bool),
        local_misorientation_threshold=float(params["local_threshold_deg"]),
        global_threshold=float(params["global_threshold_deg"]),
        footprint_tolerance=float(params["footprint_tolerance"]),
        mask=np.ascontiguousarray(work.mask.astype(np.uint8)),
        max_iterations=MAX_SEED_ATTEMPTS,
        min_grain_size=int(params["min_cell_size"]),
        recycle_small_grains=False,
        stagnation_tolerance=STAGNATION_TOLERANCE,
        random_seed=int(params.get("seed", 0)),
    )
    return np.asarray(result["segmentation"], dtype=np.int32)


def watershed(work: Workspace, markers, kam_radius_um: float):
    import disell
    import pipelines

    kam = work.kam(kam_radius_um)
    labels = disell.region_grow_watershed(
        markers, work.mask, pipelines.watershed_elevation(kam, work.mask),
        connectivity=WATERSHED_CONNECTIVITY,
    )
    return np.asarray(labels, dtype=np.int32)


# ------------------------------------------------------------------- the tasks


def config_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(canonical.encode()).hexdigest()


def task_markers(params: dict) -> dict:
    started = time.perf_counter()
    row = {"stage": "markers", **params}
    try:
        markers = flood_markers(WORK, params)
        row["accepted_markers"] = int(markers.max())
        if int(markers.max()) == 0:
            row.update({"status": "invalid", "error": "NoAcceptedMarkers"})
            return row
        labels = watershed(WORK, markers, params["kam_radius_um"])
        row.update(score_cheap(WORK.labels, labels))
        row["status"] = "ok"
    except Exception as error:                       # noqa: BLE001 - recorded, not raised
        row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
    row["runtime_seconds"] = time.perf_counter() - started
    return row


def task_kam(params: dict) -> dict:
    """One marker configuration scored across every KAM radius class."""

    started = time.perf_counter()
    rows = []
    try:
        markers = flood_markers(WORK, params)
        if int(markers.max()) == 0:
            return {"stage": "kam", **params, "status": "invalid",
                    "error": "NoAcceptedMarkers", "rows": []}
        for radius, _voxels in params["kam_grid"]:
            labels = watershed(WORK, markers, radius)
            row = {k: v for k, v in params.items() if k != "kam_grid"}
            row["kam_radius_um"] = radius
            row.update(score_cheap(WORK.labels, labels))
            row["boundary_assd_um"] = __import__("bench_metrics").boundary_assd_um(
                WORK.labels, labels, WORK.spacing
            )
            row["status"] = "ok"
            rows.append(row)
    except Exception as error:                       # noqa: BLE001
        return {"stage": "kam", **{k: v for k, v in params.items() if k != "kam_grid"},
                "status": "error", "error": f"{type(error).__name__}: {error}", "rows": []}
    return {"stage": "kam", "rows": rows, "status": "ok",
            "runtime_seconds": time.perf_counter() - started}


def task_merge(params: dict) -> dict:
    """One marker configuration scored across the merge grid."""

    import merge_cells

    started = time.perf_counter()
    rows = []
    try:
        markers = flood_markers(WORK, params)
        if int(markers.max()) == 0:
            return {"stage": "merge", "rows": [], "status": "invalid"}
        labels = watershed(WORK, markers, params["kam_radius_um"])
        for size in params["merge_sizes"]:
            for threshold in params["merge_thresholds"]:
                merged, diagnostics = merge_cells.merge_small_cells(
                    labels, WORK.field, merge_size_voxels=int(size),
                    merge_threshold_deg=float(threshold), return_diagnostics=True,
                )
                row = {k: v for k, v in params.items()
                       if k not in ("merge_sizes", "merge_thresholds")}
                row["merge_size_voxels"] = int(size)
                row["merge_threshold_deg"] = float(threshold)
                row["merges"] = diagnostics["merges"]
                row["blocked_by_threshold"] = diagnostics["blocked_by_threshold"]
                row.update(score_cheap(WORK.labels, merged))
                row["status"] = "ok"
                rows.append(row)
    except Exception as error:                       # noqa: BLE001
        return {"stage": "merge", "rows": [], "status": "error",
                "error": f"{type(error).__name__}: {error}"}
    return {"stage": "merge", "rows": rows, "status": "ok",
            "runtime_seconds": time.perf_counter() - started}


def task_baseline(params: dict) -> dict:
    started = time.perf_counter()
    row = {"stage": "baseline", **params}
    try:
        import pipelines

        labels, markers, threshold = pipelines.run_kam_threshold(
            WORK.kam(params["kam_radius_um"]), WORK.mask,
            percentile=float(params["percentile"]),
            min_cell_size=int(params["min_cell_size"]),
            connectivity=int(params["connectivity"]),
            watershed_connectivity=WATERSHED_CONNECTIVITY,
        )
        row["kam_threshold_deg"] = float(threshold)
        row["accepted_markers"] = int(markers.max())
        row.update(score_cheap(WORK.labels, np.asarray(labels, dtype=np.int32)))
        row["status"] = "ok"
    except Exception as error:                       # noqa: BLE001
        row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
    row["runtime_seconds"] = time.perf_counter() - started
    return row


def task_finalise(params: dict) -> dict:
    """Full scoring, including identity F1 and boundary ASSD."""

    started = time.perf_counter()
    row = {"stage": "final", **params}
    try:
        if params["arm"] == "KAM threshold":
            import pipelines

            labels, _, threshold = pipelines.run_kam_threshold(
                WORK.kam(params["kam_radius_um"]), WORK.mask,
                percentile=float(params["percentile"]),
                min_cell_size=int(params["min_cell_size"]),
                connectivity=int(params["connectivity"]),
                watershed_connectivity=WATERSHED_CONNECTIVITY,
            )
            labels = np.asarray(labels, dtype=np.int32)
            row["kam_threshold_deg"] = float(threshold)
        else:
            markers = flood_markers(WORK, params)
            if int(markers.max()) == 0:
                row.update({"status": "invalid", "error": "NoAcceptedMarkers"})
                return row
            labels = watershed(WORK, markers, params["kam_radius_um"])
            if int(params.get("merge_size_voxels", 0)) > 0:
                import merge_cells

                labels, diagnostics = merge_cells.merge_small_cells(
                    labels, WORK.field,
                    merge_size_voxels=int(params["merge_size_voxels"]),
                    merge_threshold_deg=float(params["merge_threshold_deg"]),
                    return_diagnostics=True,
                )
                row["merges"] = diagnostics["merges"]
        row.update(score_full(WORK.labels, labels, WORK.field, WORK.spacing))
        row["status"] = "ok"
    except Exception as error:                       # noqa: BLE001
        row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
    row["runtime_seconds"] = time.perf_counter() - started
    return row


# --------------------------------------------------------------------- storage


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"partial final line in {path}; fix or delete it")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_plain) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, tuple)):
        return list(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(f"not JSON serialisable: {type(value)}")


# ------------------------------------------------------------------ execution


class StoreBusy(RuntimeError):
    """Raised when a store is already being written by another process."""


@contextlib.contextmanager
def exclusive_store(store: Path):
    """Hold an exclusive lock on ``store`` for the duration of a stage.

    A resumable store is only safe to resume *sequentially*: the skip set is
    read once at the start, so two processes on the same store both see the
    same work outstanding and both do all of it.  That happened -- the refined
    KAM sweep and the merge zoom each ran twice by accident, which cost nothing
    scientifically (every duplicated row is identical, the pipelines being
    deterministic) but doubled the peak memory and ended in the OOM killer.

    The lock is advisory and released when the process exits for any reason,
    including a kill, so a crashed run leaves nothing to clean up by hand.
    """

    store.parent.mkdir(parents=True, exist_ok=True)
    path = store.with_suffix(store.suffix + ".lock")
    # Append mode, not write: opening for writing would truncate the holder's
    # pid before the lock attempt fails, leaving nothing to name in the error.
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.seek(0)
            held = handle.read().strip() or "another process"
            raise StoreBusy(
                f"{store.name} is already being written by {held}; "
                f"wait for it to finish or kill it before resuming"
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid {os.getpid()}\n")
        handle.flush()
        yield
    finally:
        handle.close()


#: How many tasks may be queued in the pool beyond the running ones.  Submitting
#: a whole grid up front costs memory twice over: every queued task is pickled
#: immediately, and every completed ``Future`` holds its result alive until the
#: submitting list is dropped -- which for a resumable sweep is the whole run.
#: A small backlog per worker keeps them fed without either cost.
QUEUE_DEPTH = 4


def as_they_complete(pool, worker, tasks, queue_depth: int = QUEUE_DEPTH):
    """Yield results with at most ``queue_depth`` tasks in flight per worker.

    Futures are dropped as soon as their result is handed out, so peak memory
    is set by the pool rather than by the length of the grid.
    """

    from concurrent.futures import FIRST_COMPLETED, wait

    pending = set()
    stream = iter(tasks)
    limit = max(1, queue_depth)
    while True:
        while len(pending) < limit:
            task = next(stream, None)
            if task is None:
                break
            pending.add(pool.submit(worker, task))
        if not pending:
            return
        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in finished:
            yield future.result()


def run_pool(tasks, worker, store: Path, phantom_path: Path, workers: int,
             label: str, expand: bool = False) -> None:
    """Run ``tasks`` across a process pool, appending results as they land."""

    from concurrent.futures import ProcessPoolExecutor

    if not tasks:
        print(f"{label}: nothing to do")
        return
    started = time.perf_counter()
    done = 0
    buffer: list[dict] = []
    with exclusive_store(store), ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(str(phantom_path),)
    ) as pool:
        for payload in as_they_complete(pool, worker, tasks,
                                        QUEUE_DEPTH * workers):
            buffer.extend(payload.get("rows", []) if expand else [payload])
            done += 1
            if len(buffer) >= 200:
                append_rows(store, buffer)
                buffer = []
            if done % max(1, len(tasks) // 40) == 0 or done == len(tasks):
                elapsed = time.perf_counter() - started
                rate = done / max(elapsed, 1e-9)
                remaining = (len(tasks) - done) / max(rate, 1e-9)
                print(f"  {label}: {done}/{len(tasks)} tasks  "
                      f"{elapsed/60:.1f} min elapsed, {remaining/60:.1f} min left",
                      flush=True)
        if buffer:
            append_rows(store, buffer)
            buffer = []


#: Coarser axes for the strain series, whose phantoms carry 4-7x more cells and
#: so cost more per evaluation.  The footprint classes stay exhaustive -- they
#: are the axis the study is about -- and only the tolerance and global
#: threshold are thinned.
REDUCED_TOLERANCE_FRACTIONS = (0.04, 0.20, 0.45)
REDUCED_GLOBAL_THRESHOLDS_DEG = (-1.0, 0.873)
#: The strain study measures a *trend* in the optimum, not a single optimum, so
#: the continuous local-threshold axis can be sampled more coarsely without
#: affecting the conclusion.  The footprint enumeration stays exhaustive
#: because that is the axis the study is about.
REDUCED_LOCAL_THRESHOLDS_DEG = tuple(
    np.round(np.logspace(np.log10(0.003), np.log10(0.06), 9), 6))

#: Upper radius for the strain series.  On the primary phantom the best
#: achievable recovery peaks at 1.15-1.35 um (2,010 of 2,534 cells) and then
#: declines monotonically -- 1.79 um reaches only 1,964 -- while costing 2-4x
#: more per evaluation.  Nothing above 1.6 um was ever competitive, so the
#: strain runs stop there.  The primary search keeps the full range on record
#: as the evidence for this cut.
REDUCED_MAX_RADIUS_UM = 1.6


def frozen_marker_grid(spacing, seeds=(0,)) -> list[dict]:
    """Every axis at its default, sweeping only the local threshold.

    This is the claim of ``DEFAULTS`` made falsifiable.  The one-at-a-time
    measurement cannot establish it on its own: freezing each axis separately
    says nothing about freezing them together, and the staged search never
    evaluates the frozen combination -- the finalist funnel keeps only what
    leads at each stage, so the combination is simply absent from
    ``final.jsonl``.  Running it is the only honest way to price it.

    Both arms are emitted, since ``min_cell_size`` follows from the arm.

    ``seeds`` sweeps the flood fill's seed ordering at otherwise identical
    parameters.  It is off by default because it is an ablation rather than a
    search: it separates the algorithm's own stochasticity from the variation
    between phantoms, which a replicate study on its own confounds.
    """

    import phantom_lab as lab
    classes = lab.footprint_classes(spacing, MAX_RADIUS_UM,
                                    min_radius_um=MIN_RADIUS_UM)
    radius, voxels = min(
        classes, key=lambda c: abs(c[0] - DEFAULTS["footprint_radius_um"]))
    tolerance = DEFAULTS["footprint_tolerance"]
    requirement = int(neighbour_requirements(float(tolerance), voxels)[-1])
    return [{
        "footprint_radius_um": radius,
        "footprint_voxels": voxels,
        "footprint_tolerance": float(tolerance),
        "neighbour_requirement": requirement,
        "local_threshold_deg": float(local),
        "global_threshold_deg": float(DEFAULTS["global_threshold_deg"]),
        "min_cell_size": int(min_cell_size),
        "kam_radius_um": DEFAULTS["kam_radius_um"],
        "seed": int(seed),
    } for local in LOCAL_THRESHOLDS_DEG
      for min_cell_size in sorted(set(DEFAULT_MIN_CELL_SIZE.values()))
      for seed in seeds]


def marker_grid(spacing, reduced: bool = False) -> list[dict]:
    import phantom_lab as lab

    tolerances = REDUCED_TOLERANCE_FRACTIONS if reduced else TOLERANCE_FRACTIONS
    globals_deg = REDUCED_GLOBAL_THRESHOLDS_DEG if reduced else GLOBAL_THRESHOLDS_DEG
    locals_deg = REDUCED_LOCAL_THRESHOLDS_DEG if reduced else LOCAL_THRESHOLDS_DEG
    classes = lab.footprint_classes(
        spacing, REDUCED_MAX_RADIUS_UM if reduced else MAX_RADIUS_UM,
        min_radius_um=MIN_RADIUS_UM)
    grid = []
    for radius, voxels in classes:
        requirements = set()
        for fraction in tolerances:
            # Store the tolerance, but de-duplicate on the integer requirement
            # it produces -- two tolerances that demand the same neighbour
            # count are the same configuration.
            requirement = int(neighbour_requirements(float(fraction), voxels)[-1])
            if requirement in requirements:
                continue
            requirements.add(requirement)
            for local in locals_deg:
                for global_threshold in globals_deg:
                    for min_cell_size in MIN_CELL_SIZES:
                        grid.append({
                            "footprint_radius_um": radius,
                            "footprint_voxels": voxels,
                            "footprint_tolerance": float(fraction),
                            "neighbour_requirement": requirement,
                            "local_threshold_deg": float(local),
                            "global_threshold_deg": float(global_threshold),
                            "min_cell_size": int(min_cell_size),
                            "kam_radius_um": PINNED_KAM_RADIUS_UM,
                            "seed": 0,
                        })
    return grid


MARKER_KEYS = ("footprint_radius_um", "footprint_tolerance", "local_threshold_deg",
               "global_threshold_deg", "min_cell_size", "kam_radius_um", "seed")


def marker_key(row: dict) -> str:
    return config_hash({k: row.get(k) for k in MARKER_KEYS})


def select_finalists(rows: list[dict], limit: int, keys) -> list[dict]:
    """Leaders under *every* policy the study reports, de-duplicated.

    Selecting finalists on one metric while reporting another is how a staged
    search quietly loses its own winner.  On the primary phantom the best
    marker configuration under the headline policy sits around 900th by
    absolute cell-count error -- far outside any sensible cut -- so a
    count-ranked finalist list would carry it into neither the KAM sweep nor
    the merge sweep, and the reported optimum would be whatever survived a
    filter it was never judged by.

    The budget is therefore split evenly across the ranking keys instead of
    spent entirely on whichever one happens to be listed first.  Overlap
    between the keys is removed rather than double-counted, so a configuration
    that leads under both costs one slot, not two.
    """

    keys = list(keys)
    per_key = max(limit // len(keys), 1)
    chosen: list[dict] = []
    seen: set[str] = set()
    for key in keys:
        for row in sorted(rows, key=key)[:per_key]:
            handle = marker_key(row)
            if handle in seen:
                continue
            seen.add(handle)
            chosen.append(row)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["markers", "kam", "merge", "baseline",
                                          "finalise", "report"])
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--phantom", type=Path,
                        default=HERE / "phantoms" / "primary_6p2_consistent.npz")
    parser.add_argument("--out", type=Path, default=HERE / "runs/primary")
    parser.add_argument("--finalists", type=int, default=FINALIST_COUNT)
    parser.add_argument("--reduced", action="store_true",
                        help="thin the tolerance and global-threshold axes; "
                             "used for the strain series, whose phantoms are denser")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="seed orderings to sweep in the frozen marker "
                             "grid; more than one turns it into an ablation "
                             "of the flood fill's own stochasticity")
    parser.add_argument("--frozen", action="store_true",
                        help="pin every axis to DEFAULTS and sweep only the "
                             "local threshold; prices the defaults jointly. "
                             "The kam stage is meaningless here -- the radius "
                             "is one of the pinned axes -- so skip it.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    import phantom_lab as lab

    with np.load(args.phantom) as data:
        spacing = tuple(float(v) for v in data["spacing"])

    store = {name: args.out / f"{name}.jsonl"
             for name in ("markers", "kam", "merge", "baseline", "final")}

    if args.stage == "markers":
        grid = (frozen_marker_grid(spacing, seeds=tuple(args.seeds))
                if args.frozen
                else marker_grid(spacing, reduced=args.reduced))
        seen = {marker_key(row) for row in read_rows(store["markers"])}
        todo = [task for task in grid if marker_key(task) not in seen]
        print(f"marker grid: {len(grid)} configurations, {len(todo)} still to run")
        run_pool(todo, task_markers, store["markers"], args.phantom, args.workers,
                 "markers")

    elif args.stage == "kam":
        rows = [r for r in read_rows(store["markers"]) if r.get("status") == "ok"]
        # Both reported policies contribute leaders.  ``fusion_averse_key`` is
        # not among them: it ranks on identity recall, which the cheap marker
        # scoring does not compute, so at this stage it would degenerate to its
        # tie-breakers.
        best = select_finalists(rows, args.finalists,
                                (strict_recovery_key, count_first_key))
        classes = lab.footprint_classes(spacing, MAX_RADIUS_UM,
                                        min_radius_um=MIN_RADIUS_UM)
        tasks = []
        for row in best:
            task = {k: row[k] for k in MARKER_KEYS}
            task["kam_grid"] = classes
            tasks.append(task)
        print(f"kam stage: {len(tasks)} marker configurations x {len(classes)} radii")
        run_pool(tasks, task_kam, store["kam"], args.phantom, args.workers,
                 "kam", expand=True)

    elif args.stage == "merge":
        rows = [r for r in read_rows(store["markers"]) if r.get("status") == "ok"]
        # Merging exists to rescue an over-segmentation, so the candidates are
        # configurations that produce too many cells.  They are ranked by
        # fusion -- the damage merging cannot undo -- rather than by cell
        # count or ARI, both of which punish the very over-segmentation the
        # merge step is there to clean up.
        candidates = [r for r in rows if r["cell_count_error"] > 0]
        candidates.sort(key=merge_candidate_key)
        sizes = ([DEFAULTS["merge_size_voxels"]] if args.frozen
                 else list(MERGE_SIZE_VOXELS))
        thresholds = ([DEFAULTS["merge_threshold_deg"]] if args.frozen
                      else list(MERGE_THRESHOLDS_DEG))
        tasks = []
        for row in candidates[: args.finalists]:
            task = {k: row[k] for k in MARKER_KEYS}
            task["merge_sizes"] = list(sizes)
            task["merge_thresholds"] = list(thresholds)
            tasks.append(task)
        print(f"merge stage: {len(tasks)} configurations x "
              f"{len(sizes) * len(thresholds)} merge settings")
        run_pool(tasks, task_merge, store["merge"], args.phantom, args.workers,
                 "merge", expand=True)

    elif args.stage == "baseline":
        classes = lab.footprint_classes(spacing, MAX_RADIUS_UM,
                                        min_radius_um=MIN_RADIUS_UM)
        if args.frozen:
            available = [r for r, _ in classes]
            radii = [min(available, key=lambda x, t=t: abs(x - t))
                     for t in BASELINE_REPLICATE_RADII_UM]
            radii = sorted(set(radii))
            min_sizes = BASELINE_REPLICATE_MIN_SIZES
            connectivities = BASELINE_REPLICATE_CONNECTIVITIES
        else:
            radii = [r for r, _ in classes]
            min_sizes = BASELINE_MIN_SIZES
            connectivities = BASELINE_CONNECTIVITIES
        grid = [
            {"percentile": float(p), "kam_radius_um": radius,
             "min_cell_size": int(m), "connectivity": int(c)}
            for p in BASELINE_PERCENTILES
            for radius in radii
            for m in min_sizes
            for c in connectivities
        ]
        seen = {config_hash({k: r.get(k) for k in
                             ("percentile", "kam_radius_um", "min_cell_size", "connectivity")})
                for r in read_rows(store["baseline"])}
        todo = [t for t in grid
                if config_hash({k: t[k] for k in
                                ("percentile", "kam_radius_um", "min_cell_size", "connectivity")})
                not in seen]
        print(f"baseline grid: {len(grid)} configurations, {len(todo)} still to run")
        run_pool(todo, task_baseline, store["baseline"], args.phantom, args.workers,
                 "baseline")

    elif args.stage == "finalise":
        tasks = collect_finalists(store, args.finalists)
        print(f"finalise: {len(tasks)} configurations for full scoring")
        run_pool(tasks, task_finalise, store["final"], args.phantom, args.workers,
                 "finalise")

    elif args.stage == "report":
        build_report(store, args.out)


def collect_finalists(store: dict, limit: int) -> list[dict]:
    """Best candidates from every stage, for full identity scoring.

    Candidates are drawn under *both* policies.  The two disagree sharply --
    count-first wants a large ``min_cell_size`` that deletes small cells, while
    fusion-averse wants a small one that keeps them and leaves the merge step
    to tidy up -- so scoring only one policy's leaders would leave the other
    unable to name a winner.
    """

    def count_rank(rows):
        return sorted(rows, key=count_first_key)

    tasks: list[dict] = []
    seen: set[str] = set()

    def add(row, arm, extra=()):
        keys = list(MARKER_KEYS) + list(extra)
        task = {k: row[k] for k in keys if k in row}
        task["arm"] = arm
        key = config_hash(task)
        if key not in seen:
            seen.add(key)
            tasks.append(task)

    per_policy = max(limit // 2, 1)
    for name in ("kam", "markers"):
        rows = [r for r in read_rows(store[name]) if r.get("status") == "ok"]
        for row in count_rank(rows)[:per_policy]:
            add(row, "flood fill")
        for row in sorted(rows, key=cheap_fusion_key)[:per_policy]:
            add(row, "flood fill")

    merge_rows = [r for r in read_rows(store["merge"]) if r.get("status") == "ok"]
    extra = ("merge_size_voxels", "merge_threshold_deg")
    for row in count_rank(merge_rows)[:per_policy]:
        add(row, "flood fill + merge", extra)
    # Merged rows are finished partitions, so they are ranked by the headline
    # policy rather than by the pre-merge candidate key.
    for row in sorted(merge_rows, key=cheap_fusion_key)[:per_policy]:
        add(row, "flood fill + merge", extra)

    baseline_rows = [r for r in read_rows(store["baseline"]) if r.get("status") == "ok"]
    for ranked in (count_rank(baseline_rows)[:per_policy],
                   sorted(baseline_rows, key=cheap_fusion_key)[:per_policy]):
        for row in ranked:
            task = {k: row[k] for k in ("percentile", "kam_radius_um",
                                        "min_cell_size", "connectivity")}
            task["arm"] = "KAM threshold"
            key = config_hash(task)
            if key not in seen:
                seen.add(key)
                tasks.append(task)
    return tasks


def build_report(store: dict, out: Path) -> None:
    import pandas as pd

    rows = [r for r in read_rows(store["final"]) if r.get("status") == "ok"]
    if not rows:
        print("no finalised rows yet -- run the finalise stage first")
        return
    frame = pd.DataFrame(rows)
    frame["arm"] = frame["arm"].fillna("flood fill")
    frame.to_csv(out / "capped_finalists.csv", index=False)

    report = {
        "radius_cap_um": MAX_RADIUS_UM,
        "n_finalised": len(rows),
        "strict_tau": STRICT_TAU,
        "policies": {
            STRICT_RECOVERY: {
                "description": "most true cells recovered at >= "
                               f"{STRICT_TAU} purity and completeness, then the "
                               "least contamination -- the voxels sitting in the "
                               "wrong cell. Splitting costs nothing, fusing costs "
                               "in proportion to how wrong it is. Mild "
                               "over-segmentation is permitted; the flood fill "
                               "is assessed after its merge step.",
                "mild_oversegmentation_limit": MILD_OVERSEGMENTATION,
                "retired_unmerged_surplus_tolerance":
                    RETIRED_UNMERGED_SURPLUS_TOLERANCE,
                "order": [
                    f"final partition within {MILD_OVERSEGMENTATION}x the truth",
                    f"maximum true cells recovered at {STRICT_TAU}",
                    "minimum contamination (voxels in the wrong cell)",
                    "minimum fused true cells",
                    "maximum ARI", "minimum boundary ASSD"],
            },
            COUNT_FIRST: {
                "description": "hit the true cell count, then break ties on "
                               "identity F1. Correct when the partition is the "
                               "final answer.",
                "order": ["minimum absolute cell-count error", "maximum identity F1",
                          "maximum orientation-correct F1 at 0.02 deg", "maximum ARI",
                          "minimum VI", "minimum boundary ASSD"],
            },
            FUSION_AVERSE: {
                "description": "rank on how many true cells come out right, not "
                               "on how many come out. Tolerates a surplus of "
                               "cells, which the merge step removes, but not "
                               "fusion, which nothing can undo. Cell count is a "
                               "diagnostic, not a selector.",
                "oversegmentation_limit": OVERSEGMENTATION_LIMIT,
                "order": [f"predicted cells within {OVERSEGMENTATION_LIMIT}x the truth",
                          "maximum identity recall",
                          "minimum VI-merge (under-segmentation)",
                          "maximum ARI", "minimum boundary ASSD"],
            },
        },
        "winners": {},
    }
    columns = ["arm", "n_cells_pred", "recovered_at_90", "recovery_rate_at_90",
               "contamination", "fused_true_cells", "strict_split_true_cells",
               "identity_recall",
               "identity_f1", "vi_merge_bits", "ari", "boundary_assd_um",
               "footprint_radius_um", "footprint_tolerance", "local_threshold_deg",
               "min_cell_size", "kam_radius_um", "merge_size_voxels",
               "merge_threshold_deg"]

    for policy_id, key in POLICIES.items():
        winners = {}
        for arm, group in frame.groupby("arm"):
            winners[arm] = min(group.to_dict("records"), key=key)
        report["winners"][policy_id] = winners
        print(f"\n=== best per arm under {policy_id} ===")
        table = pd.DataFrame(list(winners.values()))
        print(table[[c for c in columns if c in table]].to_string(index=False))

    (out / "capped_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_plain) + "\n"
    )
    print(f"\nwritten: {out/'capped_report.json'}")


if __name__ == "__main__":
    main()
