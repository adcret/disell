"""Version-aware accounting for the two-stage benchmark trial store.

Scientific counts deliberately exclude schema-v1 rows.  Durable storage and
scientific eligibility are different concepts: the former is an append-only
forensic count, while the latter is a deduplicated set of successful schema-v2
``(config_key, candidate_order_seed)`` trials.
"""
from __future__ import annotations

import collections

ALGORITHM_ID = "size_prioritised_multiseed_v1"
SCIENTIFIC_METRICS_VERSION = 2
BROAD_SELECTABLE_TARGET = 8_000


def trial_key(row):
    config = row.get("config_key")
    seed = row.get("candidate_order_seed", row.get("seed", row.get("random_seed")))
    if config is None or seed is None:
        return None
    return str(config), int(seed)


def is_v2(row):
    return (
        row.get("algorithm_id", row.get("algorithm")) == ALGORITHM_ID
        and row.get("scientific_metrics_version") == SCIENTIFIC_METRICS_VERSION
    )


def is_selectable_v2(row):
    return is_v2(row) and row.get("status_category") == "ok"


def is_broad_stage(row):
    stage = str(row.get("stage", ""))
    return stage == "broad" or stage.startswith("broad_v2_replacement_")


def unique_rows(rows, predicate=lambda row: True):
    """One row per semantic trial, preferring the last durable occurrence."""
    selected = {}
    for row in rows:
        key = trial_key(row)
        if key is not None and predicate(row):
            selected[key] = row
    return list(selected.values())


def accounting(rows, planned_unique_v2_trials=BROAD_SELECTABLE_TARGET):
    hashes = [row.get("config_hash") for row in rows if row.get("config_hash")]
    all_trials = {key for row in rows if (key := trial_key(row)) is not None}
    v1_trials = {
        key for row in rows
        if row.get("scientific_metrics_version") != SCIENTIFIC_METRICS_VERSION
        and (key := trial_key(row)) is not None
    }
    v2_trials = {
        key for row in rows if is_v2(row) and (key := trial_key(row)) is not None
    }
    selectable = unique_rows(rows, is_selectable_v2)
    selectable_keys = {trial_key(row) for row in selectable}
    # A later ceiling-validation row may supersede a broad row for scientific
    # selection, but it must not erase the fact that the semantic trial was
    # supplied by the broad design.
    broad = unique_rows(
        [row for row in rows if is_selectable_v2(row) and is_broad_stage(row)],
        is_selectable_v2,
    )
    broad_keys = {trial_key(row) for row in broad}
    categories = collections.Counter(row.get("status_category") for row in rows)
    v2_categories = collections.Counter(
        row.get("status_category") for row in rows if is_v2(row)
    )
    return {
        "durable_rows": len(rows),
        "superseded_v1_rows": sum(
            row.get("scientific_metrics_version") != SCIENTIFIC_METRICS_VERSION
            for row in rows
        ),
        "unique_full_hashes": len(set(hashes)),
        "unique_trials": len(all_trials),
        "selectable_v2_trials": len(selectable_keys),
        "planned_unique_v2_trials": int(planned_unique_v2_trials),
        "broad_selectable_v2_trials": len(broad_keys),
        "broad_selectable_v2_target": BROAD_SELECTABLE_TARGET,
        "broad_selectable_v2_shortfall": max(
            0, BROAD_SELECTABLE_TARGET - len(broad_keys)
        ),
        "v1_v2_semantic_trial_overlap": len(v1_trials & v2_trials),
        "status_categories_all_durable_rows": dict(categories),
        "status_categories_schema_v2": dict(v2_categories),
    }
