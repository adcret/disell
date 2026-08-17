import json

import pandas as pd

import oracle_core as oc
import two_stage_analyze as analysis
import two_stage_paired_comparison as paired
import two_stage_strain_suite as strain
from two_stage_store import normalize_result, validate_result


def identity():
    return json.loads(
        (paired.PRIMARY / "implementation_audit.json").read_text()
    )


def parameters():
    return {
        "local_threshold_deg": 0.01,
        "global_threshold_deg": -1.0,
        "footprint_tolerance": 0.5,
        "footprint_radius_um": 1.0,
        "min_cell_size": 50,
        "kam_radius_um": 1.0,
    }


def test_paired_algorithm_hashes_are_distinct_and_schema_complete():
    paired.IDENTITY = identity()
    random_hash, random_identity = paired._identity(
        paired.ALGORITHMS[0], parameters(), 4
    )
    size_hash, _ = paired._identity(paired.ALGORITHMS[1], parameters(), 4)
    assert random_hash != size_hash
    row = normalize_result({
        "config_hash": random_hash,
        "configuration_identity": random_identity,
        "algorithm": paired.ALGORITHMS[0],
        "candidate_order_seed": 4,
        "status": "ok",
    }, random_identity)
    validate_result(row)


def test_strain_hash_is_independent_of_bookkeeping_stage():
    strain.ID = identity()
    strain.ID.update(latent_sha256="phantom", truth_sha256="truth", mask_sha256="mask")
    broad, _ = strain.key(parameters(), 0, "broad")
    repeat, _ = strain.key(parameters(), 0, "five_seed_finalists")
    assert broad == repeat


def test_tolerance_selection_can_change_without_opaque_score():
    base = {
        **parameters(), "one_to_one_recovered_cells_mean": 300,
        "unmatched_true_cells_mean": 60, "merged_predicted_cells_mean": 2,
        "split_true_cells_mean": 3, "cell_count_error_mean": 0,
    }
    frame = pd.DataFrame([
        {**base, "orientation_correct_cells_at_0p005deg_mean": 280,
         "orientation_correct_cells_at_0p01deg_mean": 285,
         "orientation_correct_cells_at_0p02deg_mean": 290,
         "orientation_correct_cells_at_0p05deg_mean": 295},
        {**base, "local_threshold_deg": 0.02,
         "orientation_correct_cells_at_0p005deg_mean": 270,
         "orientation_correct_cells_at_0p01deg_mean": 290,
         "orientation_correct_cells_at_0p02deg_mean": 295,
         "orientation_correct_cells_at_0p05deg_mean": 300},
    ])
    chosen = analysis.tolerance_solution_table(frame)
    assert chosen.loc[chosen.orientation_tolerance_deg == .005,
                      "local_threshold_deg"].item() == .01
    assert chosen.loc[chosen.orientation_tolerance_deg == .05,
                      "local_threshold_deg"].item() == .02


def test_strain_finalist_selection_aggregates_seed_orders():
    base = {
        **parameters(), "status_category": "ok", "scientific_metrics_version": 2,
        "one_to_one_recovered_cells": 300, "unmatched_true_cells": 60,
        "unmatched_predictions": 10, "merged_predicted_cells": 2,
        "split_true_cells": 3, "cell_count_error": 0,
        "median_matched_mean_orientation_error_deg": .002,
        "vi_total_bits": .4, "ari": .9,
    }
    rows = [{**base, "candidate_order_seed": seed,
             "orientation_correct_cells_at_0p02deg": value}
            for seed, value in enumerate((280, 300, 290, 295, 285))]
    selected = strain.select_summary(rows)
    assert selected["n_seed_orders"] == 5
    assert selected["orientation_correct_cells_at_0p02deg_mean"] == 290
