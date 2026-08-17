import pandas as pd

import paired_ordering_analysis as analysis


def row(pair, algorithm, predicted, recovered, ari, offset):
    return {
        "pairing_key": pair,
        "algorithm_id": algorithm,
        "candidate_order_seed": 0,
        "status_category": "ok",
        "n_true_cells": 10,
        "n_predicted_cells": predicted,
        "one_to_one_recovered_cells": recovered,
        "orientation_correct_cells_at_0p02deg": recovered - 1,
        "absolute_cell_count_error": abs(predicted - 10),
        "ari": ari,
        "vi_total_bits": 1 - ari,
        "boundary_assd_um": 0.2 - ari / 10,
        "footprint_radius_um": 1.0 + offset,
        "footprint_tolerance": 0.2,
        "global_threshold_deg": 0.5,
        "kam_radius_um": 1.0,
        "local_threshold_deg": 0.05,
        "min_cell_size": 10,
    }


def test_summary_uses_matched_configuration_effects():
    frame = pd.DataFrame(
        [
            row("a", analysis.RANDOM, 14, 7, 0.7, 0),
            row("a", analysis.ORDERED, 11, 9, 0.8, 0),
            row("b", analysis.RANDOM, 10, 9, 0.8, 1),
            row("b", analysis.ORDERED, 12, 8, 0.7, 1),
        ]
    )

    summary, effects = analysis.summarize(frame, seeds=[0])

    assert summary["cohort"]["complete_pairs"] == 2
    assert summary["cohort"]["usable_configurations"] == 2
    count = summary["configuration_medians"]["absolute_cell_count_error"]
    identity = summary["configuration_medians"]["identity_f1"]
    assert (count["ordered_better"], count["ordered_worse"]) == (1, 1)
    assert (identity["ordered_better"], identity["ordered_worse"]) == (1, 1)
    assert len(effects) == 2
