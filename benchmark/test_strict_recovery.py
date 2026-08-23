#!/usr/bin/env python3
"""Tests for the strict (>= tau purity and completeness) recovery metric."""
import numpy as np
import pytest

import strict_recovery as sr


def line(assignments):
    return np.asarray(assignments, dtype=np.int32).reshape(1, 1, -1)


def test_perfect_partition_recovers_everything():
    truth = line([1]*10 + [2]*10)
    out = sr.strict_recovery(truth, truth.copy())
    assert out["recovered_at_tau"] == 2
    assert out["recovery_rate_at_tau"] == 1.0
    assert out["fused_true_cells"] == 0


def test_fusion_is_detected_and_not_counted_as_recovered():
    truth = line([1]*10 + [2]*10)
    pred = line([1]*20)                      # one cell swallows both
    out = sr.strict_recovery(truth, pred)
    assert out["recovered_at_tau"] == 0
    assert out["fused_true_cells"] == 2


def test_a_split_cell_is_not_recovered_but_is_not_fused():
    truth = line([1]*20)
    pred = line([1]*10 + [2]*10)             # halved
    out = sr.strict_recovery(truth, pred)
    assert out["recovered_at_tau"] == 0
    assert out["split_true_cells"] == 1
    assert out["fused_true_cells"] == 0


def test_ninety_percent_boundary_is_respected():
    truth = line([1]*10 + [2]*10)
    # 9 of cell 1's 10 voxels: completeness 0.9, purity 1.0 -> recovered.
    pred = line([1]*9 + [0] + [2]*10)
    out = sr.strict_recovery(truth, pred, tau=0.9)
    assert out["recovered_at_tau"] == 2
    # 8 of 10 -> completeness 0.8, below tau.
    pred = line([1]*8 + [0]*2 + [2]*10)
    out = sr.strict_recovery(truth, pred, tau=0.9)
    assert out["recovered_at_tau"] == 1


def test_impurity_blocks_recovery_even_when_complete():
    truth = line([1]*10 + [2]*10)
    # Predicted cell 1 takes all of true 1 plus 2 voxels of true 2:
    # completeness 1.0 but purity 10/12 = 0.83 < 0.9.
    pred = line([1]*12 + [2]*8)
    out = sr.strict_recovery(truth, pred, tau=0.9)
    assert 1 not in np.unique(truth) or out["recovered_at_tau"] == 0


def test_recovery_is_one_to_one_above_a_half():
    rng = np.random.default_rng(0)
    truth = rng.integers(1, 12, size=(3, 9, 9)).astype(np.int32)
    pred = rng.integers(1, 12, size=(3, 9, 9)).astype(np.int32)
    out = sr.strict_recovery(truth, pred, tau=0.9)
    assert out["recovered_at_tau"] <= out["n_cells_true"]
    assert out["recovered_at_tau"] <= out["n_cells_pred"]


def test_tau_below_a_half_is_rejected():
    truth = line([1]*10)
    with pytest.raises(ValueError):
        sr.strict_recovery(truth, truth.copy(), tau=0.4)


def test_oversegmentation_never_creates_fusion():
    truth = line([1]*30)
    pred = line([1]*10 + [2]*10 + [3]*10)    # shattered, but nothing fused
    out = sr.strict_recovery(truth, pred)
    assert out["fused_true_cells"] == 0
    assert out["split_true_cells"] == 1


def test_counts_ignore_empty_and_background_labels():
    truth = line([0]*5 + [1]*10 + [3]*10)    # label 2 absent entirely
    out = sr.strict_recovery(truth, truth.copy())
    assert out["n_cells_true"] == 2
    assert out["recovered_at_tau"] == 2


def test_purity_only_is_at_least_strict_recovery():
    rng = np.random.default_rng(3)
    truth = rng.integers(1, 8, size=(2, 8, 8)).astype(np.int32)
    pred = rng.integers(1, 8, size=(2, 8, 8)).astype(np.int32)
    out = sr.strict_recovery(truth, pred, tau=0.9)
    assert out["recovered_purity_only"] >= out["recovered_at_tau"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_contamination_grades_fusion_severity():
    """A 50/50 fusion must score worse than a 95/5 one, which counting cannot."""
    truth = line([1]*100 + [2]*100)
    even = line([1]*200)                                  # one cell, 50/50
    # Cell 1 takes 20 of cell 2's voxels: still a fusion (20 % is above the
    # 10 % "substantial" threshold) but far closer to the truth than 50/50.
    lopsided = line([1]*120 + [2]*80)
    a = sr.strict_recovery(truth, even)
    b = sr.strict_recovery(truth, lopsided)
    assert a["fused_true_cells"] == b["fused_true_cells"] == 2, "counts are equal"
    assert a["contamination"] > b["contamination"], "severity must separate them"
    assert a["contamination"] == pytest.approx(0.5)
    assert b["contamination"] == pytest.approx(20 / 200)


def test_a_small_incursion_is_not_a_fusion():
    """Below the substantial threshold an overlap is contamination, not fusion."""
    truth = line([1]*100 + [2]*100)
    pred = line([1]*105 + [2]*95)          # 5 % of cell 2 -- under the 10 % rule
    out = sr.strict_recovery(truth, pred)
    assert out["fused_true_cells"] == 0
    assert out["contamination"] == pytest.approx(5 / 200)


def test_a_perfect_partition_has_no_contamination():
    truth = line([1]*10 + [2]*10)
    out = sr.strict_recovery(truth, truth.copy())
    assert out["contamination"] == 0.0
    assert out["fused_predicted_cells"] == 0


def test_oversegmentation_is_not_contamination():
    """Splitting a cell moves no voxel into the wrong cell."""
    truth = line([1]*30)
    pred = line([1]*10 + [2]*10 + [3]*10)
    out = sr.strict_recovery(truth, pred)
    assert out["contamination"] == 0.0
