"""Deterministic tests for the running-mean global-threshold spread cap.

These tests call the pybind11 extension directly (``disell._flood_fill``)
because the running-mean behaviour is best exercised with an explicit seed
point, which the high-level Python wrappers do not expose.
"""

import numpy as np
import pytest

import disell._flood_fill as ff


# Shared synthetic geometry: a 1D property map (1, 1, N, 1) with
#   - a left plateau at 0.0
#   - a linearly interpolated bridge from 0.0 -> 0.5
#   - a right plateau at 0.5
# Adjacent differences along the bridge are 0.05, which is below the local
# threshold (0.1), so purely local growth can walk across the bridge.
LEFT = np.zeros(5)
BRIDGE = np.arange(1, 10) * 0.05  # 0.05, 0.10, ..., 0.45
RIGHT = np.full(5, 0.5)
VALUES = np.concatenate([LEFT, BRIDGE, RIGHT])
N = VALUES.size

LOCAL_THRESHOLD = 0.1
GLOBAL_THRESHOLD = 0.15
FOOTPRINT_TOLERANCE = 0.6
MIN_GRAIN_SIZE = 1
MAX_ITERATIONS = 1


def _make_inputs():
    prop = VALUES.reshape(1, 1, N, 1).astype(np.float32)
    mask = np.ones((1, 1, N), dtype=np.uint8)
    footprint = np.ones((1, 1, 3), dtype=bool)
    seed_points = np.array([[0, 0, 0]], dtype=np.int64)
    return prop, mask, footprint, seed_points


def _run(global_threshold):
    prop, mask, footprint, seed_points = _make_inputs()
    res = ff.flood_fill_random_seeds_3d(
        prop,
        footprint,
        LOCAL_THRESHOLD,
        global_threshold,
        FOOTPRINT_TOLERANCE,
        mask,
        MAX_ITERATIONS,
        MIN_GRAIN_SIZE,
        False,
        0,
        seed_points,
    )
    return res["segmentation"]


def test_global_disabled_merges_through_bridge():
    seg = _run(-1.0)[0, 0]
    # Seed labelled and growth reaches both plateaus.
    assert seg[0] != 0
    assert seg[-1] != 0
    # The whole row is one connected region with global growth disabled.
    assert np.all(seg != 0)


def test_global_enabled_stops_before_right_plateau():
    seg = _run(GLOBAL_THRESHOLD)[0, 0]
    # Seed is still labelled.
    assert seg[0] != 0
    # Growth must not reach the right plateau (value 0.5 voxels).
    right_plateau = seg[len(LEFT) + len(BRIDGE):]
    assert np.all(right_plateau == 0)
    assert seg[-1] == 0


def test_global_off_paths_are_equivalent():
    # global_threshold == 0 and a negative threshold both disable the cap and
    # must produce identical segmentations.
    seg_zero = _run(0.0)
    seg_neg = _run(-1.0)
    assert np.array_equal(seg_zero, seg_neg)
