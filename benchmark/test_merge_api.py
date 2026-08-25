"""Tests for the canonical flood-fill merge switch."""

from __future__ import annotations

import numpy as np

import phantom_lab
import pipelines


class _ToyPhantom:
    spacing_um_zyx = (1.0, 0.4, 0.4)
    field = np.array([[[[0.0, 0.0], [0.0, 0.0],
                       [0.01, 0.01], [0.01, 0.01]]]], dtype=np.float32)
    mask = np.ones(field.shape[:3], dtype=bool)

    def kam(self, radius_um):
        return np.zeros(self.field.shape[:3], dtype=np.float32)


def _fake_pipeline(field, mask, kam_map, footprint, **kwargs):
    labels = np.array([[[1, 1, 2, 2]]], dtype=np.int32)
    return labels.copy(), labels.copy(), np.array([2, 2]), {}, np.array([2, 2])


def _run(monkeypatch, **kwargs):
    monkeypatch.setattr(pipelines, "run_flood_fill_two_stage", _fake_pipeline)
    return phantom_lab.segment(
        _ToyPhantom(),
        local_threshold_deg=0.01,
        global_threshold_deg=-1.0,
        footprint_tolerance=0.04,
        footprint_radius_um=1.2,
        min_cell_size=1,
        kam_radius_um=1.2,
        seed=42,
        merge_mode="absolute",
        merge_threshold_deg=0.05,
        **kwargs,
    )


def test_merge_defaults_to_true(monkeypatch):
    default = _run(monkeypatch)
    explicit = _run(monkeypatch, merge=True)
    assert np.array_equal(default.labels, explicit.labels)
    assert default.params["merge"] is True


def test_merge_switch_exposes_premerge_result(monkeypatch):
    merged = _run(monkeypatch, merge=True)
    unmerged = _run(monkeypatch, merge=False)
    assert np.array_equal(merged.labels, np.ones_like(merged.labels))
    assert np.array_equal(unmerged.labels, np.array([[[1, 1, 2, 2]]]))
    assert unmerged.params["merge"] is False


def test_fixed_seed_is_deterministic_for_both_modes(monkeypatch):
    for merge in (True, False):
        first = _run(monkeypatch, merge=merge)
        second = _run(monkeypatch, merge=merge)
        assert np.array_equal(first.labels, second.labels)
