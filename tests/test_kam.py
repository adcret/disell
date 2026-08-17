"""Tests for disell.properties.kam: reference values, NaN handling,
boolean footprints, border behaviour and per-channel RMS units."""

import numpy as np
import pytest

from disell import kam


def brute_force_kam(field, footprint, per_channel_rms=False):
    """Naive reference implementation, (Z, Y, X, C) input."""
    Z, Y, X, C = field.shape
    kz, ky, kx = footprint.shape
    hz, hy, hx = kz // 2, ky // 2, kx // 2
    out = np.full((Z, Y, X), np.nan)
    for z in range(hz, Z - hz):
        for y in range(hy, Y - hy):
            for x in range(hx, X - hx):
                c = field[z, y, x]
                if np.isnan(c).any():
                    continue
                dists = []
                for dz in range(-hz, hz + 1):
                    for dy in range(-hy, hy + 1):
                        for dx in range(-hx, hx + 1):
                            if dz == dy == dx == 0:
                                continue
                            if not footprint[dz + hz, dy + hy, dx + hx]:
                                continue
                            n = field[z + dz, y + dy, x + dx]
                            if np.isnan(n).any():
                                continue
                            d = np.sqrt(((n - c) ** 2).sum())
                            if per_channel_rms:
                                d = d / np.sqrt(C)
                            dists.append(d)
                if dists:
                    out[z, y, x] = np.mean(dists)
    return out


def test_kam_matches_brute_force_3d():
    rng = np.random.default_rng(0)
    field = rng.normal(size=(4, 6, 7, 2))
    fp = np.ones((3, 3, 3), dtype=bool)
    got = kam(field, ndim=3, size=3, fill_invalid=np.nan)
    want = brute_force_kam(field, fp)
    np.testing.assert_allclose(got, want, equal_nan=True, rtol=1e-12)


def test_kam_boolean_footprint():
    rng = np.random.default_rng(1)
    field = rng.normal(size=(5, 6, 7, 2))
    # anisotropic footprint: in-plane 8-neighbourhood plus z face neighbours
    fp = np.zeros((3, 3, 3), dtype=bool)
    fp[1] = True
    fp[0, 1, 1] = fp[2, 1, 1] = True
    got = kam(field, ndim=3, footprint=fp, fill_invalid=np.nan)
    want = brute_force_kam(field, fp)
    np.testing.assert_allclose(got, want, equal_nan=True, rtol=1e-12)


def test_kam_nan_in_any_channel_excluded():
    field = np.zeros((1, 5, 5, 2))
    field[0, 2, 2, 1] = np.nan  # NaN only in channel 1: voxel must be invalid
    got = kam(field, ndim=3, size=(1, 3, 3), fill_invalid=np.nan)
    assert np.isnan(got[0, 2, 2])
    # neighbours of the NaN voxel are still valid: they skip the invalid one
    assert np.isfinite(got[0, 2, 1])
    assert got[0, 2, 1] == 0.0


def test_kam_border_fill_invalid():
    field = np.zeros((3, 5, 5, 2))
    got = kam(field, ndim=3, size=3, fill_invalid=np.nan)
    assert np.isnan(got[0]).all()  # z-border margin
    assert np.isnan(got[:, 0, :]).all()
    assert np.isfinite(got[1, 1:-1, 1:-1]).all()
    # backwards-compatible default: zeros at the border
    got0 = kam(field, ndim=3, size=3)
    assert (got0[0] == 0).all()


def test_kam_per_channel_rms_units():
    rng = np.random.default_rng(2)
    field = rng.normal(size=(3, 5, 5, 2))
    l2 = kam(field, ndim=3, size=3, fill_invalid=np.nan)
    rms = kam(field, ndim=3, size=3, fill_invalid=np.nan, per_channel_rms=True)
    np.testing.assert_allclose(rms, l2 / np.sqrt(2), equal_nan=True, rtol=1e-12)


def test_kam_2d_path():
    rng = np.random.default_rng(3)
    field = rng.normal(size=(6, 7, 2))
    got = kam(field, ndim=2, size=3, fill_invalid=np.nan)
    want = brute_force_kam(field[None], np.ones((1, 3, 3), bool))[0]
    np.testing.assert_allclose(got, want, equal_nan=True, rtol=1e-12)
