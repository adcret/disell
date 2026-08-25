"""Tests for disell.registration: shift recovery, NaN padding policy,
sub-pixel support and the zero-z-pad edge case."""

import numpy as np
import pytest
from scipy.ndimage import shift as ndi_shift

from disell import register, apply_transforms


def _make_stack(shift_yx, n=3, seed=0):
    """Stack of n frames, frame i shifted by i*shift_yx relative to frame 0."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(80, 90))
    from scipy.ndimage import gaussian_filter

    base = gaussian_filter(base, 3)
    frames = []
    for i in range(n):
        frames.append(
            ndi_shift(base, np.asarray(shift_yx) * i, order=1, mode="nearest")
        )
    stack = np.stack(frames)[..., None]  # (T, Y, X, 1)
    return stack


def test_register_recovers_integer_shift():
    stack = _make_stack((3, -2), n=3)
    transforms = register(stack, registration_channel=0)
    assert transforms[1] is None  # middle frame is the reference
    np.testing.assert_allclose(transforms[0], [3, -2])
    np.testing.assert_allclose(transforms[2], [-3, 2])


def test_register_subpixel_upsample():
    # true inter-frame shift 2.5 px: integer registration is off by >= 0.5,
    # sub-pixel registration must do clearly better
    stack = _make_stack((2.5, 0.0), n=3)
    transforms = register(stack, registration_channel=0, upsample_factor=20)
    assert abs(transforms[0][0] - 2.5) <= 0.3
    assert abs(transforms[2][0] + 2.5) <= 0.3


def test_apply_transforms_nan_padding_integer():
    stack = _make_stack((0, 0), n=3)
    transforms = [None, np.array([2.0, 0.0]), None]
    out = apply_transforms(stack, transforms)
    assert out.shape == stack.shape
    # rows shifted in from outside must be NaN
    assert np.isnan(out[1, :2]).all()
    assert np.isfinite(out[1, 3:]).all()
    # reference frames untouched
    np.testing.assert_allclose(out[0], stack[0])


def test_apply_transforms_fractional_shift_no_sentinel_leak():
    stack = _make_stack((0, 0), n=3)
    transforms = [None, np.array([1.5, 0.0]), None]
    out = apply_transforms(stack, transforms)
    finite = out[1][np.isfinite(out[1])]
    # no -1e10-ish sentinel values may survive in the data
    assert finite.min() > -1e6
    # the blended edge rows are conservatively NaN
    assert np.isnan(out[1, :2]).all()


def test_apply_transforms_3d_zero_zpad_regression():
    # 3D spatial volumes with pure in-plane shifts used to return an empty
    # array because of a [0:-0] slice.
    rng = np.random.default_rng(1)
    vol = rng.normal(size=(2, 4, 10, 12, 1))
    transforms = [None, np.array([0.0, 2.0, 1.0])]
    out = apply_transforms(vol, transforms)
    assert out.shape == vol.shape
    assert np.isfinite(out[0]).all()


def test_apply_transforms_registration_roundtrip():
    stack = _make_stack((4, -3), n=3)
    transforms = register(stack, registration_channel=0)
    out = apply_transforms(stack, transforms)
    # after alignment all frames should agree on the common finite region
    common = np.isfinite(out).all(axis=(0, 3))
    a = out[0, ..., 0][common]
    b = out[2, ..., 0][common]
    assert np.corrcoef(a, b)[0, 1] > 0.98
