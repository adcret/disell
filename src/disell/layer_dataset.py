"""Loader for prepared per-layer DFXM datasets.

Layout expected on disk::

    <root>/layer_1_1/{mean.npy, motors.npy, processing_info.json, segmap.npy}
    <root>/layer_2_1/...
    ...

``mean.npy``   : (H, W, C) per-pixel angular feature map (e.g. chi/phi, deg).
``motors.npy`` : (C, m, n) angular scan grid of the mosaicity scan.
``processing_info.json`` : provenance of the layer preparation.
``segmap.npy`` : optional historic per-layer 2D segmentation (reference only —
                 it is NOT a valid-domain mask).

The loader discovers layers by a numeric regular expression, sorts them
numerically, validates cross-layer consistency, applies an optional crop and
stacks the feature maps along the leading Z axis.  Nothing in this module
hard-codes a dataset path; the root always comes from the caller.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class LayerDatasetError(RuntimeError):
    """Raised when the on-disk layer dataset is missing or inconsistent."""


@dataclass
class LayerVolume:
    """A stacked (Z, Y, X, C) angular feature volume with metadata."""

    field: np.ndarray                 # (Z, Y, X, C) float32
    mask: np.ndarray                  # (Z, Y, X) bool, True = valid
    spacing_nm: Tuple[float, float, float]   # (dz, dy, dx)
    angle_unit: str
    channel_names: Tuple[str, ...]
    layer_names: List[str]
    motors: np.ndarray                # (C, m, n) angular grid (same all layers)
    processing_info: List[Dict[str, Any]]
    segmaps: Optional[np.ndarray]     # (Z, h, w) historic labels or None
    source: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape_zyx(self) -> Tuple[int, int, int]:
        return self.field.shape[:3]

    def channel_ranges(self) -> List[Tuple[float, float]]:
        """(min, max) of the motor grid per channel."""
        return [
            (float(self.motors[c].min()), float(self.motors[c].max()))
            for c in range(self.motors.shape[0])
        ]


def discover_layers(
    root: Path | str, pattern: str = r"layer_(\d+)_1$"
) -> List[Tuple[int, Path]]:
    """Find layer directories under ``root`` and sort them numerically.

    Returns a list of (layer_number, path). Raises if none are found or if
    two directories map to the same number.
    """
    root = Path(root)
    if not root.is_dir():
        raise LayerDatasetError(f"dataset root {root} is not a directory")
    rx = re.compile(pattern)
    found: List[Tuple[int, Path]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = rx.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        raise LayerDatasetError(
            f"no directories matching {pattern!r} under {root}"
        )
    numbers = [n for n, _ in found]
    if len(set(numbers)) != len(numbers):
        raise LayerDatasetError(f"duplicate layer numbers under {root}: {numbers}")
    return sorted(found, key=lambda t: t[0])


def load_layer_volume(
    root: Path | str,
    *,
    spacing_nm: Sequence[float],
    angle_unit: str = "deg",
    channel_names: Sequence[str] = ("chi", "phi"),
    crop: Optional[Sequence[int]] = None,
    pattern: str = r"layer_(\d+)_1$",
    expect_n_layers: Optional[int] = None,
    expect_shape_yx: Optional[Tuple[int, int]] = None,
    motor_range_tolerance: float = 1e-3,
    load_segmaps: bool = False,
) -> LayerVolume:
    """Load, validate and stack a per-layer dataset.

    Parameters
    ----------
    root
        Dataset root containing the ``layer_<n>_1`` directories.
    spacing_nm
        Physical voxel spacing (dz, dy, dx) in nanometres. This is metadata
        the files do not carry, so the caller must supply it.
    crop
        Optional (row0, row1, col0, col1) applied to every layer *after*
        loading, e.g. the analysis window of a previous study.
    expect_n_layers, expect_shape_yx
        Optional hard expectations; a mismatch raises.
    motor_range_tolerance
        Feature values outside [motor_min - tol, motor_max + tol] (per
        channel, in the units of ``motors.npy``) are treated as failed fits
        and masked out.
    load_segmaps
        Also load ``segmap.npy`` per layer (historic 2D labels, reference
        only). They are kept at their native shape and NOT used as a mask.
    """
    root = Path(root)
    layers = discover_layers(root, pattern)
    if expect_n_layers is not None and len(layers) != expect_n_layers:
        raise LayerDatasetError(
            f"expected {expect_n_layers} layers, found {len(layers)}: "
            f"{[p.name for _, p in layers]}"
        )

    fields: List[np.ndarray] = []
    motors_all: List[np.ndarray] = []
    infos: List[Dict[str, Any]] = []
    segmaps: List[np.ndarray] = []
    layer_names: List[str] = []

    for num, p in layers:
        mean_path = p / "mean.npy"
        motors_path = p / "motors.npy"
        info_path = p / "processing_info.json"
        for f in (mean_path, motors_path, info_path):
            if not f.exists():
                raise LayerDatasetError(f"missing file {f}")
        arr = np.load(mean_path)
        if arr.ndim != 3:
            raise LayerDatasetError(
                f"{mean_path} has shape {arr.shape}; expected (H, W, C)"
            )
        motors = np.load(motors_path)
        if motors.ndim != 3 or motors.shape[0] != arr.shape[-1]:
            raise LayerDatasetError(
                f"{motors_path} shape {motors.shape} does not provide one "
                f"grid per feature channel of {mean_path} shape {arr.shape}"
            )
        with open(info_path) as fh:
            info = json.load(fh)
        fields.append(arr)
        motors_all.append(motors)
        infos.append(info)
        layer_names.append(p.name)
        if load_segmaps:
            sp = p / "segmap.npy"
            if sp.exists():
                segmaps.append(np.load(sp))

    # --- cross-layer consistency ------------------------------------------
    shapes = {a.shape for a in fields}
    if len(shapes) != 1:
        raise LayerDatasetError(f"layer feature maps differ in shape: {shapes}")
    if expect_shape_yx is not None and fields[0].shape[:2] != tuple(expect_shape_yx):
        # only checked pre-crop when no crop is requested
        if crop is None:
            raise LayerDatasetError(
                f"layer shape {fields[0].shape[:2]} != expected {expect_shape_yx}"
            )
    for m in motors_all[1:]:
        if m.shape != motors_all[0].shape or not np.allclose(
            m, motors_all[0], atol=motor_range_tolerance
        ):
            raise LayerDatasetError("motors.npy differs between layers")
    if load_segmaps and segmaps and len({s.shape for s in segmaps}) != 1:
        raise LayerDatasetError("segmap.npy differs in shape between layers")

    C = fields[0].shape[-1]
    if len(channel_names) != C:
        raise LayerDatasetError(
            f"channel_names {channel_names} does not match C={C}"
        )

    volume = np.stack(fields, axis=0)

    n_crop_invalid = None
    if crop is not None:
        r0, r1, c0, c1 = [int(v) for v in crop]
        if not (0 <= r0 < r1 <= volume.shape[1] and 0 <= c0 < c1 <= volume.shape[2]):
            raise LayerDatasetError(
                f"crop {crop} outside layer shape {volume.shape[1:3]}"
            )
        volume = volume[:, r0:r1, c0:c1]
    if expect_shape_yx is not None and volume.shape[1:3] != tuple(expect_shape_yx):
        raise LayerDatasetError(
            f"volume YX shape {volume.shape[1:3]} != expected {expect_shape_yx}"
        )

    # --- validity mask: finite and inside the motor range ------------------
    motors0 = motors_all[0].astype(np.float64)
    mask = np.isfinite(volume).all(axis=-1)
    for c in range(C):
        lo = motors0[c].min() - motor_range_tolerance
        hi = motors0[c].max() + motor_range_tolerance
        mask &= (volume[..., c] >= lo) & (volume[..., c] <= hi)
    n_crop_invalid = int((~mask).sum())

    if len(spacing_nm) != 3:
        raise LayerDatasetError("spacing_nm must be (dz, dy, dx)")

    return LayerVolume(
        field=volume.astype(np.float32),
        mask=mask,
        spacing_nm=tuple(float(s) for s in spacing_nm),
        angle_unit=angle_unit,
        channel_names=tuple(channel_names),
        layer_names=layer_names,
        motors=motors0,
        processing_info=infos,
        segmaps=np.stack(segmaps) if (load_segmaps and segmaps) else None,
        source={
            "root": str(root),
            "pattern": pattern,
            "crop": list(crop) if crop is not None else None,
            "n_layers": len(layer_names),
            "n_invalid_voxels": n_crop_invalid,
            "raw_layer_shape": list(fields[0].shape),
        },
    )
