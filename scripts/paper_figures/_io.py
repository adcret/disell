"""Shared I/O helpers for the ``scripts/paper_figures`` scripts.

This module collects everything that is *not* segmentation logic:

- loading DFXM data with ``darling`` (preferred path),
- computing per-voxel orientation maps via ``darling.properties``,
- stacking 2D mosa-scan layers into a 3D ``(Z, Y, X, 2)`` orientation
  volume,
- loading masks from HDF5 or building them on the fly,
- saving labelled volumes / scalar fields with voxel spacing to HDF5
  and to ParaView-friendly VTI,
- writing a JSON sidecar that records every parameter used.

Conventions enforced by every helper here:

* Spatial axis order is **always** ``(Z, Y, X)`` for masks/labels and
  ``(Z, Y, X, 2)`` for orientation fields. Loading from ``darling`` is
  done in such a way that no silent transpose is performed; if the
  underlying data is already in another order, the user must convert it
  explicitly upstream.
* Voxel spacing is stored as ``(dz, dy, dx)`` in nanometres, propagated
  to every output file (HDF5 attribute, VTI ``Spacing``).
* Angles are kept in whatever unit the user specifies (``deg``,
  ``rad``, or ``mrad``); conversions, when necessary, are explicit and
  recorded in the parameter JSON.

This file is imported by the four runnable scripts in this folder. It
is not part of the ``disell`` public API.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants and small data classes
# ---------------------------------------------------------------------------

#: Axis order used everywhere in the paper-figure scripts.
SPATIAL_AXIS_ORDER = "ZYX"
#: Channel layout for orientation fields.
CHANNEL_LAYOUT = ("chi", "phi")


@dataclass
class VoxelSpacing:
    """Physical voxel spacing in nm, in (z, y, x) order.

    Used to propagate spacing to KAM kernels, to volume statistics, and
    to ParaView outputs. ``None`` for an axis means "unknown / leave
    as 1".
    """

    dz_nm: float
    dy_nm: float
    dx_nm: float

    def as_tuple_nm(self) -> Tuple[float, float, float]:
        return (self.dz_nm, self.dy_nm, self.dx_nm)

    def as_tuple_um(self) -> Tuple[float, float, float]:
        return (self.dz_nm * 1e-3, self.dy_nm * 1e-3, self.dx_nm * 1e-3)

    def voxel_volume_nm3(self) -> float:
        return float(self.dz_nm * self.dy_nm * self.dx_nm)


@dataclass
class OrientationVolume:
    """A 3D orientation volume on a regular grid.

    Attributes
    ----------
    field : ndarray
        Shape ``(Z, Y, X, 2)``, dtype ``float32``. Channel order follows
        :data:`CHANNEL_LAYOUT` = ``("chi", "phi")``.
    mask : ndarray
        Shape ``(Z, Y, X)``, dtype ``bool``. ``True`` = valid voxel.
    spacing : VoxelSpacing
        Physical voxel spacing.
    angle_unit : str
        ``"deg"``, ``"rad"`` or ``"mrad"``. Stored verbatim; nothing in
        these helpers ever silently converts it.
    source : dict
        Free-form dictionary describing what was loaded and how (HDF5
        path, scan ids, darling function used, etc.). Written into
        ``parameters_used.json``.
    """

    field: np.ndarray
    mask: np.ndarray
    spacing: VoxelSpacing
    angle_unit: str
    source: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.field.ndim != 4 or self.field.shape[-1] != 2:
            raise ValueError(
                f"orientation field must be (Z, Y, X, 2); got {self.field.shape}"
            )
        if self.mask.shape != self.field.shape[:3]:
            raise ValueError(
                f"mask shape {self.mask.shape} does not match "
                f"field spatial shape {self.field.shape[:3]}"
            )
        if self.angle_unit not in ("deg", "rad", "mrad"):
            raise ValueError(
                f"angle_unit must be 'deg', 'rad' or 'mrad'; got {self.angle_unit!r}"
            )

    @property
    def shape_zyx(self) -> Tuple[int, int, int]:
        return self.field.shape[:3]


# ---------------------------------------------------------------------------
# Parameter JSON
# ---------------------------------------------------------------------------


def write_parameters_json(out_dir: Path, params: Dict[str, Any]) -> Path:
    """Write a parameter sidecar JSON next to the script output.

    Adds a ``runtime`` section with python/platform info plus the
    starting wall-clock time. Returns the JSON path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(params)
    payload["runtime"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis_order": SPATIAL_AXIS_ORDER,
        "channel_layout": list(CHANNEL_LAYOUT),
    }
    path = out_dir / "parameters_used.json"
    with open(path, "w") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)
    return path


def _jsonable(obj: Any) -> Any:
    """Convert numpy scalars / arrays / Paths / ndarrays into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# darling-based loading
# ---------------------------------------------------------------------------


def _import_darling():
    try:
        import darling  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "darling is not installed. The paper-figure scripts require it "
            "for DFXM data loading. Install it from "
            "https://github.com/AxelHenningsson/darling and try again. "
            "If you absolutely cannot use darling, see the helper "
            "_io.load_orientation_volume_from_h5 which falls back to a "
            "tightly-scoped raw-HDF5 read; the script must be invoked "
            "with --no-darling and an explicit --field-dataset."
        ) from exc
    return __import__("darling")


def load_orientation_2d_from_darling(
    h5_path: str,
    scan_id: str,
    *,
    method: str = "mean",
    roi: Optional[Tuple[int, int, int, int]] = None,
    intensity_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one 2D mosa scan and return its (Y, X, 2) orientation map and mask.

    Parameters
    ----------
    h5_path
        Absolute path to the ID03 HDF5 file.
    scan_id
        Scan id string (e.g. ``"1.1"``).
    method
        ``"mean"`` (centre of mass via :func:`darling.properties.mean`)
        or ``"peak"`` (multi-peak finder via
        :func:`darling.properties.peaks`, dominant peak only).
    roi
        Optional ``(row_min, row_max, col_min, col_max)`` ROI on the
        detector, in pixels.
    intensity_threshold
        If given, voxels whose summed-over-motors intensity is below
        this value are masked out. Useful when the grain mask is not
        provided separately.

    Returns
    -------
    field_2d : ndarray, shape (Y, X, 2), dtype float32
        ``[..., 0]`` is the chi axis, ``[..., 1]`` is the phi axis (or
        whichever motor pair the scan was acquired on; the order
        follows the ``motor_names`` of the underlying ID03 reader).
    mask_2d : ndarray, shape (Y, X), dtype bool
        ``True`` where the orientation is meaningful.
    """
    darling = _import_darling()

    dset = darling.DataSet(h5_path, scan_id=scan_id, roi=roi, verbose=False)
    data, motors = dset.data, dset.motors

    if method == "mean":
        field_2d = darling.properties.mean(data, motors).astype(np.float32)
        intensity = data.sum(axis=tuple(range(2, data.ndim)))
    elif method == "peak":
        peakmap = darling.properties.peaks(data, k=1, coordinates=motors)
        field_2d = peakmap.mean.astype(np.float32)
        intensity = data.sum(axis=tuple(range(2, data.ndim)))
    else:
        raise ValueError(f"Unknown orientation method: {method!r}")

    if field_2d.ndim != 3 or field_2d.shape[-1] != 2:
        raise RuntimeError(
            "darling.properties returned an orientation map with shape "
            f"{field_2d.shape}; expected (Y, X, 2). Aborting rather than "
            "guessing."
        )

    valid = np.isfinite(field_2d).all(axis=-1)
    if intensity_threshold is not None:
        valid &= intensity > intensity_threshold

    return field_2d, valid.astype(bool)


def stack_layers_into_volume(
    layers: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    spacing: VoxelSpacing,
    angle_unit: str,
    source: Optional[Dict[str, Any]] = None,
) -> OrientationVolume:
    """Stack a list of (field_2d, mask_2d) pairs along the leading Z axis.

    Every layer must share the same ``(Y, X)`` shape; if they do not the
    function fails loudly.
    """
    if not layers:
        raise ValueError("layers must contain at least one (field, mask) pair")
    yx_shape = layers[0][0].shape[:2]
    for k, (f, m) in enumerate(layers):
        if f.shape[:2] != yx_shape:
            raise ValueError(
                f"layer {k} has YX shape {f.shape[:2]}, expected {yx_shape}"
            )
        if m.shape != yx_shape:
            raise ValueError(
                f"layer {k} mask has shape {m.shape}, expected {yx_shape}"
            )

    field = np.stack([f.astype(np.float32, copy=False) for f, _ in layers], axis=0)
    mask = np.stack([m.astype(bool, copy=False) for _, m in layers], axis=0)

    return OrientationVolume(
        field=field,
        mask=mask,
        spacing=spacing,
        angle_unit=angle_unit,
        source=source or {},
    )


def load_orientation_volume_from_darling(
    h5_path: str,
    scan_ids: Sequence[str],
    *,
    spacing: VoxelSpacing,
    angle_unit: str,
    method: str = "mean",
    roi: Optional[Tuple[int, int, int, int]] = None,
    intensity_threshold: Optional[float] = None,
) -> OrientationVolume:
    """Load multiple ID03 scan-ids and stack them as a 3D ``(Z, Y, X, 2)`` volume."""
    layers: List[Tuple[np.ndarray, np.ndarray]] = []
    for sid in scan_ids:
        f2d, m2d = load_orientation_2d_from_darling(
            h5_path,
            sid,
            method=method,
            roi=roi,
            intensity_threshold=intensity_threshold,
        )
        layers.append((f2d, m2d))

    return stack_layers_into_volume(
        layers,
        spacing=spacing,
        angle_unit=angle_unit,
        source={
            "loader": "darling",
            "h5_path": h5_path,
            "scan_ids": list(scan_ids),
            "method": method,
            "roi": roi,
            "intensity_threshold": intensity_threshold,
        },
    )


def load_orientation_volume_from_h5(
    h5_path: str,
    field_dataset: str,
    *,
    mask_dataset: Optional[str],
    spacing: VoxelSpacing,
    angle_unit: str,
) -> OrientationVolume:
    """Last-resort raw-HDF5 path used when ``darling`` cannot be applied.

    The package recommendation is to *always* use
    :func:`load_orientation_volume_from_darling`. This function exists
    only to support reading already-prepared per-voxel orientation
    arrays (``(Z, Y, X, 2)``) and masks (``(Z, Y, X)``) that someone has
    saved out of an external pipeline and that ``darling.DataSet``
    cannot consume. If you are tempted to extend it, please consider
    writing a darling reader instead.

    Parameters
    ----------
    h5_path
        HDF5 file containing the orientation field and (optionally) the
        mask.
    field_dataset
        HDF5 path to a 4D ``(Z, Y, X, 2)`` float array.
    mask_dataset
        HDF5 path to a 3D ``(Z, Y, X)`` boolean/uint8 array. If
        ``None``, the mask is derived as ``isfinite`` of the field.
    spacing, angle_unit
        Same meaning as in the darling path.
    """
    import h5py  # local import: only this fallback needs h5py directly

    with h5py.File(h5_path, "r") as f:
        if field_dataset not in f:
            raise KeyError(
                f"Dataset {field_dataset!r} not found in {h5_path}. "
                f"Available top-level keys: {list(f.keys())}"
            )
        field = np.asarray(f[field_dataset][...], dtype=np.float32)
        if field.ndim != 4 or field.shape[-1] != 2:
            raise ValueError(
                f"{field_dataset!r} has shape {field.shape}; "
                "expected (Z, Y, X, 2)"
            )
        if mask_dataset is None:
            mask = np.isfinite(field).all(axis=-1)
        else:
            if mask_dataset not in f:
                raise KeyError(
                    f"Mask dataset {mask_dataset!r} not found in {h5_path}"
                )
            mask = np.asarray(f[mask_dataset][...]).astype(bool)
            if mask.shape != field.shape[:3]:
                raise ValueError(
                    f"mask dataset shape {mask.shape} does not match "
                    f"field shape {field.shape[:3]}"
                )
    return OrientationVolume(
        field=field,
        mask=mask,
        spacing=spacing,
        angle_unit=angle_unit,
        source={
            "loader": "raw_hdf5",
            "h5_path": h5_path,
            "field_dataset": field_dataset,
            "mask_dataset": mask_dataset,
        },
    )


# ---------------------------------------------------------------------------
# Optional smoothing
# ---------------------------------------------------------------------------


def median_filter_orientation(
    field: np.ndarray, kernel_zyx: Sequence[int]
) -> np.ndarray:
    """Per-channel median filter on a (Z, Y, X, 2) orientation field.

    Anisotropic kernels are explicitly supported via ``kernel_zyx``,
    e.g. ``(1, 3, 3)`` to leave the z axis untouched (useful when layer
    spacing >> in-plane spacing).
    """
    from scipy.ndimage import median_filter

    if field.ndim != 4 or field.shape[-1] != 2:
        raise ValueError("field must be (Z, Y, X, 2)")
    if len(kernel_zyx) != 3:
        raise ValueError("kernel_zyx must be a 3-tuple")

    out = np.empty_like(field)
    for c in range(field.shape[-1]):
        out[..., c] = median_filter(field[..., c], size=tuple(kernel_zyx))
    return out


# ---------------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------------


def save_h5_volume(
    path: Path,
    array: np.ndarray,
    *,
    dataset: str,
    spacing: VoxelSpacing,
    angle_unit: Optional[str] = None,
    extra_attrs: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a 3D/4D array to HDF5 with axis-order and spacing attributes."""
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset(dataset, data=array, compression="gzip")
        ds.attrs["axis_order"] = SPATIAL_AXIS_ORDER + ("C" if array.ndim == 4 else "")
        ds.attrs["voxel_spacing_nm_zyx"] = np.array(spacing.as_tuple_nm(), float)
        if angle_unit is not None:
            ds.attrs["angle_unit"] = angle_unit
        if extra_attrs:
            for k, v in extra_attrs.items():
                ds.attrs[k] = v


def save_vti(
    path: Path,
    fields: Dict[str, np.ndarray],
    *,
    spacing: VoxelSpacing,
) -> None:
    """Save one or more co-registered 3D scalar arrays to a ParaView ``.vti``.

    All arrays must share the same ``(Z, Y, X)`` shape. Voxel spacing is
    stored as ``(dx, dy, dz)`` in nanometres (the VTK convention is
    ``(x, y, z)``).
    """
    try:
        import vtk
        from vtk.util import numpy_support as vns
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "vtk is required for ParaView .vti output. Install via "
            "`pip install vtk`."
        ) from exc

    if not fields:
        raise ValueError("fields must contain at least one array")

    shapes = {name: arr.shape for name, arr in fields.items()}
    base_shape = next(iter(shapes.values()))
    for name, sh in shapes.items():
        if sh != base_shape:
            raise ValueError(
                f"field {name!r} shape {sh} does not match base shape {base_shape}"
            )
    if len(base_shape) != 3:
        raise ValueError("All fields must be 3D (Z, Y, X)")

    Z, Y, X = base_shape
    image = vtk.vtkImageData()
    image.SetDimensions(X, Y, Z)
    image.SetSpacing(spacing.dx_nm, spacing.dy_nm, spacing.dz_nm)
    image.SetOrigin(0.0, 0.0, 0.0)

    for name, arr in fields.items():
        # VTK image data flattens with X varying fastest. numpy is C-order
        # with Z (axis=0) varying slowest — that already matches.
        flat = np.ascontiguousarray(arr).ravel(order="C")
        vtk_arr = vns.numpy_to_vtk(num_array=flat, deep=True)
        vtk_arr.SetName(name)
        image.GetPointData().AddArray(vtk_arr)

    # Set the first added scalar as the active scalar.
    first_name = next(iter(fields.keys()))
    image.GetPointData().SetActiveScalars(first_name)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(image)
    writer.SetCompressorTypeToZLib()
    writer.Write()


# ---------------------------------------------------------------------------
# Footprint helpers — anisotropy-aware
# ---------------------------------------------------------------------------


def isotropic_physical_footprint(
    spacing: VoxelSpacing, *, radius_nm: float, ndim: int = 3
) -> np.ndarray:
    """Build a boolean footprint that is approximately *isotropic in physical units*.

    For anisotropic voxels — typical of DFXM datasets where ``dz_nm >>
    dy_nm = dx_nm`` — a naive ``np.ones((3, 3, 3))`` footprint would
    treat one z-step like one xy-step. This helper instead returns a
    footprint that includes voxel ``(dz, dy, dx)`` iff the *physical*
    distance between the centre and that voxel is ≤ ``radius_nm``.

    The footprint always contains the centre voxel and is at least
    3 along each axis with the central plane filled in. Returns a
    boolean array of odd-sized dimensions.
    """
    if ndim not in (2, 3):
        raise ValueError("ndim must be 2 or 3")

    spacings = (
        (spacing.dz_nm, spacing.dy_nm, spacing.dx_nm)
        if ndim == 3
        else (spacing.dy_nm, spacing.dx_nm)
    )

    half_extents = [max(1, int(np.ceil(radius_nm / s))) for s in spacings]
    sizes = [2 * h + 1 for h in half_extents]

    coords = np.indices(sizes).astype(float)
    centre = np.array(half_extents).reshape((-1,) + (1,) * ndim)
    deltas = coords - centre
    dists_nm = np.sqrt(
        sum((deltas[i] * spacings[i]) ** 2 for i in range(ndim))
    )
    return (dists_nm <= radius_nm).astype(bool)
