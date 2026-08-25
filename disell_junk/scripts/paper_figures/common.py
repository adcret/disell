"""Shared plumbing for the 6.2% paper pipeline scripts.

Everything dataset-specific comes from a JSON config (see
``config_6_2pct.json``); nothing here hard-codes a data path. Outputs go to
``<package root>/paper_outputs/<dataset_name>/<analysis>/`` by default.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pf_io import (  # noqa: E402
    VoxelSpacing,
    isotropic_physical_footprint,
    save_h5_volume,
    save_vti,
    write_parameters_json,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Config and provenance
# ---------------------------------------------------------------------------


def load_config(path: Path | str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = json.load(f)
    cfg["data_root"] = str(Path(cfg["data_root"]).expanduser())
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def out_dir_for(cfg: Dict[str, Any], analysis: str,
                out_root: Optional[str] = None) -> Path:
    root = Path(out_root) if out_root else PACKAGE_ROOT / "paper_outputs" / cfg["dataset_name"]
    d = root / analysis
    d.mkdir(parents=True, exist_ok=True)
    return d


def provenance() -> Dict[str, Any]:
    import scipy
    import skimage

    import disell

    def _git(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=PACKAGE_ROOT, capture_output=True,
                text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            return "unavailable"

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "skimage": skimage.__version__,
            "disell": getattr(disell, "__version__", "editable"),
        },
    }


def spacing_from_config(cfg: Dict[str, Any]) -> VoxelSpacing:
    dz, dy, dx = cfg["spacing_nm_zyx"]
    return VoxelSpacing(dz_nm=dz, dy_nm=dy, dx_nm=dx)


# ---------------------------------------------------------------------------
# Registered-volume I/O (single interchange format for every method)
# ---------------------------------------------------------------------------


def save_registered_volume(path: Path, *, field: np.ndarray, mask: np.ndarray,
                           transforms, cfg: Dict[str, Any]) -> None:
    import h5py

    spacing = spacing_from_config(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tr = np.array([
        [np.nan, np.nan] if t is None else [float(t[0]), float(t[1])]
        for t in transforms
    ])
    with h5py.File(path, "w") as f:
        d = f.create_dataset("field", data=field.astype(np.float32), compression="gzip")
        d.attrs["axis_order"] = "ZYXC"
        d.attrs["channel_names"] = list(cfg["channel_names"])
        d.attrs["angle_unit"] = cfg["angle_unit"]
        d.attrs["voxel_spacing_nm_zyx"] = np.array(spacing.as_tuple_nm())
        m = f.create_dataset("mask", data=mask.astype(np.uint8), compression="gzip")
        m.attrs["axis_order"] = "ZYX"
        t = f.create_dataset("transforms_zyx", data=tr)
        t.attrs["note"] = "per-layer (row, col) shift applied; NaN row = reference layer"


def load_registered_volume(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    import h5py

    with h5py.File(path, "r") as f:
        field = f["field"][...]
        mask = f["mask"][...].astype(bool)
        transforms = f["transforms_zyx"][...]
        attrs = {k: f["field"].attrs[k] for k in f["field"].attrs}
    return field, mask, transforms, attrs


# ---------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------


def flood_footprint(cfg: Dict[str, Any]) -> np.ndarray:
    """Neighbourhood footprint for the flood fill, from config.

    ``inplane8_plus_z``: full 3x3 in-plane neighbourhood plus the two face
    neighbours along z — the anisotropy-aware choice used throughout: in-plane
    steps are 0.4/1.24 um, the z step 0.5 um, so full 26-connectivity would
    treat 1.9-um diagonal jumps like 0.4-um steps.
    """
    kind = cfg["segmentation"]["flood_footprint"]
    if kind == "inplane8_plus_z":
        fp = np.zeros((3, 3, 3), dtype=bool)
        fp[1] = True
        fp[0, 1, 1] = fp[2, 1, 1] = True
        return fp
    if kind == "full26":
        return np.ones((3, 3, 3), dtype=bool)
    if kind == "faces6":
        fp = np.zeros((3, 3, 3), dtype=bool)
        fp[1, 1, 1] = True
        fp[0, 1, 1] = fp[2, 1, 1] = True
        fp[1, 0, 1] = fp[1, 2, 1] = True
        fp[1, 1, 0] = fp[1, 1, 2] = True
        return fp
    raise ValueError(f"unknown flood_footprint {kind!r}")


def kam_footprint(cfg: Dict[str, Any], ndim: int = 3) -> np.ndarray:
    """Physically isotropic KAM footprint from the configured radius."""
    spacing = spacing_from_config(cfg)
    radius = float(cfg["segmentation"]["kam_radius_nm"])
    return isotropic_physical_footprint(spacing, radius_nm=radius, ndim=ndim)


# ---------------------------------------------------------------------------
# KAM with border padding (defined value wherever a valid neighbour exists)
# ---------------------------------------------------------------------------


def masked_kam(field: np.ndarray, mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Manuscript KAM (per-channel RMS) on the masked field.

    Out-of-mask voxels are set to NaN before the computation and the volume is
    padded with NaN by the kernel half-width, so border voxels get a KAM from
    their available valid neighbours instead of being undefined. Returns NaN
    where the centre is invalid or no valid neighbour exists.
    """
    from disell import kam

    ndim = mask.ndim
    f = field.astype(np.float64).copy()
    f[~mask] = np.nan
    half = [s // 2 for s in footprint.shape]
    pad = [(h, h) for h in half] + [(0, 0)]
    f_pad = np.pad(f, pad, constant_values=np.nan)
    k = kam(f_pad, ndim=ndim, footprint=footprint, fill_invalid=np.nan,
            per_channel_rms=True)
    sl = tuple(slice(h, k.shape[i] - h if h else None) for i, h in enumerate(half))
    return k[sl]


def watershed_feature_from_kam(kam_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Finite watershed elevation: NaN inside the mask becomes the in-mask
    maximum (claimed last), outside-mask values are irrelevant (masked)."""
    feat = kam_map.astype(np.float32).copy()
    inside = mask & ~np.isfinite(feat)
    finite_max = np.nanmax(feat[mask]) if np.isfinite(feat[mask]).any() else 1.0
    feat[inside] = finite_max
    feat[~mask] = finite_max
    return feat


__all__ = [
    "PACKAGE_ROOT",
    "VoxelSpacing",
    "load_config",
    "out_dir_for",
    "provenance",
    "spacing_from_config",
    "save_registered_volume",
    "load_registered_volume",
    "flood_footprint",
    "kam_footprint",
    "masked_kam",
    "watershed_feature_from_kam",
    "save_h5_volume",
    "save_vti",
    "write_parameters_json",
    "isotropic_physical_footprint",
]
