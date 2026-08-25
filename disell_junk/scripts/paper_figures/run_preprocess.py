#!/usr/bin/env python
"""Stage 1: build the registered, preprocessed 6.2% feature volume.

Loads the 11-layer dataset, refines the inter-layer registration with
sub-pixel phase correlation, applies the shifts with NaN padding, median
filters in-plane, and stores a single HDF5 volume that every downstream
method consumes. Also writes registration diagnostics.

Usage:
    python run_preprocess.py --config config_6_2pct.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from common import (
    load_config,
    out_dir_for,
    provenance,
    save_registered_volume,
    write_parameters_json,
)

import disell


def nearest_fill(channel: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid voxels by their nearest valid neighbour (per z-slice)."""
    out = channel.copy()
    for z in range(channel.shape[0]):
        inv = ~valid[z]
        if inv.any() and valid[z].any():
            idx = ndimage.distance_transform_edt(
                inv, return_distances=False, return_indices=True
            )
            out[z][inv] = channel[z][tuple(i[inv] for i in idx)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = out_dir_for(cfg, "volume", args.out_root)
    pp = cfg["preprocess"]

    vol = disell.load_layer_volume(
        cfg["data_root"],
        spacing_nm=cfg["spacing_nm_zyx"],
        angle_unit=cfg["angle_unit"],
        channel_names=cfg["channel_names"],
        crop=cfg["crop_rows_cols"],
        pattern=cfg["layer_pattern"],
        expect_n_layers=cfg["expect_n_layers"],
        expect_shape_yx=tuple(cfg["expect_shape_yx"]),
        load_segmaps=True,
    )
    print(f"loaded {vol.field.shape} from {len(vol.layer_names)} layers; "
          f"{vol.source['n_invalid_voxels']} invalid voxels masked")

    field = vol.field.astype(np.float32)
    mask0 = vol.mask.copy()

    # --- residual inter-layer registration --------------------------------
    reg_input = field.copy()
    reg_input[~mask0] = np.nan
    transforms = disell.register(
        reg_input,
        registration_channel=pp["registration_channel"],
        upsample_factor=pp["registration_upsample"],
        verbose=True,
    )
    field_nan = field.copy()
    field_nan[~mask0] = np.nan
    registered = disell.apply_transforms(field_nan, transforms)

    # shift the validity mask with the same transforms
    mask_f = disell.apply_transforms(
        mask0.astype(np.float32)[..., None], transforms
    )[..., 0]
    mask = np.isfinite(mask_f) & (mask_f >= 0.999)
    mask &= np.isfinite(registered).all(axis=-1)

    # --- in-plane median filter (manuscript preprocessing) -----------------
    kz, ky, kx = pp["median_filter_kernel_zyx"]
    filtered = np.empty_like(registered)
    for c in range(registered.shape[-1]):
        filled = nearest_fill(registered[..., c], mask)
        filtered[..., c] = ndimage.median_filter(filled, size=(kz, ky, kx))
    filtered[~mask] = np.nan

    save_registered_volume(out / "volume_registered.h5", field=filtered,
                           mask=mask, transforms=transforms, cfg=cfg)
    save_registered_volume(out / "volume_registered_unfiltered.h5",
                           field=registered, mask=mask,
                           transforms=transforms, cfg=cfg)

    # --- diagnostics -------------------------------------------------------
    Z = field.shape[0]
    shifts = np.array([[0.0, 0.0] if t is None else t for t in transforms])

    def adjacent_diff(f):
        d = []
        for z in range(Z - 1):
            a, b = f[z, ..., 0], f[z + 1, ..., 0]
            ok = np.isfinite(a) & np.isfinite(b)
            d.append(np.abs(a[ok] - b[ok]).mean())
        return d

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(range(Z), shifts[:, 0], "o-", label="rows (y)")
    axes[0].plot(range(Z), shifts[:, 1], "s-", label="cols (x)")
    axes[0].set_xlabel("layer index")
    axes[0].set_ylabel("applied shift (px)")
    axes[0].set_title("residual registration shifts")
    axes[0].legend()
    axes[1].plot(adjacent_diff(field_nan), "o-", label="before")
    axes[1].plot(adjacent_diff(registered), "s-", label="after")
    axes[1].set_xlabel("layer pair")
    axes[1].set_ylabel(r"mean $|\Delta\chi|$ (deg)")
    axes[1].set_title("adjacent-layer difference")
    axes[1].legend()
    dy_um, dx_um = cfg["spacing_nm_zyx"][1] / 1e3, cfg["spacing_nm_zyx"][2] / 1e3
    ny, nx = filtered.shape[1:3]
    im = axes[2].imshow(
        filtered[Z // 2, ..., 0], cmap="viridis", origin="upper",
        extent=[0, nx * dx_um, ny * dy_um, 0],
    )
    axes[2].set_title(f"registered+filtered chi, z={Z//2}")
    axes[2].set_xlabel("x (um)")
    axes[2].set_ylabel("y (um)")
    fig.colorbar(im, ax=axes[2], label="chi (deg)")
    fig.tight_layout()
    fig.savefig(out / "registration_diagnostics.png", dpi=200)
    plt.close(fig)

    write_parameters_json(out, {
        "stage": "preprocess",
        "config": cfg,
        "loader_source": vol.source,
        "layer_names": vol.layer_names,
        "motor_ranges": vol.channel_ranges(),
        "registration_transforms_zyx": [
            None if t is None else [float(v) for v in t] for t in transforms
        ],
        "mask_voxels": int(mask.sum()),
        "mask_fraction": float(mask.mean()),
        "provenance": provenance(),
    })
    print(f"saved registered volume to {out}; valid fraction {mask.mean():.4f}")


if __name__ == "__main__":
    main()
