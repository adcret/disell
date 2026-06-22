#!/usr/bin/env python3
"""
Batch segment all 111_june DFXM cell volumes and save 3D/4D-ready outputs.

This script is designed around Adam's working disell notebook workflow:
    1. load layer_*/mean.npy into a 3D volume;
    2. register layers inside each volume;
    3. normalise chi/phi-like feature channels;
    4. segment with disell.flood_fill_dfxm_two_stage;
    5. compute KAM and optional watershed refinement;
    6. split disconnected labels so each final label is one connected component;
    7. compute cell-size and neighbour-misorientation statistics;
    8. save arrays, tables, figures, and metadata for later 4D tracking.

Expected input structure:
    ROOT/
      111_cells_2_6-1pct_mosalayers_2x_redo/
        layer_.../
          mean.npy
      111_cells_2_6-2pct_mosalayers_2x/
        layer_.../
          mean.npy
      ...

Example:
    conda activate main
    cd ~/Documents/Scripts/packages/disell

    python scripts/segment_111_june_all.py \
        --root ~/Documents/Data/4dcells/111_june \
        --output ~/Documents/Data/4dcells/111_june/disell_batch_output \
        --local-threshold 0.015 \
        --global-threshold 0.3 \
        --footprint-tolerance 0.85 \
        --max-iterations 50000 \
        --min-grain-size 4 \
        --no-fill-remaining \
        --stagnation-tolerance 1000 \
        --random-seed -1
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats
from skimage.segmentation import relabel_sequential, watershed

import darling
import disell


DATASET_ORDER = [
    "111_cells_2_6-1pct_mosalayers_2x_redo",
    "111_cells_2_6-2pct_mosalayers_2x",
    "111_cells_2_6-3pct_mosalayers_2x",
    "111_cells_2_6-4pct_mosalayers_2x",
    "111_cells_2_6-5pct_mosalayers_2x",
    "111_cells_2_6-7pct_mosalayers_2x",
]

# Approximate nominal strain labels from folder names. Change if you know the
# calibrated true strains.
DATASET_STRAINS = {
    "111_cells_2_6-1pct_mosalayers_2x_redo": 1.0,
    "111_cells_2_6-2pct_mosalayers_2x": 2.0,
    "111_cells_2_6-3pct_mosalayers_2x": 3.0,
    "111_cells_2_6-4pct_mosalayers_2x": 4.0,
    "111_cells_2_6-5pct_mosalayers_2x": 5.0,
    "111_cells_2_6-7pct_mosalayers_2x": 7.0,
}


@dataclass
class SegmentationConfig:
    local_threshold: float
    global_threshold: float | None
    footprint_tolerance: float
    max_iterations: int
    min_grain_size: int
    stagnation_tolerance: int
    fill_remaining: bool
    random_seed: int | None
    kam_size: tuple[int, int, int]
    split_components: bool
    use_watershed: bool


def parse_float_or_none(value: str) -> float | None:
    if value.lower() in {"none", "null", "nan", "-1"}:
        return None
    return float(value)


def get_layer_number(path: Path) -> int:
    match = re.search(r"layer_(\d+)_", path.name)
    if match is None:
        match = re.search(r"layer[_-]?(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not extract layer number from {path.name!r}")
    return int(match.group(1))


def find_layer_dirs(dataset_dir: Path) -> list[Path]:
    layer_dirs = [
        p for p in dataset_dir.iterdir()
        if p.is_dir() and p.name.startswith("layer_") and (p / "mean.npy").exists()
    ]
    layer_dirs = sorted(layer_dirs, key=get_layer_number)
    if not layer_dirs:
        raise FileNotFoundError(f"No layer_*/mean.npy files found in {dataset_dir}")
    return layer_dirs


def load_volume(dataset_dir: Path) -> tuple[np.ndarray, list[str]]:
    layer_dirs = find_layer_dirs(dataset_dir)
    volume = np.stack([np.load(p / "mean.npy") for p in layer_dirs], axis=0)
    return volume, [p.name for p in layer_dirs]


def make_rgb_volume(volume: np.ndarray) -> np.ndarray:
    phi = np.linspace(np.nanmin(volume[..., 0]), np.nanmax(volume[..., 0]), 64)
    chi = np.linspace(np.nanmin(volume[..., -1]), np.nanmax(volume[..., -1]), 64)
    coord = np.meshgrid(phi, chi, indexing="ij")

    rgb_volume = []
    for z in range(volume.shape[0]):
        rgb, _, _ = darling.transforms.rgb(
            volume[z],
            norm="full",
            coordinates=coord,
        )
        rgb_volume.append(rgb)
    return np.stack(rgb_volume, axis=0)


def register_layers(volume: np.ndarray, registration_channel: int = 0) -> tuple[np.ndarray, list]:
    transforms = disell.register(volume, registration_channel=registration_channel, verbose=False)
    registered = disell.apply_transforms(volume, transforms, pad_value=np.nan)
    return registered, transforms


def normalise_for_segmentation(registered: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mask = np.all(np.isfinite(registered), axis=-1)
    seg_input = registered.copy().astype(np.float32)

    norm_info = {}
    for c in range(seg_input.shape[-1]):
        vals = seg_input[..., c][mask]
        lo, hi = np.nanpercentile(vals, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            raise ValueError(f"Bad normalisation range for channel {c}: {lo}, {hi}")
        seg_input[..., c] = (seg_input[..., c] - lo) / (hi - lo)
        norm_info[f"channel_{c}"] = {"p1": float(lo), "p99": float(hi)}

    seg_input[~np.isfinite(seg_input)] = 0.0
    return seg_input, mask, norm_info


def make_footprint() -> np.ndarray:
    # Same connectivity used in the notebook: dense in-plane footprint with
    # cross-like coupling between adjacent layers.
    footprint = np.ones((3, 3, 3), dtype=bool)
    footprint[1, :, :] = True
    footprint[0, :, 1] = True
    footprint[0, 1, :] = True
    footprint[2, 1, :] = True
    footprint[2, :, 1] = True
    return footprint


def call_two_stage(seg_input: np.ndarray, footprint: np.ndarray, mask: np.ndarray, cfg: SegmentationConfig):
    """Call disell.flood_fill_dfxm_two_stage while supporting both old and new wrappers."""
    kwargs = dict(
        property_map=seg_input,
        footprint=footprint,
        local_misorientation_threshold=cfg.local_threshold,
        global_threshold=cfg.global_threshold,
        footprint_tolerance=cfg.footprint_tolerance,
        mask=mask,
        max_iterations=cfg.max_iterations,
        min_grain_size=cfg.min_grain_size,
        stagnation_tolerance=cfg.stagnation_tolerance,
        random_seed=cfg.random_seed,
    )

    sig = inspect.signature(disell.flood_fill_dfxm_two_stage)
    if "fill_remaining" in sig.parameters:
        kwargs["fill_remaining"] = cfg.fill_remaining
    elif "recycle_small_grains" in sig.parameters:
        # In the corrected C++ backend this boolean is now used as fill_remaining.
        # If the Python wrapper has not been renamed yet, passing True here is
        # still the right thing for a space-filling final pass.
        kwargs["recycle_small_grains"] = cfg.fill_remaining

    return disell.flood_fill_dfxm_two_stage(**kwargs)


def watershed_refine(markers: np.ndarray, seg_input: np.ndarray, mask: np.ndarray, cfg: SegmentationConfig) -> tuple[np.ndarray, np.ndarray]:
    kam_volume = disell.kam(seg_input, ndim=3, size=cfg.kam_size)
    kam_volume = np.asarray(kam_volume)
    if kam_volume.shape == markers.shape + (1,):
        kam_volume = kam_volume[..., 0]
    assert kam_volume.shape == markers.shape

    feature = np.asarray(kam_volume, dtype=np.float32).copy()
    finite_inside = np.isfinite(feature[mask])
    if np.any(finite_inside):
        fill_value = np.nanmax(feature[mask][finite_inside])
    else:
        fill_value = 1.0
    feature[~np.isfinite(feature)] = fill_value

    final_labels = disell.region_grow_watershed(
        seg=markers.astype(np.int32),
        mask=mask.astype(bool),
        feature=feature,
        connectivity=1,
    ).astype(np.int32, copy=False)

    final_labels, _, _ = relabel_sequential(final_labels)
    return np.asarray(final_labels, dtype=np.int32), kam_volume


def split_disconnected_labels(labels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, :, :] = True
    structure[0, 1, 1] = True
    structure[2, 1, 1] = True

    labels_split = np.zeros_like(labels, dtype=np.int32)
    new_label = 1
    n_split_labels = 0
    n_extra_components = 0

    for old_label in np.unique(labels):
        if old_label == 0:
            continue

        cc, n_cc = ndi.label(labels == old_label, structure=structure)
        if n_cc > 1:
            n_split_labels += 1
            n_extra_components += n_cc - 1

        for component_id in range(1, n_cc + 1):
            component = cc == component_id
            labels_split[component] = new_label
            new_label += 1

    coverage_before = np.count_nonzero((labels > 0) & mask) / max(np.count_nonzero(mask), 1)
    coverage_after = np.count_nonzero((labels_split > 0) & mask) / max(np.count_nonzero(mask), 1)

    info = {
        "old_max_label": int(labels.max()),
        "new_max_label": int(labels_split.max()),
        "labels_with_multiple_components": int(n_split_labels),
        "extra_components": int(n_extra_components),
        "coverage_before_split": float(coverage_before),
        "coverage_after_split": float(coverage_after),
    }
    return labels_split, info


def compute_cell_table(labels: np.ndarray, features: np.ndarray, mask: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    label_ids, counts = np.unique(labels[(labels > 0) & mask], return_counts=True)
    n_max = int(label_ids.max()) if len(label_ids) else 0
    medians = np.full((n_max + 1, features.shape[-1]), np.nan, dtype=float)
    means = np.full((n_max + 1, features.shape[-1]), np.nan, dtype=float)

    rows = []
    for lab, count in zip(label_ids, counts):
        region = (labels == lab) & mask
        vals = features[region]
        vals = vals[np.all(np.isfinite(vals), axis=1)]
        if vals.size == 0:
            continue
        med = np.nanmedian(vals, axis=0)
        avg = np.nanmean(vals, axis=0)
        medians[int(lab)] = med
        means[int(lab)] = avg

        centered = vals - med
        spread = np.linalg.norm(centered, axis=1)

        row = {
            "label": int(lab),
            "volume_voxels": int(count),
            "internal_spread_median": float(np.nanmedian(spread)),
            "internal_spread_mean": float(np.nanmean(spread)),
            "internal_spread_q95": float(np.nanpercentile(spread, 95)),
        }
        for c in range(features.shape[-1]):
            row[f"median_c{c}"] = float(med[c])
            row[f"mean_c{c}"] = float(avg[c])
        rows.append(row)

    return pd.DataFrame(rows), medians


def neighbour_offsets() -> list[tuple[int, int, int]]:
    offsets = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            offsets.append((0, dy, dx))
    offsets.append((-1, 0, 0))
    offsets.append((1, 0, 0))

    unique_offsets = []
    seen = set()
    for dz, dy, dx in offsets:
        a = (dz, dy, dx)
        b = (-dz, -dy, -dx)
        if b in seen:
            continue
        seen.add(a)
        unique_offsets.append(a)
    return unique_offsets


def find_neighbour_pairs(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    Z, Y, X = labels.shape
    pairs = set()

    for dz, dy, dx in neighbour_offsets():
        z0a, z1a = max(0, dz), Z + min(0, dz)
        y0a, y1a = max(0, dy), Y + min(0, dy)
        x0a, x1a = max(0, dx), X + min(0, dx)

        z0b, z1b = max(0, -dz), Z + min(0, -dz)
        y0b, y1b = max(0, -dy), Y + min(0, -dy)
        x0b, x1b = max(0, -dx), X + min(0, -dx)

        a = labels[z0a:z1a, y0a:y1a, x0a:x1a]
        b = labels[z0b:z1b, y0b:y1b, x0b:x1b]
        ma = mask[z0a:z1a, y0a:y1a, x0a:x1a]
        mb = mask[z0b:z1b, y0b:y1b, x0b:x1b]

        contact = (a > 0) & (b > 0) & (a != b) & ma & mb
        if not np.any(contact):
            continue

        aa = a[contact].astype(np.int64)
        bb = b[contact].astype(np.int64)
        lo = np.minimum(aa, bb)
        hi = np.maximum(aa, bb)
        for p in zip(lo, hi):
            pairs.add(p)

    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(pairs), dtype=np.int64)


def compute_neighbour_table(labels: np.ndarray, features: np.ndarray, mask: np.ndarray, medians: np.ndarray) -> pd.DataFrame:
    pairs = find_neighbour_pairs(labels, mask)
    rows = []

    for lab_a, lab_b in pairs:
        if lab_a >= medians.shape[0] or lab_b >= medians.shape[0]:
            continue
        fa = medians[lab_a]
        fb = medians[lab_b]
        if not (np.all(np.isfinite(fa)) and np.all(np.isfinite(fb))):
            continue
        dvec = fb - fa
        row = {
            "label_a": int(lab_a),
            "label_b": int(lab_b),
            "misorientation": float(np.linalg.norm(dvec)),
        }
        for c in range(features.shape[-1]):
            row[f"delta_c{c}"] = float(dvec[c])
        rows.append(row)

    return pd.DataFrame(rows)


def fit_chi_distribution(mis: np.ndarray) -> dict:
    mis = np.asarray(mis, dtype=float)
    mis = mis[np.isfinite(mis)]
    mis = mis[mis > 0]
    if len(mis) < 10:
        return {
            "chi_k": np.nan,
            "chi_sigma": np.nan,
            "chi_mean": np.nan,
            "empirical_mean": float(np.nanmean(mis)) if len(mis) else np.nan,
            "empirical_median": float(np.nanmedian(mis)) if len(mis) else np.nan,
            "n": int(len(mis)),
        }
    k_hat, loc_hat, sigma_hat = stats.chi.fit(mis, floc=0)
    return {
        "chi_k": float(k_hat),
        "chi_sigma": float(sigma_hat),
        "chi_mean": float(stats.chi.mean(df=k_hat, loc=0, scale=sigma_hat)),
        "chi_std": float(stats.chi.std(df=k_hat, loc=0, scale=sigma_hat)),
        "empirical_mean": float(np.mean(mis)),
        "empirical_median": float(np.median(mis)),
        "empirical_std": float(np.std(mis)),
        "n": int(len(mis)),
    }


def plot_quicklook(out_dir: Path, name: str, labels: np.ndarray, rgb: np.ndarray, mask: np.ndarray, z: int | None = None) -> None:
    if z is None:
        z = labels.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb[z], aspect=2.8)
    axes[0].set_title(f"{name}: RGB, layer {z}")
    axes[0].axis("off")

    axes[1].imshow(labels[z], cmap="nipy_spectral", interpolation="nearest", aspect=2.8)
    axes[1].set_title("labels")
    axes[1].axis("off")

    unlabelled = (labels[z] == 0) & mask[z]
    axes[2].imshow(unlabelled, cmap="gray", interpolation="nearest", aspect=2.8)
    axes[2].set_title("unlabelled inside mask")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_quicklook.png", dpi=200)
    plt.close(fig)


def process_dataset(dataset_dir: Path, output_root: Path, cfg: SegmentationConfig, registration_channel: int = 0) -> dict:
    name = dataset_dir.name
    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {name} ===", flush=True)
    volume, layer_names = load_volume(dataset_dir)
    print("loaded volume:", volume.shape, flush=True)

    registered, layer_transforms = register_layers(volume, registration_channel=registration_channel)
    rgb_volume = make_rgb_volume(registered)
    seg_input, mask, norm_info = normalise_for_segmentation(registered)
    mask = mask & np.all(np.isfinite(registered), axis=-1)

    footprint = make_footprint()

    marker_result, sizes_initial = call_two_stage(seg_input, footprint, mask.astype(np.uint8), cfg)
    markers = np.asarray(marker_result["segmentation"], dtype=np.int32)
    mask_bool = mask.astype(bool)

    if cfg.use_watershed:
        labels, kam_volume = watershed_refine(markers, seg_input, mask_bool, cfg)
    else:
        labels = markers.copy()
        kam_volume = disell.kam(seg_input, ndim=3, size=cfg.kam_size)
        if kam_volume.shape == markers.shape + (1,):
            kam_volume = kam_volume[..., 0]

    labels, _, _ = relabel_sequential(labels)
    labels = np.asarray(labels, dtype=np.int32)

    split_info = {}
    if cfg.split_components:
        labels, split_info = split_disconnected_labels(labels, mask_bool)

    coverage_markers = np.count_nonzero((markers > 0) & mask_bool) / max(np.count_nonzero(mask_bool), 1)
    coverage_labels = np.count_nonzero((labels > 0) & mask_bool) / max(np.count_nonzero(mask_bool), 1)

    cell_df, cell_medians = compute_cell_table(labels, seg_input, mask_bool)
    neighbour_df = compute_neighbour_table(labels, seg_input, mask_bool, cell_medians)
    chi_info = fit_chi_distribution(neighbour_df["misorientation"].to_numpy() if len(neighbour_df) else np.array([]))

    # Save arrays.
    np.save(out_dir / "raw_volume.npy", volume)
    np.save(out_dir / "registered_volume.npy", registered)
    np.save(out_dir / "seg_input.npy", seg_input)
    np.save(out_dir / "mask.npy", mask_bool)
    np.save(out_dir / "rgb_volume.npy", rgb_volume)
    np.save(out_dir / "markers.npy", markers)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "kam_volume.npy", kam_volume)
    np.save(out_dir / "sizes_initial.npy", sizes_initial)

    cell_df.to_csv(out_dir / "cell_table.csv", index=False)
    neighbour_df.to_csv(out_dir / "neighbour_table.csv", index=False)

    summary = {
        "dataset": name,
        "nominal_strain_percent": DATASET_STRAINS.get(name, None),
        "input_path": str(dataset_dir),
        "output_path": str(out_dir),
        "layer_names": layer_names,
        "volume_shape": list(volume.shape),
        "registered_shape": list(registered.shape),
        "n_valid_voxels": int(np.count_nonzero(mask_bool)),
        "n_marker_labels": int(markers.max()),
        "n_final_labels": int(labels.max()),
        "marker_coverage_inside_mask": float(coverage_markers),
        "final_coverage_inside_mask": float(coverage_labels),
        "n_initial_seeds": int(len(sizes_initial)),
        "n_cells_table": int(len(cell_df)),
        "n_neighbour_pairs": int(len(neighbour_df)),
        "chi_fit": chi_info,
        "split_info": split_info,
        "normalisation": norm_info,
        "config": asdict(cfg),
        "layer_registration_transforms": [
            None if t is None else [float(v) for v in np.asarray(t).ravel()]
            for t in layer_transforms
        ],
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_quicklook(out_dir, name, labels, rgb_volume, mask_bool)

    print(f"saved to: {out_dir}", flush=True)
    print(f"coverage final: {coverage_labels:.4%}", flush=True)
    print(f"labels: {labels.max()}, neighbour pairs: {len(neighbour_df)}", flush=True)
    print(f"chi fit: k={chi_info.get('chi_k'):.4f}, sigma={chi_info.get('chi_sigma'):.4f}", flush=True)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("~/Documents/Data/4dcells/111_june").expanduser())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=DATASET_ORDER)
    parser.add_argument("--registration-channel", type=int, default=0)

    parser.add_argument("--local-threshold", type=float, default=0.015)
    parser.add_argument("--global-threshold", type=parse_float_or_none, default=0.3)
    parser.add_argument("--footprint-tolerance", type=float, default=0.85)
    parser.add_argument("--max-iterations", type=int, default=50_000)
    parser.add_argument("--min-grain-size", type=int, default=4)
    parser.add_argument("--stagnation-tolerance", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=-1)
    parser.add_argument("--kam-size", type=int, nargs=3, default=(3, 3, 3))
    parser.add_argument("--no-watershed", action="store_true")
    parser.add_argument("--no-split-components", action="store_true")
    parser.add_argument("--fill-remaining", action="store_true", default=False)
    parser.add_argument("--no-fill-remaining", dest="fill_remaining", action="store_false")

    args = parser.parse_args(argv)

    root = args.root.expanduser()
    if args.output is None:
        output_root = root / "disell_batch_output"
    else:
        output_root = args.output.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = SegmentationConfig(
        local_threshold=args.local_threshold,
        global_threshold=args.global_threshold,
        footprint_tolerance=args.footprint_tolerance,
        max_iterations=args.max_iterations,
        min_grain_size=args.min_grain_size,
        stagnation_tolerance=args.stagnation_tolerance,
        fill_remaining=bool(args.fill_remaining),
        random_seed=args.random_seed,
        kam_size=tuple(args.kam_size),
        split_components=not args.no_split_components,
        use_watershed=not args.no_watershed,
    )

    summaries = []
    failures = []

    for dataset in args.datasets:
        dataset_dir = root / dataset
        if not dataset_dir.exists():
            print(f"[SKIP] Missing dataset: {dataset_dir}", file=sys.stderr)
            failures.append({"dataset": dataset, "error": "missing"})
            continue
        try:
            summaries.append(process_dataset(dataset_dir, output_root, cfg, registration_channel=args.registration_channel))
        except Exception as exc:
            print(f"[FAILED] {dataset}: {exc}", file=sys.stderr)
            failures.append({"dataset": dataset, "error": repr(exc)})

    summary_df = pd.DataFrame([
        {
            "dataset": s["dataset"],
            "nominal_strain_percent": s["nominal_strain_percent"],
            "n_valid_voxels": s["n_valid_voxels"],
            "n_marker_labels": s["n_marker_labels"],
            "n_final_labels": s["n_final_labels"],
            "marker_coverage_inside_mask": s["marker_coverage_inside_mask"],
            "final_coverage_inside_mask": s["final_coverage_inside_mask"],
            "n_neighbour_pairs": s["n_neighbour_pairs"],
            "chi_k": s["chi_fit"].get("chi_k"),
            "chi_sigma": s["chi_fit"].get("chi_sigma"),
            "mean_misorientation": s["chi_fit"].get("empirical_mean"),
            "median_misorientation": s["chi_fit"].get("empirical_median"),
        }
        for s in summaries
    ])
    summary_df.to_csv(output_root / "batch_summary.csv", index=False)

    with open(output_root / "batch_summary.json", "w") as f:
        json.dump({"summaries": summaries, "failures": failures}, f, indent=2)

    print("\n=== batch complete ===")
    print("output:", output_root)
    print(summary_df)
    if failures:
        print("failures:", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
