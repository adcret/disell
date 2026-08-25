#!/usr/bin/env python3
"""Verification harness for global_threshold flood-fill parameter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1] / "src"
OUT_DIR = Path(__file__).resolve().parents[1] / "verification_output"
DATA_DIR = Path.home() / (
    "Documents/Data/4dcells/111_june/"
    "111_cells_2_6-1pct_mosalayers_2x_redo/disell_3d_output"
)

RANDOM_SEED = 42
MAX_SEED_ATTEMPTS = 5000
FOOTPRINT_TOLERANCE = 0.85
MIN_GRAIN_SIZE = 20
MAX_ITERATIONS = 5000
STAGNATION_TOLERANCE = 2000
GLOBAL_SWEEP = [2.0, 1.2, 0.8, 0.5, 0.3]
SLICE_Z = 5

# Baseline equivalence: normalized map used in the saved notebook pipeline.
BASELINE_FIELD = "seg_input.npy"
BASELINE_LOCAL = 0.02

# Global sweep: registered orientation map where intra-cell spread reaches ~0.42
# so thresholds in [0.3, 2.0] bind progressively.
SWEEP_FIELD = "volume_registered.npy"
SWEEP_LOCAL = 0.06


def _load_module(name: str, rel_path: str):
    path = _PKG_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_flood_fill():
    import glob
    import importlib.util as iu

    candidates = glob.glob(
        str(_PKG_ROOT / "disell" / "_flood_fill*.so")
    )
    candidates += glob.glob(
        "/home/adam/miniconda3/lib/python3.13/site-packages/disell/_flood_fill*.so"
    )
    if not candidates:
        raise ImportError("Could not locate compiled _flood_fill extension")
    spec = iu.spec_from_file_location("_flood_fill", candidates[0])
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.flood_fill_random_seeds_3d


flood_fill_random_seeds_3d = _load_flood_fill()
_props = _load_module("disell.properties", "disell/properties.py")
sys.modules["disell.properties"] = _props
sys.modules.setdefault("disell", type(sys)("disell"))
_stats = _load_module("disell.cell_statistics", "disell/cell_statistics.py")
cell_stats_orientation_based = _stats.cell_stats_orientation_based


def build_footprint() -> np.ndarray:
    footprint = np.zeros((3, 3, 3), dtype=bool)
    footprint[1, :, :] = True
    footprint[0, 1, 1] = True
    footprint[2, 1, 1] = True
    return footprint


def pad_for_cpp(field: np.ndarray, mask: np.ndarray, footprint: np.ndarray):
    half = tuple(s // 2 for s in footprint.shape)
    pad = ((half[0],) * 2, (half[1],) * 2, (half[2],) * 2)
    field_p = np.pad(
        np.ascontiguousarray(field, dtype=np.float32),
        pad + ((0, 0),),
        mode="constant",
        constant_values=0.0,
    )
    mask_p = np.pad(
        np.ascontiguousarray(mask.astype(np.uint8)),
        pad,
        mode="constant",
        constant_values=0,
    )
    return field_p, mask_p, half


def sample_seeds(mask: np.ndarray, n_seeds: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.flatnonzero(mask.ravel())
    chosen = rng.choice(flat, size=min(n_seeds, flat.size), replace=False)
    z = chosen // (mask.shape[1] * mask.shape[2])
    y = (chosen // mask.shape[2]) % mask.shape[1]
    x = chosen % mask.shape[2]
    return np.stack([z, y, x], axis=-1).astype(np.int64)


def run_segmentation(
    field: np.ndarray,
    mask: np.ndarray,
    footprint: np.ndarray,
    *,
    local_threshold: float,
    global_threshold,
    seeds: np.ndarray,
) -> np.ndarray:
    field_p, mask_p, half = pad_for_cpp(field, mask, footprint)
    g_thr = -1.0 if global_threshold is None else float(global_threshold)

    result = flood_fill_random_seeds_3d(
        field_p,
        footprint.astype(bool),
        float(local_threshold),
        g_thr,
        float(FOOTPRINT_TOLERANCE),
        mask_p.copy(),
        int(MAX_ITERATIONS),
        int(MIN_GRAIN_SIZE),
        False,
        int(STAGNATION_TOLERANCE),
        seeds,
    )
    seg_p = np.asarray(result["segmentation"], dtype=np.int32)
    sz, sy, sx = half
    if any(h > 0 for h in half):
        seg_p = seg_p[
            sz : seg_p.shape[0] - sz,
            sy : seg_p.shape[1] - sy,
            sx : seg_p.shape[2] - sx,
        ]
    return seg_p.copy()


def summarize(seg: np.ndarray, mask: np.ndarray, field: np.ndarray) -> dict:
    stats, _ = cell_stats_orientation_based(seg, field)
    q95 = [v["q95"] for v in stats.values()]
    _, sizes = np.unique(seg[seg > 0], return_counts=True)

    claimed = int(np.count_nonzero((seg > 0) & mask))
    mask_count = int(np.count_nonzero(mask))

    return {
        "n_labels": int(seg.max()),
        "size_mean": float(sizes.mean()) if sizes.size else 0.0,
        "size_std": float(sizes.std()) if sizes.size else 0.0,
        "claimed_voxels": claimed,
        "claimed_fraction": claimed / mask_count if mask_count else 0.0,
        "mean_q95_spread": float(np.mean(q95)) if q95 else float("nan"),
        "max_q95_spread": float(np.max(q95)) if q95 else float("nan"),
    }


def label_rgb(seg_slice: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = int(seg_slice.max())
    colors = rng.random((n + 1, 3))
    colors[0] = 0.0
    return colors[seg_slice]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mask = np.load(DATA_DIR / "mask.npy")
    footprint = build_footprint()
    field_p, mask_p, _ = pad_for_cpp(
        np.load(DATA_DIR / SWEEP_FIELD), mask, footprint
    )
    seeds = sample_seeds(
        mask_p.astype(bool), MAX_SEED_ATTEMPTS, np.random.default_rng(RANDOM_SEED)
    )

    print("Data inspection")
    for name in [BASELINE_FIELD, SWEEP_FIELD, "mask.npy", "labels_raw.npy"]:
        p = DATA_DIR / name
        a = np.load(p, mmap_mode="r")
        print(f"  {name}: shape={a.shape}, dtype={a.dtype}")

    # --- Step 2: baseline equivalence on seg_input ---
    seg_field = np.load(DATA_DIR / BASELINE_FIELD)
    old_seg_path = OUT_DIR / "old_baseline" / "labels_baseline_old_cpp.npy"
    if not old_seg_path.exists():
        raise FileNotFoundError(f"missing old baseline labels at {old_seg_path}")

    old_seg = np.load(old_seg_path)
    new_baseline_seg = run_segmentation(
        seg_field,
        mask,
        footprint,
        local_threshold=BASELINE_LOCAL,
        global_threshold=None,
        seeds=seeds,
    )
    np.save(OUT_DIR / "labels_baseline_new.npy", new_baseline_seg)
    baseline_summary = summarize(new_baseline_seg, mask, seg_field)
    old_summary = summarize(old_seg, mask, seg_field)

    print("\nBaseline equivalence (old C++ vs new, global disabled, seg_input):")
    print(f"  labels identical: {np.array_equal(old_seg, new_baseline_seg)}")
    for key in ("n_labels", "size_mean", "size_std", "claimed_voxels", "claimed_fraction"):
        print(f"  {key}: old={old_summary[key]} new={baseline_summary[key]}")

    # --- Step 3: global threshold sweep on volume_registered ---
    sweep_field = np.load(DATA_DIR / SWEEP_FIELD)
    results = {"baseline_disabled": summarize(
        run_segmentation(
            sweep_field, mask, footprint,
            local_threshold=SWEEP_LOCAL,
            global_threshold=None,
            seeds=seeds,
        ),
        mask,
        sweep_field,
    )}
    segs = {}

    baseline_seg = run_segmentation(
        sweep_field, mask, footprint,
        local_threshold=SWEEP_LOCAL,
        global_threshold=None,
        seeds=seeds,
    )
    segs["baseline"] = baseline_seg
    np.save(OUT_DIR / "labels_sweep_baseline.npy", baseline_seg)

    print(f"\nGlobal threshold sweep ({SWEEP_FIELD}, local={SWEEP_LOCAL}):")
    print(f"  baseline: {results['baseline_disabled']}")

    for gt in GLOBAL_SWEEP:
        key = f"global_{gt:g}"
        seg = run_segmentation(
            sweep_field, mask, footprint,
            local_threshold=SWEEP_LOCAL,
            global_threshold=gt,
            seeds=seeds,
        )
        segs[key] = seg
        results[key] = summarize(seg, mask, sweep_field)
        np.save(OUT_DIR / f"labels_{key}.npy", seg)
        print(f"  {key}: {results[key]}")

    max_spreads = [results[f"global_{gt:g}"]["max_q95_spread"] for gt in GLOBAL_SWEEP]
    monotonic = all(
        max_spreads[i] >= max_spreads[i + 1] for i in range(len(max_spreads) - 1)
    )
    print(f"\nMax q95 spread (loose -> tight): {max_spreads}")
    print(f"  monotonic non-increasing: {monotonic}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ("baseline", "baseline (global disabled)"),
        ("global_0.8", "global_threshold=0.8"),
        ("global_0.5", "global_threshold=0.5"),
    ]
    for ax, (key, title) in zip(axes, panels):
        rgb = label_rgb(segs[key][SLICE_Z])
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_title(f"{title}\nz={SLICE_Z}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    png_path = OUT_DIR / "comparison_slice.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = {
        "data_dir": str(DATA_DIR),
        "baseline_equivalence": {
            "field": BASELINE_FIELD,
            "local_threshold": BASELINE_LOCAL,
            "labels_identical": bool(np.array_equal(old_seg, new_baseline_seg)),
            "old_summary": old_summary,
            "new_summary": baseline_summary,
        },
        "global_sweep": {
            "field": SWEEP_FIELD,
            "local_threshold": SWEEP_LOCAL,
            "global_values": GLOBAL_SWEEP,
            "results": results,
            "max_spread_monotonic": monotonic,
            "max_spreads": max_spreads,
        },
        "parameters": {
            "footprint_tolerance": FOOTPRINT_TOLERANCE,
            "min_grain_size": MIN_GRAIN_SIZE,
            "max_iterations": MAX_ITERATIONS,
            "stagnation_tolerance": STAGNATION_TOLERANCE,
            "random_seed": RANDOM_SEED,
            "max_seed_attempts": MAX_SEED_ATTEMPTS,
        },
        "png": str(png_path),
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    return 0 if monotonic and np.array_equal(old_seg, new_baseline_seg) else 1


if __name__ == "__main__":
    raise SystemExit(main())
