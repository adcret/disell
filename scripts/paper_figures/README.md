# Paper-figure scripts

Runnable scripts that produce the figures and quantitative comparisons
discussed in `MANUSCRIPT_CRITIQUE.md`. They use `darling` for DFXM
loading and `disell` for segmentation. Read `AUDIT.md` first — these
scripts deliberately route around several known defects in the
`disell` package.

## Files

| File                              | Purpose                                                                                                |
|-----------------------------------|--------------------------------------------------------------------------------------------------------|
| `_io.py`                          | Shared loading / saving helpers (darling-first, anisotropy-aware, deterministic).                      |
| `segment_3d_cells.py`             | Multi-seed flood-fill + watershed pipeline, produces `labels_3d.h5`, KAM, markers, VTI, slice overlays. |
| `segment_3d_kam_baseline.py`      | KAM-thresholding baseline (the manuscript's "fails in 3D" path), produces labels + CSV stats + VTI.    |
| `compare_3d_methods.py`           | Quantitative comparison of the two methods on the *same* volume.                                       |
| `make_3d_paper_figure.py`         | Publication-quality 2D slice + ParaView-ready 3D export.                                               |

All four scripts:

* are deterministic when a `--random-seed` is provided (only
  `segment_3d_cells.py` involves randomness);
* fail loudly with a useful message if a required dataset is missing;
* expose every parameter at the CLI;
* dump every parameter into a `parameters_used.json` next to the
  output;
* preserve voxel spacing in every HDF5 / VTI export;
* keep the spatial axis order as `(Z, Y, X)`.

## Required Python dependencies

Beyond `disell` itself:

```
darling           # https://github.com/AxelHenningsson/darling
h5py
numpy
scipy
scikit-image
matplotlib
vtk               # for ParaView .vti output
```

## Quick reference

A full pipeline using the same volume for all four scripts looks like
this. Replace dataset paths, scan ids and units with values valid for
your own data — these are *not* defaults.

```bash
# 1. Multi-seed flood fill + watershed
python segment_3d_cells.py \
    --input-h5  /data/al1050_pct5.h5 \
    --scan-ids  1.1 2.1 3.1 4.1 5.1 \
    --orientation-method mean \
    --voxel-spacing-nm 400 50 50 \
    --angle-unit deg \
    --kam-kernel-zyx 3 5 5 \
    --ff-kernel-zyx 3 3 3 \
    --local-threshold 0.005 \
    --footprint-tolerance 0.85 \
    --min-cell-size 200 \
    --max-seed-attempts 8000 \
    --stagnation-tolerance 1500 \
    --random-seed 42 \
    --out-dir runs/al1050_pct5/method

# 2. KAM-thresholding baseline at the 70th percentile
python segment_3d_kam_baseline.py \
    --input-h5  /data/al1050_pct5.h5 \
    --scan-ids  1.1 2.1 3.1 4.1 5.1 \
    --orientation-method mean \
    --voxel-spacing-nm 400 50 50 \
    --angle-unit deg \
    --kam-kernel-zyx 3 5 5 \
    --threshold-percentile 70 \
    --erosion-iterations 1 --dilation-iterations 1 \
    --connectivity 1 \
    --min-component-size 200 \
    --out-dir runs/al1050_pct5/kam_p70

# 3. Quantitative comparison
python compare_3d_methods.py \
    --input-h5  /data/al1050_pct5.h5 \
    --scan-ids  1.1 2.1 3.1 4.1 5.1 \
    --voxel-spacing-nm 400 50 50 \
    --angle-unit deg \
    --floodfill-labels-h5 runs/al1050_pct5/method/labels_3d.h5 \
    --floodfill-kam-h5    runs/al1050_pct5/method/kam_3d.h5 \
    --kam-baseline-labels-h5 runs/al1050_pct5/kam_p70/kam_threshold_labels_3d.h5 \
    --min-cell-size 200 \
    --exclude-edge-touching \
    --lognormal-fit \
    --out-dir runs/al1050_pct5/comparison

# 4. Paper figure + ParaView-ready 3D export
python make_3d_paper_figure.py \
    --input-h5  /data/al1050_pct5.h5 \
    --scan-ids  1.1 2.1 3.1 4.1 5.1 \
    --voxel-spacing-nm 400 50 50 \
    --angle-unit deg \
    --labels-h5 runs/al1050_pct5/method/labels_3d.h5 \
    --slice-z 12 \
    --orientation-channel chi \
    --scalebar-um 10 \
    --n-cells-3d 3 \
    --out-dir runs/al1050_pct5/figure
```

## How to sweep the KAM threshold (for the §4.2 figure)

The current `compare_3d_methods.py` runs at one operating point. To
show the failure mode of KAM-thresholding as a function of the
threshold, drive `segment_3d_kam_baseline.py` from a shell loop:

```bash
for p in 50 60 70 80 90 95; do
    python segment_3d_kam_baseline.py \
        --input-h5 /data/al1050_pct5.h5 \
        --scan-ids 1.1 2.1 3.1 4.1 5.1 \
        --voxel-spacing-nm 400 50 50 \
        --angle-unit deg \
        --threshold-percentile ${p} \
        --out-dir runs/al1050_pct5/kam_sweep/p${p}
done
```

`largest_component_fraction` and `n_components` from each
`parameters_used.json` (or the `kam_connected_component_stats.summary.csv`)
can then be plotted manually. This is the quantitative version of the
"KAM fails in 3D" claim that the manuscript currently states only
qualitatively.

## Required data

These scripts do not ship with example data and do not assume any
HDF5 dataset names. The user must specify:

* `--input-h5`: an ID03-style HDF5 file readable by
  `darling.DataSet`,
* `--scan-ids`: one scan id per Z layer,
* `--voxel-spacing-nm`: explicit `(dz, dy, dx)` in nm,
* `--angle-unit`: `deg`, `rad` or `mrad`.

If `darling` cannot read your file, pass `--no-darling --field-dataset
PATH/TO/CHI_PHI` and the script will read a preprocessed
`(Z, Y, X, 2)` orientation array directly. This path is intentionally
unattractive: the package recommendation is to use `darling` for all
loading.

## Determinism

Only `segment_3d_cells.py` involves randomness. To make it
reproducible, the script:

1. accepts `--random-seed` (mandatory),
2. samples the seed pool in numpy with that seed,
3. passes the (deterministic) seed list to the C++ via the
   `seed_points` argument so the C++ random draw is bypassed.

The C++ kernel itself is non-seedable today (`AUDIT.md`, B8); rather
than patching the C++, the scripts compute a deterministic seed list
in Python and feed it through the `seed_points` argument that
`flood_fill_random_seeds_3d` already accepts. Same result, no source
changes.

## Known caveats

* The C++ flood fill mutates the mask in place; the wrapper here
  copies it before passing it down.
* The C++ flood fill does not bounds-check neighbour indices; the
  wrapper here pads the input by `max(footprint half-extent)` and
  crops back.
* KAM is not aware of voxel spacing; if `--kam-radius-nm` is provided
  the kernel is built from a physical radius, otherwise the explicit
  `--kam-kernel-zyx` voxel-count kernel is used.
* The watershed is done in skimage's voxel grid (also unaware of
  spacing); for anisotropic voxels this preferentially cuts along the
  densely-sampled axis. This is documented in the manuscript critique.
* The "graph-cut" comparison method named in the paper is *not* in
  the public `disell` source, so it cannot be compared here. The
  comparison falls back to flood-fill + watershed vs. KAM-threshold
  only.
