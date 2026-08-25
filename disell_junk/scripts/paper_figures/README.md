# Paper pipeline — 6.2% 3D single-crystal dataset

Reproducible scripts generating the figures and quantitative analyses for
*Segmentation of Dislocation Cell Structures in 3D Dark-Field X-ray
Microscopy Data* from the prepared 11-layer 6.2% dataset
(`layer_<n>_1/{mean.npy, motors.npy, processing_info.json, segmap.npy}`).

## One command

```bash
python run_all.py --config config_6_2pct.json            # everything
python run_all.py --config config_6_2pct.json --skip-sensitivity
```

Outputs land in `<package root>/paper_outputs/6-2pct/<stage>/` (git-ignored).
Every stage writes a `parameters_used.json` sidecar recording the config,
input identities, parameters, random seed, software versions and git state.

## Stages (dependency order)

| script | outputs | purpose |
|---|---|---|
| `run_preprocess.py` | `volume/volume_registered.h5` (+unfiltered, diagnostics) | load + validate the 11 layers, crop to the analysis window, sub-pixel inter-layer registration, NaN-padded shifts, in-plane median filter, validity mask |
| `run_segment_3d.py` | `segmentation_3d/{markers,labels,kam}_3d.h5`, `.vti`, overlays | full manuscript method: two-stage deterministic flood fill (local + running-mean global threshold, footprint tolerance, min size with parking), physical-footprint KAM, marker watershed, connected-component validation/splitting |
| `run_slicewise.py` | `slicewise/labels_slicewise.h5`, coherence CSVs, figures | per-layer 2D segmentation with in-plane parameters; label-invariant slice-to-slice coherence (variation of information, matched overlap) for 2D vs direct 3D |
| `run_kam_baseline.py` | `kam_baseline/threshold_sweep.csv`, labels, response figure | conventional KAM thresholding on the *same* KAM field; percolation/fragmentation sweep; declared operating-point rule (max component count) |
| `run_comparison.py` | `comparison/method_summaries.csv`, per-cell CSVs, figures | cell counts, empirical volume distributions, equivalent diameters, edge truncation, intra-cell spread σ_k, boundary-band KAM E_bd, method agreement |
| `fig_3d_cells.py` | `fig_3d_cells/fig_3d_cells_6-2pct.{pdf,png}` | publication figure: slice + zoom + 3D rendering of three neighbouring cells selected by explicit criteria |
| `run_sensitivity.py` | `sensitivity/sensitivity_sweep.csv`, `seed_agreement.csv`, figure | one-at-a-time parameter sweep + 5 random seeds on a fixed ROI |
| `run_refinement_comparison.py` | `refinement_comparison/refinement_comparison.csv` | watershed vs mean-feature region growing from identical markers (ROI); graph cut not implemented |

## Dataset facts encoded in `config_6_2pct.json`

* channels: `mean.npy[..., 0]` = χ, `[..., 1]` = φ (diffry), **degrees**
  (order fixed by darling ID03 defaults);
* analysis window: rows 400:600, cols 100:600 of the 925×925 maps
  (recovered from the historic `segmap.npy` by cross-correlation, identical
  in all 11 layers) → stacked volume (11, 200, 500, 2);
* voxel spacing (dz, dy, dx) = (500, 1240, 400) nm — user-provided;
  note the manuscript's Table 1 "(50, 150) nm" does not match;
* `segmap.npy` is a historic per-layer 2D segmentation, **not** a mask;
* a preparation-time inter-layer shift was already applied; the residual
  (≈ −0.55 px/layer along rows) is removed here by sub-pixel phase
  correlation (`normalization=None`; the skimage default "phase"
  normalization silently fails on smooth angular fields).

## Conventions

* arrays are (Z, Y, X[, C]); HDF5/VTI carry axis order and spacing
  attributes; VTI spacing is (dx, dy, dz) per VTK convention;
* thresholds and KAM are per-channel RMS feature distances in degrees
  (`disell.kam(..., per_channel_rms=True)`), matching the manuscript ρ;
* flood-fill footprint `inplane8_plus_z` = full 3×3 in-plane + z face
  neighbours (anisotropy-aware); KAM uses a physically isotropic footprint
  of the configured radius;
* watershed connectivity 1 (face); undefined KAM inside the mask is set to
  the in-mask maximum (claimed last) before watershed.

Tests for the scientific conventions live in `<package root>/tests/`
(`pytest tests/`).
