#!/usr/bin/env python3
"""A small, readable front end to the phantom segmentation benchmark.

This module exists so that a notebook can try a parameter set and look at the
result in three lines.  It wraps the same code the study uses --
``pipelines.run_flood_fill_two_stage`` on the frozen 360-cell phantom -- and
adds nothing to it except caching and plotting.

One segmentation takes about three seconds, which is what makes interactive
tuning worth having.

Scripted use::

    import phantom_lab as lab
    ph = lab.load_phantom()
    res = lab.segment(ph, **lab.BEST_PARAMS)
    print(lab.score(ph, res))
    lab.show_slice(ph, res, z=12)

Interactive use, in a notebook with ipywidgets::

    lab.tune(ph)                      # sliders for both arms, live scores

Searching::

    lab.sweep(ph, "min_cell_size", [20, 60, 116, 180])
    lab.grid(ph, "min_cell_size", [60, 116], "footprint_tolerance", [0.04, 0.2])

The two arms are ``"flood fill"`` (the paper's algorithm) and
``"KAM threshold"`` (the comparator); :func:`run` dispatches between them, and
:data:`ARM_PARAMETERS` says which parameters each one takes.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: The study's primary phantom: 6.2 %-strain cell size *and* 6.2 %-strain
#: misorientation together.
PHANTOM_NPZ = HERE / "phantoms" / "primary_6p2_consistent.npz"


#: Fixed run settings of the study.  They are not parameters under search:
#: the iteration caps only have to be large enough not to bind, and the
#: watershed connectivity is face-only in both arms.
MAX_SEED_ATTEMPTS = 700_000
STAGNATION_TOLERANCE = 2_000
WATERSHED_CONNECTIVITY = 1

#: Radius cap for this study.  Both the flood-fill neighbourhood and the KAM
#: kernel are restricted to 1.5 um, so neither smooths across more than about
#: a third of a typical cell diameter.
MAX_RADIUS_UM = 2.0

#: Nothing below this is used.  At 1.0 um z spacing a kernel of radius r spans
#: floor(r) neighbours in z, so below 0.9 um the footprint has no out-of-plane
#: reach worth the name and the operator is effectively two-dimensional.  The
#: search found a hard ARI cliff there: below it every configuration collapsed
#: to about 0.55, above it they sit at 0.86-0.89.
MIN_RADIUS_UM = 0.9

#: The count-first winner from ``analysis/count_first_v1``: smallest absolute
#: cell-count error, then identity F1.  On seed 0 it recovers exactly 360
#: cells (ARI 0.885).
#:
#: **It is outside the 1.5 um cap** -- its footprint radius is 1.95 um -- so it
#: is kept as the historical reference point, not as a starting point.  Use
#: :data:`CAPPED_START` for anything under the cap.
BEST_PARAMS = {
    "local_threshold_deg": 0.006804162534344432,
    "global_threshold_deg": 0.8729926391777059,
    "footprint_tolerance": 0.04,
    "footprint_radius_um": 1.948897380571898,
    "min_cell_size": 116,
    "kam_radius_um": 1.1737337858305015,
}


# --------------------------------------------------------------- the phantom


#: Provisional best-known parameters **under the 1.5 um cap**, pending the
#: systematic search in ``capped_search.py``.  The smaller neighbourhood forces
#: a much larger local threshold (0.02 against 0.0068) to keep the cell count
#: under control.
CAPPED_START = {
    "local_threshold_deg": 0.020,
    "global_threshold_deg": 0.873,
    "footprint_tolerance": 0.04,
    "footprint_radius_um": 1.2,
    "min_cell_size": 20,
    "kam_radius_um": 1.2,
    "merge_size_voxels": 0,
    "merge_threshold_deg": 0.0,
}


def footprint_classes(
    spacing_um_zyx=(1.0, 0.4, 0.4),
    max_radius_um: float = MAX_RADIUS_UM,
    *,
    min_radius_um: float = MIN_RADIUS_UM,
    minimum_voxels: int = 5,
    step_um: float = 0.005,
):
    """Every distinct footprint a radius up to the cap can rasterise to.

    Radius is not really a continuous parameter: at 0.4 um in-plane spacing,
    radii below 1.5 um produce only a handful of distinct neighbourhoods.  This
    returns one representative radius per distinct footprint, which is what a
    search should enumerate instead of sampling a continuous range.  Footprints
    below ``minimum_voxels`` (the degenerate centre-only case) are dropped.

    Returns a list of ``(radius_um, voxel_count)``, smallest first.
    """

    import pipelines

    seen: dict[bytes, tuple[float, int]] = {}
    for radius in np.arange(float(min_radius_um), float(max_radius_um) + 1e-9,
                            float(step_um)):
        footprint = pipelines.isotropic_footprint(spacing_um_zyx, float(radius))
        key = footprint.shape + (footprint.tobytes(),)
        count = int(footprint.sum())
        if key not in seen and count >= int(minimum_voxels):
            seen[key] = (round(float(radius), 4), count)
    return sorted(seen.values(), key=lambda item: item[1])


@dataclass
class Phantom:
    """The frozen ground-truth volume, plus a KAM cache keyed by radius."""

    labels: np.ndarray            # (Z, Y, X) int32, 1-based, 360 cells
    field: np.ndarray             # (Z, Y, X, 2) float32, degrees
    mask: np.ndarray              # (Z, Y, X) bool
    spacing_um_zyx: tuple[float, float, float]
    _kam_cache: dict = _field(default_factory=dict, repr=False)

    @property
    def n_cells(self) -> int:
        """Cells actually present.  Label ids can have gaps: a Laguerre site
        that loses everywhere leaves an empty label, so ``labels.max()`` can
        exceed the true count.  Every metric counts unique labels, so this
        does too."""

        return int(np.unique(self.labels[self.labels > 0]).size)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.labels.shape)

    #: How many KAM fields to keep.  Each is ~2.4 MB; tuning sweeps radii, so
    #: the cache is bounded rather than left to grow.
    KAM_CACHE_SIZE = 12

    def kam(self, radius_um: float) -> np.ndarray:
        """Manuscript KAM at one footprint radius, computed once per radius.

        Radii are keyed by the footprint they actually rasterise to, so two
        nearby radii that produce the same neighbourhood share one field --
        the same equivalence the search's ``RadiusClasses`` relies on.
        """

        import pipelines

        footprint = pipelines.isotropic_footprint(self.spacing_um_zyx, float(radius_um))
        key = (footprint.shape, footprint.tobytes())
        if key not in self._kam_cache:
            while len(self._kam_cache) >= self.KAM_CACHE_SIZE:
                self._kam_cache.pop(next(iter(self._kam_cache)))
            self._kam_cache[key] = pipelines.masked_kam(
                self.field, self.mask, footprint
            )
        return self._kam_cache[key]

    def footprint_voxels(self, radius_um: float) -> int:
        """How many voxels a radius rasterises to -- useful while tuning."""

        import pipelines

        return int(
            pipelines.isotropic_footprint(self.spacing_um_zyx, float(radius_um)).sum()
        )

    def __repr__(self) -> str:
        z, y, x = self.shape
        sz, sy, sx = self.spacing_um_zyx
        return (
            f"Phantom({z}x{y}x{x} voxels, "
            f"{z * sz:.0f}x{y * sy:.0f}x{x * sx:.0f} um, "
            f"{self.n_cells} cells)"
        )


#: The strain series built by ``strain_phantoms.py`` from the measured DFXM
#: trend in Zelenika et al., Sci Rep 15, 8655 (2025).  Cell size falls and
#: misorientation rises with strain; 6.2 % is extrapolated past the paper's
#: measured 0.6-4.6 % range.
STRAIN_PHANTOM_DIR = HERE / "phantoms"


def load_strain_phantom(strain_key: str, realization: int = 0) -> Phantom:
    """Load one phantom of the strain series, e.g. ``lab.load_strain_phantom("4p6")``.

    Keys are ``"2p4"``, ``"3p5"``, ``"4p6"`` and ``"6p2"``.  Each carries the
    paper's cell size and misorientation for that strain, with wall geometry,
    intracell structure and noise held fixed, so any change in the best
    segmentation parameters is attributable to the microstructure alone.
    """

    path = STRAIN_PHANTOM_DIR / f"strain_{strain_key}_r{realization}.npz"
    if not path.exists():
        available = sorted(q.stem for q in STRAIN_PHANTOM_DIR.glob("*.npz")) \
            if STRAIN_PHANTOM_DIR.exists() else []
        raise FileNotFoundError(
            f"{path} is missing. Build the series with "
            f"`python strain_phantoms.py build`. Present: {available or 'none'}"
        )
    return load_phantom(path)


def strain_series(realization: int = 0) -> dict[str, Phantom]:
    """Every strain phantom that has been built, keyed by strain key."""

    import strain_phantoms

    out = {}
    for key in strain_phantoms.STRAIN_TREND:
        try:
            out[key] = load_strain_phantom(key, realization)
        except FileNotFoundError:
            continue
    return out


def load_phantom(path: Path | str = PHANTOM_NPZ) -> Phantom:
    """Load the frozen phantom the whole study is scored against.

    ``field`` is the *latent* label-derived angular field, which is what the
    benchmark segments -- the same choice ``oracle_core.load_workspace`` makes.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing.  Build it with "
            "`python strain_phantoms.py primary` from the benchmark directory."
        )
    with np.load(path) as data:
        labels = np.ascontiguousarray(data["labels"])
        field = np.ascontiguousarray(data["latent"])
        spacing = tuple(float(v) for v in data["spacing"])
    return Phantom(
        labels=labels,
        field=field,
        mask=labels > 0,
        spacing_um_zyx=spacing,
    )


# ------------------------------------------------------------- segmentations


@dataclass
class Result:
    """One segmentation: the markers, the refined partition, its settings."""

    labels: np.ndarray      # (Z, Y, X) int32, final partition
    markers: np.ndarray     # (Z, Y, X) int32, pre-watershed markers (0 = wall)
    kam: np.ndarray         # (Z, Y, X) float32, the watershed elevation source
    method: str
    params: dict
    seed: int
    merge_diagnostics: dict | None = None

    @property
    def n_cells(self) -> int:
        return int(np.unique(self.labels[self.labels > 0]).size)

    def __repr__(self) -> str:
        return (
            f"Result({self.method}, seed={self.seed}, "
            f"{self.n_cells} cells, {int(self.markers.max())} markers)"
        )


def segment(
    phantom: Phantom,
    *,
    local_threshold_deg: float,
    global_threshold_deg: float,
    footprint_tolerance: float,
    footprint_radius_um: float,
    min_cell_size: int,
    kam_radius_um: float,
    seed: int = 0,
    merge_size_voxels: int = 0,
    merge_threshold_deg: float = 0.0,
    merge_mode: str = "absolute",
    merge_factor: float = 0.25,
) -> Result:
    """Run the paper's algorithm: size-ordered flood fill + one KAM watershed.

    The first six arguments are the searched parameters.  ``seed`` only sets
    the order candidate seeds are visited in; changing it is the cheapest way
    to see how stable a parameter set is.

    Setting ``merge_size_voxels`` above zero adds the orientation-gated merge
    step from :mod:`merge_cells`.  ``merge_mode="relative"`` calibrates the
    threshold against the misorientation between large regions in this volume,
    so no angle has to be supplied; ``"absolute"`` uses
    ``merge_threshold_deg`` directly.  The step is: segment with a small ``min_cell_size``, then
    fold each fragment below ``merge_size_voxels`` into the neighbour whose
    mean orientation is closest, provided they differ by less than
    ``merge_threshold_deg``.  This is the alternative to controlling the cell
    count with a large ``min_cell_size``, which buys count control by deleting
    genuinely small cells.
    """

    import pipelines

    footprint = pipelines.isotropic_footprint(
        phantom.spacing_um_zyx, footprint_radius_um
    )
    kam = phantom.kam(kam_radius_um)
    labels, markers, _, _, _ = pipelines.run_flood_fill_two_stage(
        phantom.field,
        phantom.mask,
        kam,
        footprint,
        local_threshold_deg=float(local_threshold_deg),
        global_threshold_deg=float(global_threshold_deg),
        footprint_tolerance=float(footprint_tolerance),
        min_cell_size=int(min_cell_size),
        max_seed_attempts=MAX_SEED_ATTEMPTS,
        stagnation_tolerance=STAGNATION_TOLERANCE,
        random_seed=int(seed),
        watershed_connectivity=WATERSHED_CONNECTIVITY,
        recycle_small_grains=False,
    )
    if int(markers.max()) == 0:
        raise RuntimeError(
            "No markers were accepted -- this parameter set is invalid.  "
            "Usually min_cell_size is too large or local_threshold_deg too small."
        )
    labels = np.asarray(labels, dtype=np.int32)
    params = {
        "local_threshold_deg": float(local_threshold_deg),
        "global_threshold_deg": float(global_threshold_deg),
        "footprint_tolerance": float(footprint_tolerance),
        "footprint_radius_um": float(footprint_radius_um),
        "min_cell_size": int(min_cell_size),
        "kam_radius_um": float(kam_radius_um),
    }
    method = "flood fill (size ordered)"
    diagnostics = None
    if int(merge_size_voxels) > 0:
        import merge_cells

        labels, diagnostics = merge_cells.merge_small_cells(
            labels, phantom.field,
            merge_size_voxels=int(merge_size_voxels),
            merge_threshold_deg=float(merge_threshold_deg),
            merge_mode=merge_mode,
            merge_factor=float(merge_factor),
            local_threshold_deg=float(local_threshold_deg),
            return_diagnostics=True,
        )
        params["merge_size_voxels"] = int(merge_size_voxels)
        params["merge_threshold_deg"] = diagnostics["merge_threshold_deg"]
        params["merge_mode"] = merge_mode
        if merge_mode != "absolute":
            params["merge_factor"] = float(merge_factor)
        method = "flood fill + merge"
    return Result(
        labels=labels,
        markers=np.asarray(markers, dtype=np.int32),
        kam=kam,
        method=method,
        params=params,
        seed=int(seed),
        merge_diagnostics=diagnostics,
    )


def segment_kam_baseline(
    phantom: Phantom,
    *,
    percentile: float,
    kam_radius_um: float,
    min_cell_size: int,
    connectivity: int = 1,
) -> Result:
    """The KAM-threshold comparator: low-KAM components + the same watershed.

    ``percentile`` is the KAM percentile below which a voxel is cell interior.
    """

    import pipelines

    kam = phantom.kam(kam_radius_um)
    labels, markers, threshold = pipelines.run_kam_threshold(
        kam,
        phantom.mask,
        percentile=float(percentile),
        min_cell_size=int(min_cell_size),
        connectivity=int(connectivity),
        watershed_connectivity=WATERSHED_CONNECTIVITY,
    )
    return Result(
        labels=np.asarray(labels, dtype=np.int32),
        markers=np.asarray(markers, dtype=np.int32),
        kam=kam,
        method="KAM threshold",
        params={
            "percentile": float(percentile),
            "kam_threshold_deg": float(threshold),
            "kam_radius_um": float(kam_radius_um),
            "min_cell_size": int(min_cell_size),
            "connectivity": int(connectivity),
        },
        seed=0,
    )


# ------------------------------------------------------------------ scoring


def score(phantom: Phantom, result: Result, *, identity: bool = True) -> dict:
    """Score a partition against the ground truth.

    Returns the partition metrics (ARI, VI, boundary ASSD, cell counts) and,
    unless ``identity=False``, the object-level identity scores.  A predicted
    cell counts as the same object as a true cell only if they overlap by at
    least 60 % in both directions; ``identity_f1`` is then
    ``2 * recovered / (n_true + n_pred)``, the same definition
    ``two_stage_oracle.selection_metrics`` uses to rank the study.
    """

    import bench_metrics
    import strict_recovery as sr

    out = bench_metrics.evaluate(
        phantom.labels, result.labels, phantom.spacing_um_zyx
    )
    out["count_error_percent"] = 100.0 * out["cell_count_error"] / out["n_cells_truth"]
    # The study's headline objective: cells that came out almost exactly right,
    # and how far the rest are from ground truth in voxels.  Splitting a cell
    # costs nothing here; fusing two costs in proportion to how wrong it is.
    strict = sr.strict_recovery(phantom.labels, result.labels, tau=0.9)
    out.update({
        "recovered_at_90": strict["recovered_at_tau"],
        "recovery_rate_at_90": strict["recovery_rate_at_tau"],
        "contamination": strict["contamination"],
        "fused_true_cells": strict["fused_true_cells"],
        "strict_split_true_cells": strict["split_true_cells"],
    })
    if identity:
        import object_orientation_metrics as oom

        _, objects = oom.match_cells(
            phantom.labels,
            result.labels,
            phantom.field,
            phantom.spacing_um_zyx,
            purity_threshold=0.6,
            completeness_threshold=0.6,
        )
        n_true = int(objects["n_true_cells"])
        n_pred = int(objects["n_predicted_cells"])
        recovered = int(objects["one_to_one_recovered_cells"])
        orientation_correct = int(objects["orientation_correct_cells_at_0p02deg"])
        out.update(
            {
                "identity_f1": 2 * recovered / max(n_true + n_pred, 1),
                # Recall is the "are the cells I do have excellent" number:
                # the fraction of TRUE cells recovered one-to-one.  It is the
                # objective when a merge step runs downstream, because spare
                # fragments do not enter it but fusion and shattering both do.
                "identity_recall": recovered / max(n_true, 1),
                "identity_precision": recovered / max(n_pred, 1),
                "recovered_cells": recovered,
                "orientation_f1_0p02": (
                    2 * orientation_correct / max(n_true + n_pred, 1)
                ),
                "split_true_cells": int(objects["split_true_cells"]),
                "merged_predicted_cells": int(objects["merged_predicted_cells"]),
                "median_orientation_error_deg": objects[
                    "median_matched_mean_orientation_error_deg"
                ],
            }
        )
    return out


def score_table(phantom: Phantom, results: dict, **kwargs):
    """Score several named results side by side.  Returns a pandas DataFrame."""

    import pandas as pd

    rows = []
    for name, result in results.items():
        row = {"name": name, "method": result.method, "seed": result.seed}
        row.update(score(phantom, result, **kwargs))
        rows.append(row)
    return pd.DataFrame(rows).set_index("name")


# ----------------------------------------------------------------- plotting


def _label_colours(labels: np.ndarray, seed: int = 0):
    """A random but reproducible colour per label; label 0 is black."""

    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(seed)
    n = int(labels.max()) + 1
    colours = rng.uniform(0.25, 1.0, size=(n, 3))
    colours[0] = 0.0
    return ListedColormap(colours)


def match_to_truth(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Recolour a prediction so each cell takes the id it most overlaps.

    This is a *display* aid only -- it makes the two label panels comparable by
    eye.  It is not a matching metric: several predicted cells can be recoloured
    to the same truth id, which is exactly what an over-segmentation looks like.
    """

    n_pred = int(prediction.max()) + 1
    n_truth = int(truth.max()) + 1
    flat = truth.ravel().astype(np.int64) * n_pred + prediction.ravel().astype(np.int64)
    counts = np.bincount(flat, minlength=n_truth * n_pred).reshape(n_truth, n_pred)
    counts[0, :] = 0
    mapping = counts.argmax(axis=0).astype(np.int32)
    mapping[0] = 0
    return mapping[prediction]


def show_slice(
    phantom: Phantom,
    result: Result | None = None,
    z: int | None = None,
    *,
    channel: int = 0,
    match_colours: bool = True,
    figsize: tuple[float, float] = (16, 9),
    seed: int = 0,
):
    """Six panels through one z slice: field, KAM, truth, prediction, overlays.

    ``z`` defaults to the middle slice.  With ``result=None`` only the phantom
    panels are drawn.  Set ``match_colours=False`` to give the prediction its
    own arbitrary colours instead of its best-overlap truth colours.
    """

    import matplotlib.pyplot as plt

    if z is None:
        z = phantom.shape[0] // 2
    truth = phantom.labels[z]
    extent = _extent(phantom)

    field_panel = (
        f"1. angular field, channel {channel} (deg)",
        lambda ax: _show_field(ax, phantom.field[z, ..., channel], extent),
    )
    truth_panel = (
        "ground truth cells",
        lambda ax: ax.imshow(
            truth, cmap=_label_colours(phantom.labels, seed),
            vmin=0, vmax=phantom.labels.max(), interpolation="nearest", extent=extent,
        ),
    )
    if result is None:
        panels = [field_panel, truth_panel]
    else:
        pred = result.labels[z]
        shown = match_to_truth(phantom.labels, result.labels)[z] if match_colours else pred
        cmap = _label_colours(phantom.labels if match_colours else result.labels, seed)
        vmax = (phantom.labels if match_colours else result.labels).max()
        radius = result.params.get("kam_radius_um", float("nan"))
        panels = [
            # Top row follows the algorithm: field -> KAM -> markers.
            field_panel,
            (f"2. KAM, r = {radius:.2f} um (deg)",
             lambda ax: _show_kam(ax, result.kam[z], extent)),
            ("3. markers before watershed (grey = wall)",
             lambda ax: _show_markers(ax, result.markers[z], extent, seed)),
            # Bottom row is the comparison: truth beside prediction, then both.
            truth_panel,
            (f"predicted cells ({result.n_cells} vs {phantom.n_cells} true)",
             lambda ax: ax.imshow(shown, cmap=cmap, vmin=0, vmax=vmax,
                                  interpolation="nearest", extent=extent)),
            ("boundaries: truth white, predicted red, agreed yellow",
             lambda ax: _show_boundaries(ax, truth, pred, extent)),
        ]

    ncols = 3 if len(panels) > 2 else 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax, (title, draw) in zip(axes.ravel(), panels):
        image = draw(ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        if image is not None and title.endswith("(deg)"):
            fig.colorbar(image, ax=ax, fraction=0.046, shrink=0.85)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"z = {z} of {phantom.shape[0] - 1}", fontsize=12)
    fig.tight_layout()
    return fig


def show_slices(
    phantom: Phantom,
    result: Result,
    z_values=None,
    *,
    match_colours: bool = True,
    figsize_per_row: float = 3.2,
    seed: int = 0,
):
    """One row per z: ground truth, prediction, and the boundary overlay."""

    import matplotlib.pyplot as plt

    if z_values is None:
        depth = phantom.shape[0]
        z_values = [depth // 6, depth // 2, 5 * depth // 6]
    z_values = list(z_values)
    extent = _extent(phantom)
    matched = match_to_truth(phantom.labels, result.labels) if match_colours else result.labels
    cmap = _label_colours(phantom.labels if match_colours else result.labels, seed)
    vmax = (phantom.labels if match_colours else result.labels).max()

    fig, axes = plt.subplots(
        len(z_values), 3,
        figsize=(3 * figsize_per_row * 1.15, len(z_values) * figsize_per_row),
        squeeze=False,
    )
    for row, z in enumerate(z_values):
        truth = phantom.labels[z]
        axes[row, 0].imshow(truth, cmap=_label_colours(phantom.labels, seed),
                            vmin=0, vmax=phantom.labels.max(),
                            interpolation="nearest", extent=extent)
        axes[row, 1].imshow(matched[z], cmap=cmap, vmin=0, vmax=vmax,
                            interpolation="nearest", extent=extent)
        _show_boundaries(axes[row, 2], truth, result.labels[z], extent)
        axes[row, 0].set_ylabel(f"z = {z}\ny (um)")
        if row == 0:
            for ax, title in zip(
                axes[row],
                ["ground truth", "predicted", "truth white / predicted red"],
            ):
                ax.set_title(title, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("x (um)")
    fig.tight_layout()
    return fig


def show_size_distribution(phantom: Phantom, results: dict, *, bins: int = 40):
    """Cell-volume histograms: ground truth against one or more predictions."""

    import matplotlib.pyplot as plt

    voxel = float(np.prod(phantom.spacing_um_zyx))
    truth_volumes = np.bincount(phantom.labels.ravel())[1:] * voxel
    edges = np.histogram_bin_edges(np.log10(truth_volumes), bins=bins)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(np.log10(truth_volumes), bins=edges, histtype="stepfilled",
            alpha=0.35, color="0.35", label=f"ground truth (n = {truth_volumes.size})")
    for name, result in results.items():
        volumes = np.bincount(result.labels.ravel())[1:] * voxel
        volumes = volumes[volumes > 0]
        ax.hist(np.log10(volumes), bins=edges, histtype="step", linewidth=1.8,
                label=f"{name} (n = {volumes.size})")
    ax.set_xlabel("log10 cell volume (um^3)")
    ax.set_ylabel("cells")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def cell_diagnosis(
    phantom: Phantom,
    result: Result,
    *,
    substantial_fraction: float = 0.10,
) -> dict:
    """Classify every ground-truth cell as recovered, split, merged, or missed.

    ``recovered`` uses the study's criterion (60 % overlap in both directions,
    one-to-one).  A cell that fails it is called ``split`` when two or more
    predictions each take a substantial share of it, ``merged`` when the
    prediction covering it also covers another true cell substantially, and
    ``missed`` otherwise.  Returns ``{"status": array indexed by truth id,
    "counts": {...}}``.
    """

    import object_orientation_metrics as oom

    pairs, _ = oom.match_cells(
        phantom.labels, result.labels, phantom.field, phantom.spacing_um_zyx,
        purity_threshold=0.6, completeness_threshold=0.6,
    )
    rows, cols, overlap, truth_sizes, pred_sizes = oom.contingency(
        phantom.labels, result.labels
    )
    n_truth = truth_sizes.size

    substantial_for_truth = overlap >= substantial_fraction * truth_sizes[rows]
    substantial_for_pred = overlap >= substantial_fraction * pred_sizes[cols]
    fragments = np.bincount(rows[substantial_for_truth], minlength=n_truth)
    sources = np.bincount(cols[substantial_for_pred], minlength=pred_sizes.size)

    dominant = np.zeros(n_truth, dtype=np.int64)
    best = np.zeros(n_truth, dtype=np.int64)
    for t, p, n in zip(rows, cols, overlap):
        if n > best[t]:
            best[t], dominant[t] = n, p

    status = np.array(["missed"] * n_truth, dtype=object)
    status[0] = "background"
    status[fragments >= 2] = "split"
    merged = (fragments < 2) & (sources[dominant] >= 2)
    status[merged] = "merged"
    for pair in pairs:
        status[pair["true_cell"]] = "recovered"

    present = np.flatnonzero(truth_sizes)
    present = present[present > 0]
    counts = {
        name: int(np.sum(status[present] == name))
        for name in ("recovered", "split", "merged", "missed")
    }
    return {"status": status, "counts": counts}


#: Colours used by :func:`show_errors`.
STATUS_COLOURS = {
    "background": (0.0, 0.0, 0.0),
    "recovered": (0.20, 0.65, 0.30),   # green
    "split": (0.95, 0.65, 0.15),       # orange
    "merged": (0.30, 0.45, 0.90),      # blue
    "missed": (0.85, 0.15, 0.15),      # red
}


def show_errors(
    phantom: Phantom,
    result: Result,
    z_values=None,
    *,
    diagnosis: dict | None = None,
    figsize_per_panel: float = 4.0,
):
    """Colour each ground-truth cell by whether the segmentation recovered it.

    Pass a previously computed ``diagnosis`` to avoid recomputing the match.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if diagnosis is None:
        diagnosis = cell_diagnosis(phantom, result)
    if z_values is None:
        depth = phantom.shape[0]
        z_values = [depth // 6, depth // 2, 5 * depth // 6]
    z_values = list(z_values)

    order = ["background", "recovered", "split", "merged", "missed"]
    index = {name: i for i, name in enumerate(order)}
    coded = np.array([index[str(s)] for s in diagnosis["status"]], dtype=np.uint8)
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([STATUS_COLOURS[name] for name in order])
    extent = _extent(phantom)

    fig, axes = plt.subplots(
        1, len(z_values),
        figsize=(len(z_values) * figsize_per_panel, figsize_per_panel * 1.15),
        squeeze=False,
    )
    for ax, z in zip(axes[0], z_values):
        ax.imshow(coded[phantom.labels[z]], cmap=cmap, vmin=0, vmax=len(order) - 1,
                  interpolation="nearest", extent=extent)
        ax.set_title(f"z = {z}", fontsize=10)
        ax.set_xlabel("x (um)")
    axes[0, 0].set_ylabel("y (um)")
    counts = diagnosis["counts"]
    fig.legend(
        handles=[
            Patch(facecolor=STATUS_COLOURS[name], label=f"{name} ({counts[name]})")
            for name in order[1:]
        ],
        loc="lower center", ncol=4, frameon=False, fontsize=9,
    )
    fig.suptitle("ground-truth cells coloured by outcome", fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


# --------------------------------------------------------- plotting internals


def _extent(phantom: Phantom):
    _, sy, sx = phantom.spacing_um_zyx
    _, ny, nx = phantom.shape
    return (0.0, nx * sx, ny * sy, 0.0)


def _show_field(ax, plane, extent):
    return ax.imshow(plane, cmap="twilight", interpolation="nearest", extent=extent)


def _show_kam(ax, plane, extent):
    finite = plane[np.isfinite(plane)]
    top = float(np.percentile(finite, 99)) if finite.size else 1.0
    return ax.imshow(plane, cmap="magma", vmin=0.0, vmax=top,
                     interpolation="nearest", extent=extent)


def _show_markers(ax, plane, extent, seed):
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(seed)
    n = int(plane.max()) + 1
    colours = rng.uniform(0.25, 1.0, size=(max(n, 1), 3))
    colours[0] = 0.45  # unassigned wall voxels
    return ax.imshow(plane, cmap=ListedColormap(colours), vmin=0, vmax=max(n - 1, 1),
                     interpolation="nearest", extent=extent)


def _outline(plane: np.ndarray) -> np.ndarray:
    """Voxels whose right or lower neighbour carries a different label."""

    edge = np.zeros(plane.shape, dtype=bool)
    edge[:-1, :] |= plane[:-1, :] != plane[1:, :]
    edge[:, :-1] |= plane[:, :-1] != plane[:, 1:]
    return edge


def _show_boundaries(ax, truth_plane, pred_plane, extent):
    rgb = np.zeros(truth_plane.shape + (3,), dtype=float)
    truth_edge = _outline(truth_plane)
    pred_edge = _outline(pred_plane)
    rgb[truth_edge] = 1.0                       # white
    rgb[pred_edge] = (1.0, 0.15, 0.15)          # red
    rgb[truth_edge & pred_edge] = (1.0, 0.9, 0.2)  # yellow where they agree
    return ax.imshow(rgb, interpolation="nearest", extent=extent)


# ------------------------------------------------------------------ sweeping

#: Bounds of the study's declared search box, plus what each parameter does.
#: ``tune`` builds its sliders from this, so widening a range here widens the
#: playground.  ``log`` marks the parameters that are searched logarithmically.
PARAMETER_GUIDE = {
    "local_threshold_deg": {
        "range": (0.002, 0.16), "log": True, "step": 0.0002,
        "help": "max angle between neighbouring voxels of one cell. "
                "Smaller = more, finer cells.",
    },
    "global_threshold_deg": {
        "range": (0.005, 2.0), "log": True, "step": 0.005,
        "help": "max angle from the growing cell's running mean; caps total "
                "drift inside one cell. Untick 'use global' to disable.",
    },
    "footprint_tolerance": {
        "range": (0.04, 0.96), "log": False, "step": 0.01,
        "help": "fraction of the neighbourhood that must agree before a voxel "
                "is accepted. Higher = stricter, cleaner walls, fewer cells.",
    },
    "footprint_radius_um": {
        "range": (0.9, 2.0), "log": False, "step": 0.005,
        "help": "physical radius of that neighbourhood, in um. Restricted to "
                "0.9-2.0: below 0.9 the kernel has no z reach, and only 22 "
                "distinct footprints exist in the range, so this is effectively "
                "a discrete choice (see footprint_classes()).",
    },
    "min_cell_size": {
        "range": (3, 200), "log": False, "step": 1,
        "help": "smallest accepted marker, in voxels. Raise it to kill fragments.",
    },
    "kam_radius_um": {
        "range": (0.9, 2.0), "log": False, "step": 0.005,
        "help": "radius of the KAM field used as watershed elevation. Moves "
                "where boundaries land; does not change how many cells there are.",
    },
    "merge_size_voxels": {
        "range": (0, 600), "log": False, "step": 5,
        "help": "fold cells smaller than this into their closest neighbour "
                "after the watershed. 0 disables merging.",
    },
    "merge_threshold_deg": {
        "range": (0.0, 0.6), "log": False, "step": 0.005,
        "help": "a fragment only merges if its mean orientation is within this "
                "of the neighbour's. Gates the merge so genuine interfaces survive.",
    },
    "percentile": {
        "range": (5.0, 95.0), "log": False, "step": 0.5,
        "help": "KAM percentile below which a voxel counts as cell interior. "
                "Higher = fewer, larger cells. (KAM baseline only.)",
    },
}

#: A reasonable starting point for the KAM baseline -- roughly the best cell
#: count a short hand search finds (306 cells, identity F1 0.85).  It is a
#: starting point for tuning, **not** a published value: the paper's KAM
#: comparator is the cell-count-matched partition in
#: ``continuation_results/kam_analysis/kam_cell_count_matched.npz``.
KAM_BASELINE_PARAMS = {
    "percentile": 45.0,
    "kam_radius_um": 1.2,
    "min_cell_size": 10,
}

#: The columns worth watching while tuning, in the study's priority order.
KEY_METRICS = [
    "n_cells_pred", "recovered_at_90", "contamination", "fused_true_cells",
    "strict_split_true_cells", "cell_count_error", "identity_f1",
    "ari", "boundary_assd_um",
]


def run(phantom: Phantom, params: dict, *, method: str = "flood fill", seed: int = 0):
    """Segment with either arm from one parameter dict.

    ``method`` is ``"flood fill"`` or ``"KAM threshold"``.  Unknown keys are
    ignored, so one dict can carry the settings of both arms.
    """

    if method == "flood fill":
        wanted = set(BEST_PARAMS)
        return segment(phantom, seed=seed, **{k: v for k, v in params.items() if k in wanted})
    if method == "KAM threshold":
        wanted = {"percentile", "kam_radius_um", "min_cell_size", "connectivity"}
        return segment_kam_baseline(
            phantom, **{k: v for k, v in params.items() if k in wanted}
        )
    raise ValueError(f"unknown method {method!r}; use 'flood fill' or 'KAM threshold'")


def sweep(
    phantom: Phantom,
    name: str,
    values,
    *,
    base: dict | None = None,
    method: str = "flood fill",
    seed: int = 0,
    identity: bool = True,
    verbose: bool = True,
):
    """Vary one parameter, hold the rest, and score every value.

    Returns a DataFrame indexed by the swept value.  Configurations that
    produce no accepted markers are kept as a row of NaN with the reason in
    ``error`` -- they are genuinely invalid, not a bug, and the study counts
    them in its parameter-difficulty estimate.
    """

    import pandas as pd

    if base is None:
        base = KAM_BASELINE_PARAMS if method == "KAM threshold" else BEST_PARAMS
    rows = []
    for value in values:
        params = dict(base)
        params[name] = value
        row = {name: value, "error": None}
        try:
            result = run(phantom, params, method=method, seed=seed)
            row.update(score(phantom, result, identity=identity))
        except (RuntimeError, ValueError) as error:
            row["error"] = str(error).split(".")[0]
        rows.append(row)
        if verbose:
            note = row["error"] or (
                f"{row['n_cells_pred']:4d} cells "
                f"({row['cell_count_error']:+d}), F1 {row['identity_f1']:.3f}"
            )
            print(f"{name} = {value!s:<10} {note}")
    return pd.DataFrame(rows).set_index(name)


def sweep_plot(frame, *, target: int = 360, figsize=(11, 4)):
    """Plot a :func:`sweep`: recovered cell count and identity F1 against it."""

    import matplotlib.pyplot as plt

    x = frame.index.to_numpy(dtype=float)
    fig, (left, right) = plt.subplots(1, 2, figsize=figsize)

    left.axhline(target, color="0.6", linestyle="--", linewidth=1, label="truth")
    left.plot(x, frame["n_cells_pred"], "o-", color="#1f77b4")
    left.set_ylabel("predicted cells")
    left.legend(fontsize=9)

    right.plot(x, frame["identity_f1"], "o-", color="#2ca02c", label="identity F1")
    if "ari" in frame:
        right.plot(x, frame["ari"], "s--", color="#9467bd", alpha=0.7, label="ARI")
    right.set_ylabel("score")
    right.legend(fontsize=9)

    for ax in (left, right):
        ax.set_xlabel(frame.index.name)
        ax.grid(alpha=0.25)
        if x.min() > 0 and x.max() / max(x.min(), 1e-12) > 30:
            ax.set_xscale("log")
    fig.tight_layout()
    return fig


def grid(
    phantom: Phantom,
    name_a: str,
    values_a,
    name_b: str,
    values_b,
    *,
    base: dict | None = None,
    method: str = "flood fill",
    seed: int = 0,
    verbose: bool = True,
):
    """Sweep two parameters together.  Returns a tidy DataFrame, one row each.

    Costs one segmentation per combination (~3 s), so keep the lists short --
    a 4 x 4 grid is about a minute.
    """

    import pandas as pd

    if base is None:
        base = KAM_BASELINE_PARAMS if method == "KAM threshold" else BEST_PARAMS
    rows = []
    for a in values_a:
        for b in values_b:
            params = dict(base)
            params[name_a], params[name_b] = a, b
            row = {name_a: a, name_b: b, "error": None}
            try:
                result = run(phantom, params, method=method, seed=seed)
                row.update(score(phantom, result))
            except (RuntimeError, ValueError) as error:
                row["error"] = str(error).split(".")[0]
            rows.append(row)
        if verbose:
            print(f"{name_a} = {a!s:<10} done")
    return pd.DataFrame(rows)


def grid_plot(frame, *, metric: str = "cell_count_error", figsize=(7, 5), annotate=True):
    """Heatmap of a :func:`grid`.  Diverging around zero for count error."""

    import matplotlib.pyplot as plt

    name_a, name_b = frame.columns[0], frame.columns[1]
    table = frame.pivot(index=name_a, columns=name_b, values=metric)
    values = table.to_numpy(dtype=float)

    diverging = metric in {"cell_count_error", "count_error_percent"}
    limit = np.nanmax(np.abs(values)) if diverging else None
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        values, cmap="RdBu_r" if diverging else "viridis", aspect="auto",
        vmin=-limit if diverging else None, vmax=limit if diverging else None,
    )
    ax.set_xticks(range(table.shape[1]), [f"{v:g}" for v in table.columns])
    ax.set_yticks(range(table.shape[0]), [f"{v:g}" for v in table.index])
    ax.set_xlabel(name_b)
    ax.set_ylabel(name_a)
    ax.set_title(metric)
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.3g}", ha="center", va="center",
                            fontsize=8, color="black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- the live tuner


#: Which sliders each arm owns.  The two arms share parameter *names*
#: (``min_cell_size``, ``kam_radius_um``) but not values -- a sensible minimum
#: marker size is ~116 voxels for the flood fill and ~10 for KAM components --
#: so each arm gets its own widgets and its own stored settings.
ARM_PARAMETERS = {
    "flood fill": list(BEST_PARAMS) + ["merge_size_voxels", "merge_threshold_deg"],
    "KAM threshold": ["percentile", "kam_radius_um", "min_cell_size"],
}


class Tuner:
    """Sliders for both arms, with the slice view and the scores beside them.

    Segmenting takes a couple of seconds, so nothing runs until you press
    **Run**.  Moving the ``z`` slider or switching the view only redraws the
    result that is already computed, which is instant.

    Each arm remembers its own settings, so switching back and forth does not
    lose them.  ``keep`` stores the current run under a name; ``results`` is
    the dict of everything kept, ready for ``score_table`` or
    ``show_size_distribution``.
    """

    def __init__(
        self,
        phantom: Phantom,
        params: dict | None = None,
        *,
        method: str = "flood fill",
        z: int | None = None,
        run_now: bool = True,
        figsize: tuple[float, float] = (12.5, 7.0),
    ):
        self.phantom = phantom
        self.figsize = figsize
        self.result: Result | None = None
        self.scores: dict | None = None
        self.diagnosis: dict | None = None
        self.results: dict[str, Result] = {}
        self._z = phantom.shape[0] // 2 if z is None else int(z)
        self._method = method
        self._values = {
            "flood fill": dict(CAPPED_START),
            "KAM threshold": dict(KAM_BASELINE_PARAMS),
        }
        if params:
            self._values[method].update(
                {k: v for k, v in params.items() if k in ARM_PARAMETERS[method]}
            )
        self._build()
        if run_now:
            self.run()

    # -- the parts that work with or without widgets ------------------------

    @property
    def method(self) -> str:
        if self._widgets:
            return self._widgets["method"].value
        return self._method

    @property
    def params(self) -> dict:
        """The parameters the current arm would use, as a plain dict."""

        self._read_widgets()
        chosen = dict(self._values[self.method])
        # -1 disables the global threshold.  It is never written back onto the
        # slider, which only carries positive values.
        if "global_threshold_deg" in chosen and not self._use_global():
            chosen["global_threshold_deg"] = -1.0
        return chosen

    def all_params(self) -> dict:
        """Both arms' current settings, for saving or reporting."""

        current = self.method
        return {
            arm: (self.params if arm == current else dict(values))
            for arm, values in self._values.items()
        }

    def run(self) -> Result | None:
        """Segment with the current settings and score the outcome."""

        self._status("running ...", "info")
        try:
            self.result = run(
                self.phantom, self.params, method=self.method, seed=self._seed()
            )
            self.scores = score(self.phantom, self.result)
            self.diagnosis = None
            self._status(self._headline(), "success")
        except (RuntimeError, ValueError) as error:
            self.result, self.scores, self.diagnosis = None, None, None
            self._status(f"invalid configuration -- {error}", "danger")
        self._draw()
        return self.result

    def keep(self, name: str | None = None) -> str:
        """Store the current run for later comparison.  Returns its name."""

        if self.result is None:
            raise RuntimeError("nothing to keep -- the last run was invalid.")
        if name is None:
            name = f"run {len(self.results) + 1} ({self.method})"
        self.results[name] = self.result
        self._refresh_kept()
        return name

    def table(self):
        """Score every kept run side by side."""

        if not self.results:
            raise RuntimeError("nothing kept yet -- press Keep, or call .keep().")
        return score_table(self.phantom, self.results)[KEY_METRICS]

    def reset(self) -> None:
        """Put both arms back to their published starting points."""

        self._values = {
            "flood fill": dict(CAPPED_START),
            "KAM threshold": dict(KAM_BASELINE_PARAMS),
        }
        self._write_widgets()
        self._status("sliders reset to the published parameters.", "info")

    def figure(self, view: str | None = None):
        """Build the current figure without displaying it."""

        if self.result is None:
            return None
        view = view or (self._widgets["view"].value if self._widgets else "panels")
        width, height = self.figsize
        if view == "panels":
            return show_slice(self.phantom, self.result, z=self._z,
                              match_colours=self._match_colours(),
                              figsize=self.figsize)
        if view == "three slices":
            return show_slices(self.phantom, self.result,
                               match_colours=self._match_colours(),
                               figsize_per_row=width / 3.45)
        if view == "errors":
            if self.diagnosis is None:
                self.diagnosis = cell_diagnosis(self.phantom, self.result)
            return show_errors(self.phantom, self.result,
                               z_values=[self._z], diagnosis=self.diagnosis,
                               figsize_per_panel=min(height, width / 2))
        raise ValueError(f"unknown view {view!r}")

    def _seed(self) -> int:
        if self.method != "flood fill":
            return 0
        return int(self._widgets["seed"].value) if self._widgets else 0

    def _use_global(self) -> bool:
        if self.method != "flood fill":
            return True
        if self._widgets:
            return bool(self._widgets["use_global"].value)
        return float(self._values["flood fill"]["global_threshold_deg"]) > 0

    def _match_colours(self) -> bool:
        return bool(self._widgets["match"].value) if self._widgets else True

    def _headline(self) -> str:
        s = self.scores
        return (
            f"recovered@90 {s['recovered_at_90']}/{s['n_cells_truth']}"
            f" ({s['recovery_rate_at_90']:.1%})   "
            f"contamination {s['contamination']:.4f}   "
            f"fused {s['fused_true_cells']}   split {s['strict_split_true_cells']}"
            f"<br>{s['n_cells_pred']} cells ({s['cell_count_error']:+d})   "
            f"ARI {s['ari']:.3f}   boundary {s['boundary_assd_um']:.3f} um"
        )

    # -- widget construction ------------------------------------------------

    @staticmethod
    def _key(arm: str, name: str) -> str:
        return f"{arm}:{name}"

    def _build(self) -> None:
        try:
            import ipywidgets as W
        except ImportError:
            self._widgets = None
            self.panel = None
            return

        self._widgets = w = {}
        style = {"description_width": "150px"}
        wide = W.Layout(width="440px")

        def slider(arm, name):
            guide = PARAMETER_GUIDE[name]
            low, high = guide["range"]
            value = self._values[arm][name]
            common = dict(description=name, continuous_update=False,
                          readout_format=".4g", style=style, layout=wide,
                          tooltip=guide["help"])
            if guide["log"]:
                return W.FloatLogSlider(base=10, min=np.log10(low), max=np.log10(high),
                                        step=0.01, value=value, **common)
            if name == "min_cell_size":
                common["readout_format"] = "d"
                return W.IntSlider(min=int(low), max=int(high),
                                   step=int(guide["step"]), value=int(value), **common)
            return W.FloatSlider(min=low, max=high, step=guide["step"],
                                 value=value, **common)

        for arm, names in ARM_PARAMETERS.items():
            for name in names:
                key = self._key(arm, name)
                w[key] = slider(arm, name)
                w[key].observe(self._on_parameter, names="value")

        w["use_global"] = W.Checkbox(
            value=self._values["flood fill"]["global_threshold_deg"] > 0,
            description="use global threshold", indent=False,
        )
        w["use_global"].observe(self._on_parameter, names="value")
        w["seed"] = W.IntSlider(description="seed", value=0, min=0, max=19, step=1,
                                continuous_update=False, style=style, layout=wide)
        w["method"] = W.ToggleButtons(
            options=list(ARM_PARAMETERS), value=self._method,
            style={"button_width": "150px"},
        )
        w["method"].observe(self._on_method, names="value")

        w["footprint_info"] = W.HTML()
        w["kam_info"] = W.HTML()
        w["status"] = W.HTML()
        w["kept"] = W.HTML()
        w["out"] = W.Output()
        w["help"] = W.HTML(
            "<div style='font-size:0.85em;color:#666;max-width:460px'>"
            + "<br>".join(f"<b>{name}</b> &mdash; {guide['help']}"
                          for name, guide in PARAMETER_GUIDE.items())
            + "</div>"
        )

        w["run"] = W.Button(description="Run", button_style="primary", icon="play")
        w["run"].on_click(lambda _: self.run())
        w["keep"] = W.Button(description="Keep", icon="bookmark")
        w["keep"].on_click(self._on_keep)
        w["reset"] = W.Button(description="Reset", icon="undo")
        w["reset"].on_click(lambda _: self.reset())
        w["clear"] = W.Button(description="Clear kept", icon="trash")
        w["clear"].on_click(self._on_clear)

        w["z"] = W.IntSlider(description="z slice", value=self._z, min=0,
                             max=self.phantom.shape[0] - 1, step=1,
                             continuous_update=False, style=style, layout=wide)
        w["z"].observe(self._on_view, names="value")
        w["view"] = W.Dropdown(description="view",
                               options=["panels", "three slices", "errors"],
                               value="panels", style=style,
                               layout=W.Layout(width="300px"))
        w["view"].observe(self._on_view, names="value")
        w["match"] = W.Checkbox(value=True, indent=False,
                                description="match prediction colours to truth")
        w["match"].observe(self._on_view, names="value")

        key = self._key
        w["box:flood fill"] = W.VBox([
            w[key("flood fill", "local_threshold_deg")],
            W.HBox([w[key("flood fill", "global_threshold_deg")], w["use_global"]]),
            w[key("flood fill", "footprint_tolerance")],
            w[key("flood fill", "footprint_radius_um")], w["footprint_info"],
            w[key("flood fill", "min_cell_size")],
            w[key("flood fill", "kam_radius_um")], w["kam_info"],
            w["seed"],
        ])
        w["box:KAM threshold"] = W.VBox([
            w[key("KAM threshold", "percentile")],
            w[key("KAM threshold", "kam_radius_um")],
            w[key("KAM threshold", "min_cell_size")],
        ])
        w["parameters"] = W.VBox([w[f"box:{self._method}"]])

        controls = W.VBox([
            w["method"],
            w["parameters"],
            W.HBox([w["run"], w["keep"], w["reset"], w["clear"]]),
            w["status"],
            w["kept"],
            W.Accordion(children=[w["help"]], titles=("what the parameters do",),
                        selected_index=None),
        ])
        self.panel = W.VBox([
            W.HBox([controls, w["out"]]),
            W.HBox([w["view"], w["z"], w["match"]]),
        ])
        self._update_info()
        self._refresh_kept()

    # -- widget callbacks ---------------------------------------------------

    def _read_widgets(self) -> None:
        """Pull the active arm's sliders into its stored settings."""

        if not self._widgets:
            return
        arm = self.method
        for name in ARM_PARAMETERS[arm]:
            value = self._widgets[self._key(arm, name)].value
            self._values[arm][name] = int(value) if name == "min_cell_size" else value
        self._z = int(self._widgets["z"].value)

    def _write_widgets(self) -> None:
        if not self._widgets:
            return
        for arm, names in ARM_PARAMETERS.items():
            for name in names:
                widget = self._widgets[self._key(arm, name)]
                low, high = PARAMETER_GUIDE[name]["range"]
                widget.value = min(max(self._values[arm][name], low), high)
        self._widgets["use_global"].value = (
            float(self._values["flood fill"]["global_threshold_deg"]) > 0
        )
        self._update_info()

    def _on_parameter(self, _change) -> None:
        self._update_info()

    def _on_method(self, change) -> None:
        w = self._widgets
        w["parameters"].children = [w[f"box:{change['new']}"]]
        self._update_info()

    def _on_view(self, _change) -> None:
        if self._widgets:
            self._z = int(self._widgets["z"].value)
        self._draw()

    def _on_keep(self, _button) -> None:
        try:
            name = self.keep()
        except RuntimeError as error:
            self._status(str(error), "warning")
        else:
            self._status(f"kept as '{name}'.  {self._headline()}", "success")

    def _on_clear(self, _button) -> None:
        self.results.clear()
        self._refresh_kept()

    def _update_info(self) -> None:
        """Show how many voxels the current radii rasterise to."""

        if not self._widgets:
            return
        w = self._widgets
        grey = "<span style='font-size:0.85em;color:#666'>{}</span>"
        arm = self.method
        if arm == "flood fill":
            radius = w[self._key(arm, "footprint_radius_um")].value
            w["footprint_info"].value = grey.format(
                f"&nbsp;&nbsp;&rarr; {self.phantom.footprint_voxels(radius)}"
                " voxels in the neighbourhood"
            )
            global_slider = w[self._key(arm, "global_threshold_deg")]
            global_slider.disabled = not w["use_global"].value
        kam_radius = w[self._key(arm, "kam_radius_um")].value
        w["kam_info"].value = grey.format(
            f"&nbsp;&nbsp;&rarr; {self.phantom.footprint_voxels(kam_radius)}"
            " voxels in the KAM kernel"
        )

    def _refresh_kept(self) -> None:
        if not self._widgets:
            return
        names = ", ".join(self.results) or "nothing kept yet"
        self._widgets["kept"].value = (
            f"<span style='font-size:0.85em;color:#666'>kept: {names} "
            "&mdash; call <code>.table()</code> to compare</span>"
        )

    def _status(self, message: str, style: str = "info") -> None:
        colour = {"info": "#666", "success": "#1a7f37", "warning": "#9a6700",
                  "danger": "#b42318"}[style]
        if self._widgets:
            self._widgets["status"].value = (
                f"<div style='color:{colour};font-family:monospace;"
                f"font-size:0.9em;max-width:470px'>{message}</div>"
            )
        else:
            print(message)

    def _draw(self) -> None:
        import matplotlib.pyplot as plt

        if not self._widgets:
            return
        from IPython.display import clear_output, display

        with self._widgets["out"]:
            clear_output(wait=True)
            figure = self.figure()
            if figure is None:
                print("no result to draw -- fix the parameters and press Run.")
                return
            display(figure)
            plt.close(figure)

    def _ipython_display_(self):
        from IPython.display import display

        if self.panel is None:
            raise RuntimeError(
                "ipywidgets is not available in this kernel, so the tuner has "
                "no controls.  Use lab.segment / lab.sweep / lab.grid instead."
            )
        display(self.panel)


def tune(phantom: Phantom, params: dict | None = None, **kwargs) -> Tuner:
    """Open the interactive tuner.  See :class:`Tuner`."""

    return Tuner(phantom, params, **kwargs)
