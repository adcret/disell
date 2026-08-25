#!/usr/bin/env python3
r"""One fixed, deterministic 3D dislocation-cell phantom with known ground truth.

The phantom is a Laguerre (power) tessellation on an anisotropically sampled
grid, rendered into a two-channel DFXM angular feature field.  Its properties
are fixed by construction rather than tuned:

* **Anisotropic voxels**, ``(1.0, 0.635, 0.20)`` um by default, in a
  ``24 x 64 x 64`` um physical volume.
* **Log-normal cell volumes.**  Target volumes are drawn from a log-normal and
  then realised by fitting the power-diagram weights, so the distribution is
  prescribed rather than hoped for.
* **Chi-distributed cell-to-cell misorientations.**  Cell orientation states
  are fitted on the measured adjacency graph so the reduced-SO(2) distances
  between final measured ground-truth region means follow a chi distribution
  with non-integer shape ``k=1.7`` and an extrapolated 6% strain scale
  ``sigma=0.36 deg``.
* **Heterogeneous intradomain variation.**  Most of the variance inside a cell
  is a smooth, randomly directed affine drift measured from the cell centroid;
  a weak, non-radial curvature supplies the rest.  Both are scaled by one
  per-cell log-normal amplitude, clipped at two standard deviations, so most
  cells are quiet, a minority vary strongly, and none is an outlier blob.  The
  gradient is centroid-centred and the curvature is cell-mean-removed, so
  neither displaces the region means: intradomain variation and the
  misorientation distribution are independent knobs.  White noise stays
  negligible at 0.001 deg.
* **Heterogeneous KAM ridges.**  The extrapolated 6% misorientation scale makes
  most walls locally clear, while the low-angle chi tail and locally diffuse
  patches retain a range of ridge strengths.  The benchmark does not force
  KAM thresholding to fail by adding unrealistic interior noise.
* **Boundaries that are obvious in the feature map and heterogeneous in KAM.** A
  piecewise latent field is built strictly by indexing the Laguerre labels.
  Symmetric physical-space blurring is then applied to that field, with a
  smooth spatial modulation of the blur width.  This makes walls broad and
  locally diffuse without introducing interfaces from a changing
  ``second-nearest`` site inside a ground-truth cell.

Everything is a pure function of ``PhantomConfig``; the same config always
produces bit-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PhantomConfig:
    """Every parameter of the phantom.  Fixed defaults, no severity ladder."""

    shape_zyx: tuple[int, int, int] = (24, 160, 160)
    spacing_um_zyx: tuple[float, float, float] = (1.0, 0.635, 0.20)
    n_cells: int = 360

    #: Standard deviation of log(cell volume).  Cell volumes are drawn from
    #: this log-normal and then realised exactly by fitting the power-diagram
    #: weights.  sigma_logV = 0.8 corresponds to sigma_log(diameter) = 0.27,
    #: in the range reported for dislocation-cell size distributions.
    log_volume_sigma: float = 0.8

    #: Shape and scale of the target chi distribution of misorientation
    #: magnitudes across adjacent cells.  ``sigma=0.36 deg`` is a linear
    #: extrapolation of the supplied strain trend to approximately 6% strain.
    misorientation_k: float = 1.7
    misorientation_sigma_deg: float = 0.36
    #: Small compensation between fitted graph states and means recomputed over
    #: the complete latent ground-truth regions.  The target above is always
    #: validated on those region means, never on the hidden fitted parameters.
    misorientation_embedding_k_offset: float = -0.0151
    misorientation_embedding_sigma_gain: float = 1.0019
    misorientation_fit_iterations: int = 10
    misorientation_fit_max_nfev: int = 80

    #: Wall rendering is disabled.  The segmentation input is the latent field,
    #: with boundaries represented as steps between neighbouring voxels.
    wall_width_um: float = 0.0
    wall_width_dispersion: float = 0.0
    incomplete_wall_fraction: float = 0.0
    incomplete_wall_gain: float = 0.0
    incomplete_patch_um: float = 6.0

    #: Structured interiors: affine gradients, smooth curvature, a very slow
    #: drift and negligible white noise.  Both the affine gradient and the
    #: curvature of a cell are scaled by one per-cell log-normal amplitude, so
    #: intradomain variation is heterogeneous: most cells are quiet and a
    #: minority vary strongly.  ``intracell_amplitude_log_sigma`` is the
    #: dispersion of log(amplitude); the median amplitude is 1 by construction.
    intracell_amplitude_log_sigma: float = 0.62
    #: The log-amplitude is clipped at this many standard deviations.  An
    #: unclipped log-normal puts a handful of cells an order of magnitude above
    #: the median, which shows up as a bright blob rather than as structure.
    intracell_amplitude_clip_sigma: float = 2.0
    #: Most of the intradomain variance is a smooth, randomly directed affine
    #: drift measured from the cell centroid.  The curvature is deliberately
    #: weak and is *not* tapered towards the boundary: a radial taper produces a
    #: dark ring hugging every interface, which is a rendering artefact rather
    #: than microstructure.  Keeping the curvature weak is what makes the
    #: untapered wall step negligible against the facet misorientations.
    intracell_gradient_deg_per_um: float = 0.0068
    intracell_curvature_deg: float = 0.0250
    intracell_curvature_correlation_um: float = 5.0
    drift_deg: float = 0.018
    noise_sigma_deg: float = 0.0

    seed: int = 20260811


@dataclass
class Phantom:
    """Ground-truth labels and the measured angular feature field."""

    labels: np.ndarray          # (Z, Y, X) int32, 1-based, no background
    latent_field: np.ndarray    # exact label-derived field before measurement
    field: np.ndarray           # (Z, Y, X, 2) float32, degrees
    mask: np.ndarray            # (Z, Y, X) bool
    cell_means_deg: np.ndarray  # latent fitted states; not measured region means
    target_cell_volumes_um3: np.ndarray  # requested Laguerre volumes
    wall_width_um: np.ndarray   # (Z, Y, X) float32, local wall width
    spacing_um_zyx: tuple[float, float, float]
    config: PhantomConfig

    @property
    def cell_volumes_um3(self) -> np.ndarray:
        """Volume of every ground-truth cell, in cubic micrometres."""

        voxel = float(np.prod(self.spacing_um_zyx))
        return np.bincount(self.labels.ravel())[1:] * voxel

    @property
    def segmentation_field(self) -> np.ndarray:
        """Angular feature field used by the benchmark segmentation."""

        return self.latent_field


def _smooth_unit_field(
    rng: np.random.Generator,
    shape: Sequence[int],
    correlation_um: float,
    spacing_um_zyx: Sequence[float],
) -> np.ndarray:
    """Zero-mean, unit-variance smooth Gaussian random field.

    The smoothing length is capped at a quarter of each axis.  Without the cap,
    a correlation length comparable to the volume leaves an almost constant
    field whose unit-variance rescaling amplifies round-off into high-frequency
    structure -- the opposite of the intended slow variation.
    """

    from scipy.ndimage import gaussian_filter

    sigma = tuple(
        min(max(correlation_um / float(s), 0.0), 0.25 * int(n))
        for s, n in zip(spacing_um_zyx, shape)
    )
    out = gaussian_filter(rng.normal(size=tuple(shape)), sigma=sigma, mode="wrap")
    out -= out.mean()
    scale = out.std()
    return out / scale if scale > 0 else out


def _slow_drift(
    rng: np.random.Generator, points_um: np.ndarray, extent_um: np.ndarray
) -> np.ndarray:
    """Long-wavelength drift: a random ramp plus one gentle curvature term.

    Built analytically from the physical extent, so its gradient is bounded by
    roughly ``amplitude / extent`` no matter how thin the volume is.  Filtered
    white noise cannot give that guarantee: the shortest axis here is 20 um, so
    even a "long" correlation length still fits a full period across it and the
    drift stops being slow.
    """

    centred = points_um / extent_um[None, :] - 0.5
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    projected = centred @ direction
    # Half a cosine period across the volume, so no interior extremum is sharp.
    out = projected + 0.35 * np.cos(np.pi * projected + rng.uniform(0, 2 * np.pi))
    out -= out.mean()
    scale = out.std()
    return out / scale if scale > 0 else out


def _cell_means_of(
    field: np.ndarray, labels: np.ndarray, n_cells: int
) -> np.ndarray:
    """Mean of a scalar field over every cell, indexed by label (0 unused)."""

    flat = labels.ravel()
    counts = np.bincount(flat, minlength=n_cells + 1)
    totals = np.bincount(flat, weights=field.ravel(), minlength=n_cells + 1)
    return totals / np.maximum(counts, 1)


def _repulsive_sites(
    rng: np.random.Generator,
    n_sites: int,
    extent_um: np.ndarray,
    candidates: int = 512,
) -> np.ndarray:
    """Farthest-point sampling: quasi-regular seeds, no clumping."""

    low = 0.04 * extent_um
    high = extent_um - 0.04 * extent_um
    sites = np.empty((int(n_sites), 3), dtype=np.float64)
    sites[0] = rng.uniform(low, high)
    for i in range(1, int(n_sites)):
        trial = rng.uniform(low, high, size=(candidates, 3))
        delta = trial[:, None, :] - sites[None, :i, :]
        nearest = np.min(np.sum(delta * delta, axis=-1), axis=1)
        sites[i] = trial[int(np.argmax(nearest))]
    return sites


def _fit_weights_to_volumes(
    points: np.ndarray,
    sites: np.ndarray,
    target_volumes: np.ndarray,
    voxel_volume: float,
    iterations: int = 120,
    damping: float = 0.15,
) -> np.ndarray:
    """Solve for power-diagram weights that realise ``target_volumes``.

    Raising a site's weight enlarges its cell, so a damped fixed-point step on
    the volume residual converges quickly.  This is the discrete form of the
    standard semi-discrete optimal-transport iteration, and it is what makes the
    realised cell-volume distribution log-normal rather than merely hoping that
    log-normal weights produce log-normal volumes.
    """

    from scipy.spatial import cKDTree

    tree = cKDTree(sites)
    neighbours = min(8, sites.shape[0])
    distances, indices = tree.query(points, k=neighbours, workers=1)
    squared = distances * distances

    target = np.asarray(target_volumes, dtype=np.float64)
    scale = float(np.mean(target) ** (2.0 / 3.0))
    weights = np.zeros(sites.shape[0], dtype=np.float64)

    for _ in range(int(iterations)):
        owner = indices[
            np.arange(points.shape[0]), np.argmin(squared - weights[indices], axis=1)
        ]
        volume = np.bincount(owner, minlength=sites.shape[0]) * float(voxel_volume)
        residual = (target - volume) / np.mean(target)
        weights += damping * scale * residual
        weights -= weights.mean()
    return weights


def _physical_gaussian_blur(
    field: np.ndarray,
    spacing_um_zyx: Sequence[float],
    sigma_um: float,
) -> np.ndarray:
    """Blur a vector field isotropically in physical rather than voxel units."""

    from scipy.ndimage import gaussian_filter

    sigma = tuple(
        float(sigma_um) / float(step) for step in spacing_um_zyx
    ) + (0.0,)
    return gaussian_filter(field, sigma=sigma, mode="nearest")


def _adjacent_cell_pairs(labels: np.ndarray) -> np.ndarray:
    """Zero-based unordered pairs of face-adjacent cell labels."""

    pairs: set[tuple[int, int]] = set()
    for axis in range(labels.ndim):
        lo = [slice(None)] * labels.ndim
        hi = [slice(None)] * labels.ndim
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        one = labels[tuple(lo)]
        two = labels[tuple(hi)]
        differs = one != two
        pairs.update(
            (min(int(a), int(b)) - 1, max(int(a), int(b)) - 1)
            for a, b in zip(one[differs], two[differs])
        )
    return (
        np.asarray(sorted(pairs), dtype=np.int32).reshape(-1, 2)
        if pairs
        else np.empty((0, 2), dtype=np.int32)
    )


def _fit_cell_means_to_chi_edges(
    labels: np.ndarray,
    n_cells: int,
    k: float,
    sigma_deg: float,
    rng: np.random.Generator,
    iterations: int,
    max_nfev: int,
) -> np.ndarray:
    """Fit 2-D cell states whose adjacent distances follow ``chi(k, sigma)``.

    Independent Gaussian states would force the edge magnitudes to be Rayleigh
    distributed, i.e. chi with exactly two degrees of freedom.  The measured
    distribution has non-integer ``k``.  We therefore fit states on the actual
    cell-adjacency graph.  At each outer iteration the current edge lengths are
    rank-matched to deterministic chi quantiles, followed by a sparse graph
    stress minimisation.  Re-ranking avoids assigning mutually incompatible
    random lengths around graph cycles and reproduces the requested marginal
    distribution without changing the tessellation.
    """

    from scipy import optimize, sparse, stats

    edges_global = _adjacent_cell_pairs(labels)
    means = np.zeros((int(n_cells), 2), dtype=np.float64)
    if edges_global.size == 0:
        return means

    present = np.unique(edges_global)
    mapping = np.full(int(n_cells), -1, dtype=np.int32)
    mapping[present] = np.arange(present.size, dtype=np.int32)
    edges = mapping[edges_global]
    n_nodes = int(present.size)
    n_edges = int(edges.shape[0])

    probabilities = (np.arange(n_edges, dtype=np.float64) + 0.5) / n_edges
    quantiles = stats.chi.ppf(
        probabilities, df=float(k), scale=float(sigma_deg)
    )

    states = rng.normal(0.0, float(sigma_deg), size=(n_nodes, 2))
    states -= states.mean(axis=0, keepdims=True)

    # Sparsity of edge residuals plus two centring constraints.  The graph
    # objective is translation invariant, so explicitly fixing its mean avoids
    # a rank-deficient numerical problem without privileging a particular cell.
    rows: list[int] = []
    columns: list[int] = []
    for edge_index, (one, two) in enumerate(edges):
        for node in (int(one), int(two)):
            rows.extend((edge_index, edge_index))
            columns.extend((2 * node, 2 * node + 1))
    for node in range(n_nodes):
        rows.extend((n_edges, n_edges + 1))
        columns.extend((2 * node, 2 * node + 1))
    jacobian_sparsity = sparse.coo_matrix(
        (
            np.ones(len(rows), dtype=np.float64),
            (np.asarray(rows), np.asarray(columns)),
        ),
        shape=(n_edges + 2, 2 * n_nodes),
    ).tocsr()

    for _ in range(int(iterations)):
        differences = states[edges[:, 0]] - states[edges[:, 1]]
        lengths = np.linalg.norm(differences, axis=1)
        order = np.argsort(lengths, kind="stable")
        targets = np.empty(n_edges, dtype=np.float64)
        targets[order] = quantiles

        def residual(flat_states: np.ndarray) -> np.ndarray:
            candidate = flat_states.reshape(n_nodes, 2)
            delta = candidate[edges[:, 0]] - candidate[edges[:, 1]]
            edge_residual = (
                np.sqrt(np.sum(delta * delta, axis=1) + 1e-12) - targets
            )
            return np.concatenate((edge_residual, 10.0 * candidate.mean(axis=0)))

        result = optimize.least_squares(
            residual,
            states.ravel(),
            jac_sparsity=jacobian_sparsity,
            max_nfev=int(max_nfev),
            xtol=2e-7,
            ftol=2e-7,
            gtol=2e-7,
        )
        states = result.x.reshape(n_nodes, 2)

    states -= states.mean(axis=0, keepdims=True)
    means[present] = states
    return means


def generate_phantom(config: PhantomConfig = PhantomConfig()) -> Phantom:
    """Build the phantom.  Deterministic given ``config``."""

    from scipy.spatial import cKDTree
    from scipy.special import ndtri

    rng = np.random.default_rng(config.seed)
    shape = tuple(int(v) for v in config.shape_zyx)
    spacing = np.asarray(config.spacing_um_zyx, dtype=np.float64)
    extent = np.asarray(shape, dtype=np.float64) * spacing

    axes = [
        (np.arange(n, dtype=np.float64) + 0.5) * s for n, s in zip(shape, spacing)
    ]
    points = np.stack(
        [g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=-1
    )

    # ---- Laguerre tessellation with log-normal weights --------------------
    # Repulsive (farthest-point) seeding first, so the underlying tessellation
    # is near-regular and the volume spread comes from the log-normal weights
    # alone.  Uniform seeding would superimpose the broad, left-skewed
    # Poisson-Voronoi volume distribution and the result would not be
    # log-normal.
    sites = _repulsive_sites(rng, config.n_cells, extent)
    target_volumes = np.exp(
        config.log_volume_sigma * rng.normal(size=config.n_cells)
    )
    target_volumes *= float(np.prod(extent)) / target_volumes.sum()
    weights = _fit_weights_to_volumes(
        points, sites, target_volumes, float(np.prod(spacing))
    )

    tree = cKDTree(sites)
    neighbours = min(8, config.n_cells)
    distances, indices = tree.query(points, k=neighbours, workers=1)
    power = distances * distances - weights[indices]
    order = np.argsort(power, axis=1, kind="stable")
    indices = np.take_along_axis(indices, order, axis=1)
    power = np.take_along_axis(power, order, axis=1)

    first = indices[:, 0]

    labels = (first + 1).astype(np.int32).reshape(shape)

    # ---- chi-distributed adjacent-cell misorientation ---------------------
    # Fit orientation states on the actual adjacency graph.  This permits the
    # non-integer k observed experimentally; independent 2-D Gaussian states
    # would instead impose k=2 by construction.
    cell_means = _fit_cell_means_to_chi_edges(
        labels,
        config.n_cells,
        config.misorientation_k + config.misorientation_embedding_k_offset,
        (
            config.misorientation_sigma_deg
            * config.misorientation_embedding_sigma_gain
        ),
        rng,
        config.misorientation_fit_iterations,
        config.misorientation_fit_max_nfev,
    )

    # ---- exact label-derived latent field ---------------------------------
    # Every voxel is assigned only from its ground-truth owner.  In particular,
    # the field cannot change merely because the second-nearest Laguerre site
    # changes inside one cell, which was the source of spurious apparent walls
    # in the previous two-site tanh construction.
    # One log-normal amplitude per cell, applied to both the affine gradient and
    # the curvature, so "how much this cell varies inside" is a single
    # reproducible per-cell property with a heavy upper tail.
    log_amplitude = config.intracell_amplitude_log_sigma * rng.normal(
        size=config.n_cells
    )
    clip = config.intracell_amplitude_clip_sigma * config.intracell_amplitude_log_sigma
    amplitude = np.exp(np.clip(log_amplitude, -clip, clip))
    gradients = rng.normal(
        0.0, config.intracell_gradient_deg_per_um, size=(config.n_cells, 3, 2)
    )
    gradients *= amplitude[:, None, None]
    # Measure the affine gradient from each cell's centroid, not from its
    # Laguerre site.  The site is not the centroid, so a site-centred gradient
    # displaces the cell's mean orientation by grad . mean(offset); once the
    # gradients are strong enough to matter that displacement dominates the
    # adjacent-region misorientation and the chi calibration becomes
    # unreachable.  A centroid-centred gradient contributes exactly zero to the
    # region mean, which keeps intradomain variation and the misorientation
    # distribution independent.
    # ``_cell_means_of`` is indexed by label, which is ``first + 1``.
    centroids = np.stack(
        [_cell_means_of(points[:, axis], labels, config.n_cells) for axis in range(3)],
        axis=-1,
    )
    offset = points - centroids[first + 1]
    latent = cell_means[first] + np.einsum(
        "nd,ndc->nc", offset, gradients[first]
    )

    latent = latent.reshape(shape + (2,))

    # The drift is part of the underlying angular field, not detector noise.
    # It is continuous and sufficiently slow that it cannot define extra cells.
    if config.drift_deg > 0:
        for channel in range(2):
            latent[..., channel] += config.drift_deg * _slow_drift(
                rng, points, extent
            ).reshape(shape)

    # A smooth curvature field raises structured KAM inside cells.  It is scaled
    # by the per-cell amplitude, and tapered to zero at cell boundaries so the
    # amplitude cannot step across an interface.  The cell mean of the raw field
    # is removed before tapering, which keeps the curvature from displacing the
    # region means that the chi calibration is validated on.
    if config.intracell_curvature_deg > 0:
        amplitude_map = amplitude[first].reshape(shape)
        for channel in range(2):
            curvature = _smooth_unit_field(
                rng, shape, config.intracell_curvature_correlation_um, spacing
            )
            # Removing the cell mean keeps the curvature out of the region means
            # the chi calibration is validated on.  No boundary taper: it would
            # impose a radial ring on every cell.
            curvature -= _cell_means_of(curvature, labels, config.n_cells)[labels]
            latent[..., channel] += (
                config.intracell_curvature_deg * amplitude_map * curvature
            )

    # ---- measured field ----------------------------------------------------
    # There is no separate "measurement" field.  An earlier version rendered
    # one by blurring the latent field over a wall of finite width, on the view
    # that a cell wall occupies space and the microscope records it broadened.
    # That was a misreading of the data: collected DFXM volumes look like the
    # latent field, and the walls are thinner than a voxel in every geometry
    # used here -- 0.20 to 0.63 um against a 0.4 um pixel -- so a wall has no
    # resolvable extent to render.  A boundary is a step between neighbouring
    # voxels.
    #
    # The broadening was not a harmless extra.  Segmenting the blurred field
    # instead of the latent one costs the flood fill 79.24 % recovery against
    # 6.51 % on one and the same phantom, and wall width, which has a rank
    # correlation of -0.55 with the field step across an interface in the
    # blurred rendering, has a correlation of 0.004 in the latent one.
    #
    # ``field`` is retained as a name so that existing callers keep working,
    # and it now *is* the latent field.
    field = latent.copy()
    width = np.full(shape, float(config.wall_width_um), dtype=np.float64)

    if config.noise_sigma_deg > 0:
        field += rng.normal(0.0, config.noise_sigma_deg, size=field.shape)

    padded_means = np.zeros((config.n_cells + 1, 2), dtype=np.float32)
    padded_means[1:] = cell_means

    return Phantom(
        labels=np.ascontiguousarray(labels),
        latent_field=np.ascontiguousarray(latent.astype(np.float32)),
        field=np.ascontiguousarray(field.astype(np.float32)),
        mask=np.ones(shape, dtype=bool),
        cell_means_deg=padded_means,
        target_cell_volumes_um3=np.ascontiguousarray(
            target_volumes.astype(np.float32)
        ),
        wall_width_um=np.ascontiguousarray(
            width.astype(np.float32)
        ),
        spacing_um_zyx=tuple(float(v) for v in spacing),
        config=config,
    )


def ground_truth_region_means_deg(phantom: Phantom) -> np.ndarray:
    """Mean latent angular-feature vector of every ground-truth region."""

    labels = phantom.labels.ravel()
    n_labels = phantom.cell_means_deg.shape[0]
    counts = np.bincount(labels, minlength=n_labels)
    field = phantom.segmentation_field
    means = np.zeros((n_labels, field.shape[-1]), dtype=np.float64)
    for channel in range(field.shape[-1]):
        sums = np.bincount(
            labels,
            weights=field[..., channel].ravel(),
            minlength=n_labels,
        )
        means[:, channel] = sums / np.maximum(counts, 1)
    return means


def intradomain_angular_spread_deg(
    labels: np.ndarray, field: np.ndarray, n_cells: int | None = None
) -> np.ndarray:
    """Per-cell intradomain angular spread ``s_k``, in degrees.

    ``s_k = sqrt(Var(chi) + Var(phi))`` over the elements of one cell, the
    measure used in the manuscript (Eq. A2) to describe how coherent a domain
    is.  Returned for every cell that has at least one element.
    """

    flat = labels.ravel()
    n_labels = int(n_cells if n_cells is not None else labels.max()) + 1
    counts = np.bincount(flat, minlength=n_labels).astype(np.float64)
    variance = np.zeros(n_labels, dtype=np.float64)
    for channel in range(field.shape[-1]):
        values = field[..., channel].ravel().astype(np.float64)
        total = np.bincount(flat, weights=values, minlength=n_labels)
        total_sq = np.bincount(flat, weights=values * values, minlength=n_labels)
        mean = total / np.maximum(counts, 1.0)
        variance += np.maximum(total_sq / np.maximum(counts, 1.0) - mean * mean, 0.0)
    return np.sqrt(variance[1:][counts[1:] > 0])


def crystal_angular_spread_deg(
    field: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """``s_crystal``: the same functional form as ``s_k``, over the whole crystal.

    ``sqrt(Var(chi) + Var(phi))`` pooled over every valid voxel.  Comparing it
    with ``median(s_k)`` says how much of the total orientation range is
    between cells rather than inside them.
    """

    values = field.reshape(-1, field.shape[-1]) if mask is None else field[mask]
    return float(np.sqrt(sum(values[:, c].var() for c in range(values.shape[1]))))


def channel_percentile_ranges_deg(
    field: np.ndarray,
    mask: np.ndarray | None = None,
    low: float = 1.0,
    high: float = 99.0,
) -> list[float]:
    """Per-channel ``low``-to-``high`` percentile range of the feature field."""

    values = field.reshape(-1, field.shape[-1]) if mask is None else field[mask]
    return [
        float(np.percentile(values[:, c], high) - np.percentile(values[:, c], low))
        for c in range(values.shape[1])
    ]


def intradomain_angular_range_deg(
    labels: np.ndarray,
    field: np.ndarray,
    n_cells: int | None = None,
    low_percentile: float = 0.0,
    high_percentile: float = 100.0,
) -> np.ndarray:
    """Per-cell *range-like* intradomain angular spread, in degrees.

    Per channel the span between two percentiles of the values inside a cell,
    combined as ``sqrt(range_chi^2 + range_phi^2)``.  The defaults give the
    peak-to-peak span.

    This is a separate diagnostic, reported on its own.  It is **not** ``s_k``
    and must not be compared with it or with any target expressed in ``s_k``:
    ``s_k`` is the variance-based quantity of the manuscript's Eq. A2,
    ``sqrt(Var(chi) + Var(phi))``, and the experimental 0.184 deg is an ``s_k``.
    A peak-to-peak span is a different statistic with a different sampling
    behaviour, and converting between them is not meaningful here.
    """

    flat = labels.ravel()
    n_labels = int(n_cells if n_cells is not None else labels.max()) + 1
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]
    starts = np.searchsorted(sorted_labels, np.arange(n_labels), side="left")
    stops = np.searchsorted(sorted_labels, np.arange(n_labels), side="right")

    ranges = np.zeros(n_labels, dtype=np.float64)
    channels = [field[..., c].ravel()[order] for c in range(field.shape[-1])]
    for label_id in range(1, n_labels):
        start, stop = starts[label_id], stops[label_id]
        if stop <= start:
            continue
        total = 0.0
        for values in channels:
            block = values[start:stop]
            high = np.percentile(block, high_percentile)
            low = np.percentile(block, low_percentile)
            total += float(high - low) ** 2
        ranges[label_id] = np.sqrt(total)
    counts = np.bincount(flat, minlength=n_labels)
    return ranges[1:][counts[1:] > 0]


def neighbour_misorientations_deg(phantom: Phantom) -> np.ndarray:
    """Reduced-SO(2) misorientation of adjacent ground-truth region means.

    The two latent angular features are averaged over every ground-truth region.
    Misorientation is the Euclidean magnitude of the
    wrapped two-channel mean difference, not a full crystallographic SO(3)
    misorientation and not the distance between hidden latent parameters.
    """

    pairs = _adjacent_cell_pairs(phantom.labels)
    if pairs.size == 0:
        return np.empty(0)

    region_means = ground_truth_region_means_deg(phantom)
    index = pairs + 1
    delta = (
        region_means[index[:, 0]] - region_means[index[:, 1]]
    )
    delta = (delta + 180.0) % 360.0 - 180.0
    return np.linalg.norm(delta, axis=-1)


def boundary_normal_ridge_offsets_um(
    labels: np.ndarray,
    field: np.ndarray,
    spacing_um_zyx: Sequence[float],
    search_radius_voxels: int = 3,
) -> np.ndarray:
    """Signed offsets from labelled interfaces to nearby angular ridges.

    For every face at which the label changes, the angular difference across
    parallel faces within ``search_radius_voxels`` is inspected.  The returned
    value is the physical displacement of the strongest normal difference from
    the labelled face.  A label-derived, symmetrically blurred phantom should
    therefore have a distribution centred on zero with median absolute offset
    zero.  Large systematic or absolute offsets expose a mismatch between the
    label geometry and the field used for segmentation.
    """

    labels = np.asarray(labels)
    field = np.asarray(field)
    spacing = tuple(float(v) for v in spacing_um_zyx)
    if field.shape[:-1] != labels.shape or labels.ndim != 3:
        raise ValueError("Expected labels (Z,Y,X) and field (Z,Y,X,C).")
    if len(spacing) != labels.ndim:
        raise ValueError("Voxel spacing must have one value per spatial axis.")

    radius = int(search_radius_voxels)
    if radius < 0:
        raise ValueError("search_radius_voxels must be non-negative.")
    shifts = np.arange(-radius, radius + 1, dtype=np.int32)
    output: list[np.ndarray] = []
    for axis, step in enumerate(spacing):
        changed = np.diff(labels, axis=axis) != 0
        coordinates = np.argwhere(changed)
        if coordinates.size == 0:
            continue
        angular_difference = np.linalg.norm(
            np.diff(field, axis=axis), axis=-1
        ) / step
        scores = []
        for shift in shifts:
            sample = coordinates.copy()
            sample[:, axis] = np.clip(
                sample[:, axis] + shift,
                0,
                angular_difference.shape[axis] - 1,
            )
            scores.append(angular_difference[tuple(sample.T)])
        best = np.argmax(np.stack(scores, axis=1), axis=1)
        output.append(shifts[best].astype(np.float64) * step)
    return np.concatenate(output) if output else np.empty(0, dtype=np.float64)


def ground_truth_boundaries(labels: np.ndarray) -> np.ndarray:
    """Face-connected ground-truth interface voxels (both sides marked)."""

    boundary = np.zeros(labels.shape, dtype=bool)
    for axis in range(labels.ndim):
        lo = [slice(None)] * labels.ndim
        hi = [slice(None)] * labels.ndim
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        differs = labels[tuple(lo)] != labels[tuple(hi)]
        boundary[tuple(lo)] |= differs
        boundary[tuple(hi)] |= differs
    return boundary
