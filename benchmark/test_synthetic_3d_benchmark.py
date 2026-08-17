"""Tests for the synthetic 3D dislocation-cell benchmark.

Three things have to hold for the result to mean anything: the phantom really
has the properties it claims, the metrics point the way they claim, and the
figure draws exactly-one-pixel opaque boundaries over one shared background.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import bench_metrics
import figures
import oracle_analyze
import overnight_continue
import oracle_select
import phantom as phantom_module
import pipelines
import synthetic_3d_benchmark as benchmark
from phantom import PhantomConfig, generate_phantom

#: Small phantom for the fast tests.  Distributional claims are checked on the
#: real one, generated once for the whole module.
SMALL = PhantomConfig(shape_zyx=(8, 30, 30), n_cells=14, seed=3)


def test_surrogate_excludes_nonfinite_features_and_target(tmp_path, capsys):
    n_rows = 55
    data = {
        "config_key": [f"config-{i}" for i in range(n_rows)],
        "ari_mean": np.linspace(0.1, 0.9, n_rows),
    }
    for offset, name in enumerate(oracle_select.PARAMETER_NAMES):
        data[name] = np.linspace(1.0 + offset, 2.0 + offset, n_rows)
    frame = pd.DataFrame(data)
    frame.loc[0, "ari_mean"] = np.nan
    frame.loc[1, "ari_mean"] = np.inf
    frame.loc[2, "local_threshold_deg"] = -np.inf
    frame.loc[3, "kam_radius_um"] = np.nan

    report = oracle_analyze.surrogate(frame, tmp_path, seed=7)["ari_mean"]

    assert report["total_rows"] == 55
    assert report["retained_rows"] == 51
    assert report["excluded_rows"] == 4
    assert report["status"] == "ok"
    assert "total=55, retained=51, excluded=4" in capsys.readouterr().out
    predictions = pd.read_csv(tmp_path / "surrogate_predictions.csv")
    assert predictions["ari_mean_cv_prediction"].notna().sum() == 51


def test_continuation_plan_has_stable_unique_phantom_ids():
    specs = [
        overnight_continue.phantom_spec(strain, realization, difficulty)
        for strain in overnight_continue.STRAINS
        for realization in range(3)
        for difficulty in overnight_continue.DIFFICULTIES
    ]
    ids = [item[0] for item in specs]
    assert len(ids) == len(set(ids)) == 36
    assert all(item[1].spacing_um_zyx == (1.0, 0.4, 0.4) for item in specs)
    assert all(item[2]["diameter_to_volume_assumption"] ==
               "equivalent-sphere approximation" for item in specs)


def test_continuation_measurable_proxies_need_no_labels():
    field = np.arange(6 * 7 * 8 * 2, dtype=float).reshape(6, 7, 8, 2)
    proxies = overnight_continue.measurable_proxies(field, (1.0, 0.4, 0.4))
    assert proxies["nn_diff_p95_deg"] >= proxies["nn_diff_p90_deg"]
    assert proxies["cell_scale_proxy_um"] > 0
    assert proxies["anisotropy"] == pytest.approx(2.5)


@pytest.fixture(scope="module")
def full_phantom():
    return generate_phantom(PhantomConfig())


@pytest.fixture(scope="module")
def small_phantom():
    return generate_phantom(SMALL)


@pytest.fixture(scope="module")
def full_kam(full_phantom):
    return pipelines.masked_kam(
        full_phantom.field,
        full_phantom.mask,
        pipelines.isotropic_footprint(full_phantom.spacing_um_zyx, 1.0),
    )


# --------------------------------------------------------------------------
# Phantom
# --------------------------------------------------------------------------


def test_phantom_is_deterministic(small_phantom):
    again = generate_phantom(SMALL)
    assert np.array_equal(small_phantom.labels, again.labels)
    assert np.array_equal(small_phantom.field, again.field)
    assert np.array_equal(small_phantom.wall_width_um, again.wall_width_um)


def test_phantom_shape_spacing_and_labelling(small_phantom):
    assert small_phantom.labels.shape == SMALL.shape_zyx
    assert small_phantom.field.shape == SMALL.shape_zyx + (2,)
    assert small_phantom.spacing_um_zyx == SMALL.spacing_um_zyx
    # Anisotropic by construction: the z voxel is 2.5x the in-plane voxel.
    assert small_phantom.spacing_um_zyx[0] == 2.5 * small_phantom.spacing_um_zyx[1]
    assert np.all(small_phantom.labels > 0), "every voxel belongs to a cell"
    assert np.all(np.isfinite(small_phantom.field))


def test_cell_volumes_are_log_normal(full_phantom):
    volumes = full_phantom.cell_volumes_um3
    volumes = volumes[volumes > 0]
    assert volumes.size > 100
    log_volumes = np.log(volumes)
    # The realised dispersion tracks the requested one and log-volume is normal.
    assert log_volumes.std() == pytest.approx(
        full_phantom.config.log_volume_sigma, abs=0.15
    )
    assert abs(stats.skew(log_volumes)) < 0.6
    normality = stats.kstest(
        log_volumes, "norm", args=(log_volumes.mean(), log_volumes.std())
    )
    assert normality.pvalue > 0.05


def test_misorientations_are_chi_distributed(full_phantom):
    """Adjacent-region misorientations must follow the requested chi(k, sigma).

    ``k`` is not an integer, so the states are fitted on the cell-adjacency
    graph rather than drawn independently; independent 2-D Gaussians would
    force ``k = 2``.  The check is on the realised region means, not on the
    hidden fitted parameters.
    """

    values = phantom_module.neighbour_misorientations_deg(full_phantom)
    assert values.size > 500
    assert values.min() >= 0.0

    fitted_k, loc, fitted_sigma = stats.chi.fit(values, floc=0)
    assert loc == 0.0
    assert fitted_k == pytest.approx(full_phantom.config.misorientation_k, abs=0.15)
    assert fitted_sigma == pytest.approx(
        full_phantom.config.misorientation_sigma_deg, rel=0.10
    )
    # k < 2 puts real weight at low angle, which is what a KAM threshold has to
    # cope with; without that tail the wall network would be uniformly strong.
    assert 0.01 < float(np.mean(values < 0.10)) < 0.25


def test_segmentation_field_is_the_latent_field(full_phantom):
    assert np.shares_memory(full_phantom.segmentation_field, full_phantom.latent_field)
    assert full_phantom.segmentation_field.shape == full_phantom.labels.shape + (2,)


def test_labels_and_segmentation_field_agree(full_phantom):
    """No angular ridge may sit away from the labelled interface."""

    offsets = phantom_module.boundary_normal_ridge_offsets_um(
        full_phantom.labels,
        full_phantom.segmentation_field,
        full_phantom.spacing_um_zyx,
    )
    assert offsets.size > 0
    assert float(np.median(np.abs(offsets))) == 0.0


def test_cell_interiors_are_quiet_and_walls_are_well_marked(
    full_phantom, full_kam
):
    from scipy.ndimage import distance_transform_edt

    boundary = phantom_module.ground_truth_boundaries(full_phantom.labels)
    distance = distance_transform_edt(
        ~boundary, sampling=full_phantom.spacing_um_zyx
    )
    finite = np.isfinite(full_kam)
    core = (distance > 1.5) & finite

    core_kam = float(np.median(full_kam[core]))
    wall_kam = float(np.median(full_kam[boundary & finite]))
    # Interiors are quiet, so walls stand out by an order of magnitude.  The
    # percolation that follows is therefore caused by the low-angle tail of the
    # misorientation distribution, not by noisy interiors.
    assert core_kam < 0.02, core_kam
    assert wall_kam / core_kam > 5.0, wall_kam / core_kam

    step = float(np.median(phantom_module.neighbour_misorientations_deg(full_phantom)))
    assert step > 10.0 * core_kam


def test_wall_widths_are_heterogeneous(full_phantom):
    """Walls must not all share one blur width."""

    widths = full_phantom.wall_width_um
    low, median, high = np.percentile(widths, [10, 50, 90])
    assert low > 0.0
    assert high > 1.5 * low
    assert median == pytest.approx(full_phantom.config.wall_width_um, rel=0.8)


def test_kam_thresholding_has_no_safe_operating_point(full_phantom, full_kam):
    """Percolation at one end, unseeded cells at the other, on this phantom.

    This is the claim the benchmark exists to demonstrate, so it is asserted
    rather than left to a figure: over the whole threshold sweep there is no
    percentile at which the KAM markers neither leak across weak walls nor
    leave a large share of cells without a marker.
    """

    from scipy.ndimage import generate_binary_structure, label

    valid = full_phantom.mask & np.isfinite(full_kam)
    values = full_kam[valid]
    structure = generate_binary_structure(3, 1)
    n_cells = int(full_phantom.labels.max())

    reports = []
    for percentile in (5, 20, 40, 60, 80):
        threshold = float(np.percentile(values, percentile))
        markers, _ = label(valid & (full_kam < threshold), structure=structure)
        markers = pipelines._drop_small_labels(markers.astype(np.int32), 20)
        reports.append(bench_metrics.marker_confusion(markers, full_phantom.labels))

    unseeded = [r["cells_unseeded"] for r in reports]
    leaked = [r["percolating_marker_volume_fraction"] for r in reports]
    worst = [r["max_cells_per_marker"] for r in reports]

    # Low thresholds starve the marker set.
    assert unseeded[0] > 0.2 * n_cells
    # High thresholds percolate: the leaked volume fraction rises monotonically
    # and one marker ends up spanning most of the volume.  The *count* of
    # percolating markers is not monotone -- it peaks and then falls as the
    # leaking components merge into one another -- so severity is measured by
    # volume and by the worst marker instead.
    assert leaked == sorted(leaked)
    assert leaked[-1] > 0.9
    assert worst[-1] > 0.5 * n_cells
    # No sweep point is clean on both counts at once.
    assert all(
        leak > 0.02 or starved > 0.1 * n_cells
        for leak, starved in zip(leaked, unseeded)
    )


def test_intradomain_spread_is_broad_and_on_target(full_phantom):
    """Heterogeneous by construction: most cells quiet, a minority not."""

    spread = phantom_module.intradomain_angular_spread_deg(
        full_phantom.labels,
        full_phantom.segmentation_field,
        full_phantom.config.n_cells,
    )
    assert spread.size == full_phantom.config.n_cells
    median = float(np.median(spread))
    p95 = float(np.percentile(spread, 95))
    assert 0.025 <= median <= 0.035, median
    assert 0.08 <= p95 <= 0.12, p95
    # Broad, right-skewed: the top of the distribution is several times the
    # median and a real minority of cells sits well above it.
    assert 2.4 <= p95 / median <= 3.6, p95 / median
    # The log-normal tail is clipped, so no cell is an order of magnitude out.
    assert float(np.percentile(spread, 99)) < 6.0 * median
    assert float(np.max(spread)) < 12.0 * median
    assert 0.05 < float(np.mean(spread > 2.0 * median)) < 0.35
    assert float(np.percentile(spread, 5)) < 0.5 * median


def test_spread_and_range_are_different_statistics(full_phantom):
    """s_k and the range-like statistic must be reported separately.

    They differ by a factor of several, so quoting one against a target meant
    for the other misstates intradomain variation by most of an order of
    magnitude -- which is exactly the mistake this test exists to catch.
    """

    spread = phantom_module.intradomain_angular_spread_deg(
        full_phantom.labels,
        full_phantom.segmentation_field,
        full_phantom.config.n_cells,
    )
    angular_range = phantom_module.intradomain_angular_range_deg(
        full_phantom.labels,
        full_phantom.segmentation_field,
        full_phantom.config.n_cells,
    )
    assert angular_range.size == spread.size
    # A peak-to-peak span is always at least as large as the standard deviation
    # it is computed from, and here several times larger.
    assert np.all(angular_range >= spread - 1e-9)
    ratio = float(np.median(angular_range) / np.median(spread))
    assert 3.0 < ratio < 7.0, ratio


def test_angular_range_is_a_percentile_span():
    """Explicit check of the range definition on a known field."""

    labels = np.ones((1, 1, 5), dtype=np.int32)
    field = np.zeros((1, 1, 5, 2), dtype=np.float32)
    field[..., 0] = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    field[..., 1] = 0.0
    span = phantom_module.intradomain_angular_range_deg(labels, field, 1)
    assert span[0] == pytest.approx(4.0)

    field[..., 1] = np.array([0.0, 3.0, 3.0, 3.0, 3.0])
    span = phantom_module.intradomain_angular_range_deg(labels, field, 1)
    assert span[0] == pytest.approx(5.0)  # sqrt(4^2 + 3^2)


def test_per_cell_amplitude_is_log_normal_and_reproducible():
    """The multiplier is drawn once per cell from a reproducible log-normal."""

    config = replace(SMALL, intracell_amplitude_log_sigma=0.62)
    first = generate_phantom(config)
    again = generate_phantom(config)
    assert np.array_equal(first.segmentation_field, again.segmentation_field)

    quiet = generate_phantom(replace(config, intracell_amplitude_log_sigma=0.0))
    varied = generate_phantom(replace(config, intracell_amplitude_log_sigma=0.9))
    quiet_spread = phantom_module.intradomain_angular_spread_deg(
        quiet.labels, quiet.segmentation_field, config.n_cells
    )
    varied_spread = phantom_module.intradomain_angular_spread_deg(
        varied.labels, varied.segmentation_field, config.n_cells
    )
    # A wider amplitude distribution must widen the spread distribution without
    # simply translating it.
    quiet_ratio = np.percentile(quiet_spread, 95) / np.median(quiet_spread)
    varied_ratio = np.percentile(varied_spread, 95) / np.median(varied_spread)
    assert varied_ratio > quiet_ratio


def test_intradomain_variation_does_not_move_the_region_means(full_phantom):
    """Gradient and curvature must not perturb the misorientation distribution.

    The gradient is centroid-centred and the curvature is cell-mean-removed and
    tapered, so switching both off must leave the adjacent-region misorientation
    distribution essentially unchanged.  Without that decoupling the chi
    calibration cannot be reached at these amplitudes.
    """

    flat = generate_phantom(
        replace(
            full_phantom.config,
            intracell_gradient_deg_per_um=0.0,
            intracell_curvature_deg=0.0,
        )
    )
    with_variation = phantom_module.neighbour_misorientations_deg(full_phantom)
    without = phantom_module.neighbour_misorientations_deg(flat)
    assert np.median(with_variation) == pytest.approx(np.median(without), rel=0.05)
    assert with_variation.std() == pytest.approx(without.std(), rel=0.10)


def test_no_boundary_adjacent_ring_in_intradomain_field(full_phantom):
    """The intradomain field must not brighten with distance from a wall.

    A radial taper (used previously to keep a per-cell curvature amplitude
    continuous) prints a dark ring around every cell.  The curvature is now
    weak and untapered instead, so the residual must be flat against
    distance-to-boundary.
    """

    from scipy import stats
    from scipy.ndimage import distance_transform_edt

    field = full_phantom.segmentation_field.astype(np.float64)
    labels = full_phantom.labels
    residual = np.empty_like(field)
    for channel in range(field.shape[-1]):
        means = phantom_module._cell_means_of(
            field[..., channel], labels, full_phantom.config.n_cells
        )
        residual[..., channel] = field[..., channel] - means[labels]
    magnitude = np.linalg.norm(residual, axis=-1)
    distance = distance_transform_edt(
        ~phantom_module.ground_truth_boundaries(labels),
        sampling=full_phantom.spacing_um_zyx,
    )

    shells = [
        float(np.median(magnitude[(distance >= lo) & (distance < hi)]))
        for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5))
    ]
    assert max(shells) < 2.0 * min(shells), shells
    correlation = stats.spearmanr(
        magnitude.ravel()[::37], distance.ravel()[::37]
    ).statistic
    assert abs(correlation) < 0.25, correlation


def test_affine_drift_dominates_the_intradomain_variance(full_phantom):
    """80-90% of the intradomain variance must be the smooth affine drift."""

    config = full_phantom.config
    def pooled(cfg):
        made = generate_phantom(cfg)
        spread = phantom_module.intradomain_angular_spread_deg(
            made.labels, made.segmentation_field, cfg.n_cells
        )
        return float(np.mean(spread**2))

    total = pooled(config)
    affine = pooled(replace(config, intracell_curvature_deg=0.0))
    assert 0.78 <= affine / total <= 0.92, affine / total


def test_drift_is_slow():
    """The drift must vary on the scale of the volume, not of the voxel.

    Tested on the drift term itself: the phantom field also steps across cell
    walls, and those steps are supposed to be sharp.
    """

    rng = np.random.default_rng(0)
    shape = (20, 120, 120)
    spacing = np.array([1.0, 0.4, 0.4])
    extent = np.array(shape) * spacing
    axes = [(np.arange(n) + 0.5) * s for n, s in zip(shape, spacing)]
    points = np.stack(
        [g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=-1
    )
    drift = phantom_module._slow_drift(rng, points, extent).reshape(shape)

    for axis, step in enumerate(spacing):
        gradient = float(np.abs(np.diff(drift, axis=axis)).max()) / step
        # Unit variance over the volume: a slow field cannot change by more than
        # a few standard deviations across the whole extent, let alone one voxel.
        assert gradient * extent[axis] < 12.0, (axis, gradient)


def test_ground_truth_boundaries_mark_both_sides():
    labels = np.ones((2, 4, 4), dtype=np.int32)
    labels[..., 2:] = 2
    boundary = phantom_module.ground_truth_boundaries(labels)
    assert boundary[..., 1].all() and boundary[..., 2].all()
    assert not boundary[..., 0].any() and not boundary[..., 3].any()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_marker_confusion_detects_percolation_and_fragmentation():
    truth = np.zeros((2, 4, 12), dtype=np.int32)
    truth[..., :4] = 1
    truth[..., 4:8] = 2
    truth[..., 8:] = 3

    perfect = truth.copy()
    report = bench_metrics.marker_confusion(perfect, truth, min_overlap_voxels=4)
    assert report["percolating_markers"] == 0
    assert report["max_cells_per_marker"] == 1
    assert report["cells_split"] == 0
    assert report["cells_unseeded"] == 0
    assert report["percolating_marker_volume_fraction"] == pytest.approx(0.0)

    # One marker leaking across the first wall: percolation.
    leaked = np.where(truth == 3, 2, 1).astype(np.int32)
    report = bench_metrics.marker_confusion(leaked, truth, min_overlap_voxels=4)
    assert report["percolating_markers"] == 1
    assert report["max_cells_per_marker"] == 2
    assert report["percolating_marker_volume_fraction"] > 0.5

    # One cell claimed by two markers: fragmentation.
    split = truth.copy()
    split[..., :2] = 4
    report = bench_metrics.marker_confusion(split, truth, min_overlap_voxels=4)
    assert report["cells_split"] == 1
    assert report["percolating_markers"] == 0

    # A cell with no marker at all.
    starved = np.where(truth == 2, 0, truth).astype(np.int32)
    report = bench_metrics.marker_confusion(starved, truth, min_overlap_voxels=4)
    assert report["cells_unseeded"] == 1


def test_marker_confusion_ignores_empty_ground_truth_labels():
    """An unused label must not be reported as an unseeded cell."""

    truth = np.zeros((2, 2, 8), dtype=np.int32)
    truth[..., :4] = 1
    truth[..., 4:] = 7  # labels 2..6 are never used
    markers = truth.copy()
    report = bench_metrics.marker_confusion(markers, truth, min_overlap_voxels=2)
    assert report["cells_unseeded"] == 0
    assert report["max_cells_per_marker"] == 1


def test_metrics_are_exact_for_a_perfect_partition():
    labels = np.zeros((6, 12, 12), dtype=np.int32)
    labels[:, :6, :6] = 1
    labels[:, :6, 6:] = 2
    labels[:, 6:, :6] = 3
    labels[:, 6:, 6:] = 4

    result = bench_metrics.evaluate(labels, labels.copy(), (1.0, 0.4, 0.4))
    assert result["ari"] == pytest.approx(1.0)
    assert result["vi_total_bits"] == pytest.approx(0.0, abs=1e-9)
    assert result["boundary_assd_um"] == pytest.approx(0.0)
    assert result["cell_count_error"] == 0


def test_vi_split_and_merge_point_the_right_way():
    truth = np.zeros((4, 8, 8), dtype=np.int32)
    truth[:, :4] = 1
    truth[:, 4:] = 2

    merged = np.ones_like(truth)
    _, split, merge = bench_metrics.variation_of_information(truth, merged)
    assert merge > 0 and split == pytest.approx(0.0, abs=1e-9)

    oversegmented = truth.copy()
    oversegmented[:, :2] = 3
    _, split, merge = bench_metrics.variation_of_information(truth, oversegmented)
    assert split > 0 and merge == pytest.approx(0.0, abs=1e-9)


def test_boundary_assd_uses_the_anisotropic_spacing():
    truth = np.zeros((8, 16, 16), dtype=np.int32)
    truth[..., :8] = 1
    truth[..., 8:] = 2
    prediction = np.zeros_like(truth)
    prediction[..., :10] = 1
    prediction[..., 10:] = 2

    coarse = bench_metrics.boundary_assd_um(truth, prediction, (1.0, 0.4, 0.4))
    fine = bench_metrics.boundary_assd_um(truth, prediction, (1.0, 0.2, 0.2))
    # The same two-voxel offset costs half as much when columns are half as wide.
    assert coarse == pytest.approx(2.0 * fine, rel=1e-6)
    assert coarse > 0.0


def test_evaluate_rejects_unlabelled_voxels():
    truth = np.ones((2, 4, 4), dtype=np.int32)
    prediction = truth.copy()
    prediction[0, 0, 0] = 0
    with pytest.raises(ValueError):
        bench_metrics.evaluate(truth, prediction, (1.0, 0.4, 0.4))


def test_oracle_object_metrics_detect_known_split_and_merge():
    import oracle_metrics

    truth = np.ones((2, 10, 20), dtype=np.int32)
    truth[..., 10:] = 2
    split = truth.copy()
    split[..., :5] = 3
    result = oracle_metrics.evaluate_partition(truth, split, (1.0, .4, .4))
    assert result["true_cells_split"] == 1
    assert result["excess_fragments_total"] == 1
    assert result["pred_cells_merging"] == 0
    merged = np.ones_like(truth)
    result = oracle_metrics.evaluate_partition(truth, merged, (1.0, .4, .4))
    assert result["pred_cells_merging"] == 1
    assert result["true_cells_split"] == 0


def test_oracle_perfect_partition_has_perfect_object_and_boundary_scores():
    import oracle_metrics

    truth = np.ones((3, 8, 8), dtype=np.int32)
    truth[..., 4:] = 2
    result = oracle_metrics.evaluate_partition(truth, truth, (1.0, .4, .4))
    assert result["matched_iou_mean"] == pytest.approx(1)
    assert result["matched_dice_mean"] == pytest.approx(1)
    assert result["object_precision_at_0p75"] == pytest.approx(1)
    assert result["boundary_f1_at_0p4um"] == pytest.approx(1)
    assert result["interface_precision"] == pytest.approx(1)


def test_oracle_store_resumes_after_a_truncated_line(tmp_path):
    from oracle_store import Store

    path = tmp_path / "trials.jsonl"
    path.write_text('{"config_key":"broken"')
    store = Store(path)
    row = {"config_key": "valid", "random_seed": 7, "status": "ok"}
    assert store.append([row]) == 1
    resumed = Store(path)
    assert resumed.has("valid", 7)


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------


def test_isotropic_footprint_is_a_physical_sphere():
    footprint = pipelines.isotropic_footprint((1.0, 0.4, 0.4), 1.2)
    # 1.2 um reaches 1 voxel along z and 3 along y and x.  Regression guard:
    # 1.2 / 0.4 is 2.9999999999999996, so a plain floor would give (3, 5, 5).
    assert footprint.shape == (3, 7, 7)
    assert footprint[tuple(s // 2 for s in footprint.shape)]
    # Reaches further, in voxels, along the finely sampled axes.
    assert footprint[1, 3, :].sum() > footprint[:, 3, 3].sum()
    # Every included offset is within the physical radius.
    offsets = np.argwhere(footprint) - np.array([1, 3, 3])
    physical = offsets * np.array([1.0, 0.4, 0.4])
    assert np.all(np.linalg.norm(physical, axis=1) <= 1.2 + 1e-9)


def test_kam_is_zero_on_a_constant_field():
    field = np.full((6, 6, 6, 2), 0.3, dtype=np.float32)
    mask = np.ones(field.shape[:3], dtype=bool)
    kam = pipelines.masked_kam(field, mask, np.ones((3, 3, 3), dtype=bool))
    assert np.allclose(kam[mask], 0.0, atol=1e-6)


def _flood_fill_kwargs(**overrides):
    kwargs = dict(
        local_threshold_deg=0.05, global_threshold_deg=0.2, footprint_tolerance=0.5,
        min_cell_size=10, max_seed_attempts=1500, stagnation_tolerance=500,
        random_seed=0, watershed_connectivity=1,
    )
    kwargs.update(overrides)
    return kwargs


def test_both_arms_return_a_complete_partition(small_phantom):
    spacing = small_phantom.spacing_um_zyx
    kam = pipelines.masked_kam(
        small_phantom.field, small_phantom.mask,
        pipelines.isotropic_footprint(spacing, 1.0),
    )
    flood, markers = pipelines.run_flood_fill(
        small_phantom.field, small_phantom.mask, kam,
        pipelines.isotropic_footprint(spacing, 0.9), **_flood_fill_kwargs()
    )
    threshold_labels, _, threshold = pipelines.run_kam_threshold(
        kam, small_phantom.mask, percentile=25, min_cell_size=10,
        connectivity=1, watershed_connectivity=1,
    )
    assert markers.max() > 0
    assert threshold > 0
    for labels in (flood, threshold_labels):
        assert labels.shape == small_phantom.labels.shape
        assert np.all(labels > 0), "watershed must leave no unlabelled voxel"


def test_flood_fill_does_not_mutate_the_caller_mask(small_phantom):
    spacing = small_phantom.spacing_um_zyx
    kam = pipelines.masked_kam(
        small_phantom.field, small_phantom.mask,
        pipelines.isotropic_footprint(spacing, 1.0),
    )
    before = small_phantom.mask.copy()
    pipelines.run_flood_fill(
        small_phantom.field, small_phantom.mask, kam,
        pipelines.isotropic_footprint(spacing, 0.9), **_flood_fill_kwargs()
    )
    assert np.array_equal(small_phantom.mask, before)


def test_flood_fill_is_reproducible_for_one_seed_order(small_phantom):
    spacing = small_phantom.spacing_um_zyx
    kam = pipelines.masked_kam(
        small_phantom.field, small_phantom.mask,
        pipelines.isotropic_footprint(spacing, 1.0),
    )
    footprint = pipelines.isotropic_footprint(spacing, 0.9)
    first, _ = pipelines.run_flood_fill(
        small_phantom.field, small_phantom.mask, kam, footprint,
        **_flood_fill_kwargs(random_seed=7)
    )
    again, _ = pipelines.run_flood_fill(
        small_phantom.field, small_phantom.mask, kam, footprint,
        **_flood_fill_kwargs(random_seed=7)
    )
    assert np.array_equal(first, again)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def test_single_pixel_boundaries_are_one_pixel_wide():
    labels = np.ones((6, 6), dtype=np.int32)
    labels[:, 3:] = 2
    boundary = figures.single_pixel_boundaries(labels)
    assert boundary[:, 2].all()
    assert not boundary[:, 3].any()
    assert boundary.sum(axis=1).max() == 1


def test_overlay_is_opaque_black_with_no_alpha_channel():
    rgb = np.full((4, 4, 3), 0.7)
    boundary = np.zeros((4, 4), dtype=bool)
    boundary[1, 1] = True
    out = figures.overlay(rgb, boundary)
    assert np.array_equal(out[1, 1], np.zeros(3))
    assert np.array_equal(out[0], rgb[0])
    assert out.shape[-1] == 3


def test_display_factors_make_pixels_physically_square():
    # A (y, x) slice with 1.0 um rows and 0.4 um columns.
    factors = figures.isotropic_display_factors((1.0, 0.4), upsample=2)
    assert factors == (5, 2)
    assert 1.0 / factors[0] == pytest.approx(0.4 / factors[1])


def test_panels_differ_only_in_their_black_lines(small_phantom):
    reference = figures.feature_reference_deg(small_phantom.field[0])
    background = figures.feature_rgb(small_phantom.field[0], reference)
    other = (small_phantom.labels[0] // 2) + 1
    rasters = [
        figures.overlay(background, figures.single_pixel_boundaries(labels))
        for labels in (small_phantom.labels[0], other)
    ]
    black = [np.all(raster == 0.0, axis=-1) for raster in rasters]
    agree = ~(black[0] | black[1])
    assert np.array_equal(rasters[0][agree], rasters[1][agree])


def test_rendered_boundaries_are_exactly_one_output_pixel(tmp_path, small_phantom):
    """Read the saved PNG back and compare it to the raster that was drawn.

    If a panel of the PNG equals the composited raster pixel for pixel, then the
    one-pixel boundaries in that raster are one pixel in the output.  That is a
    stronger and less fiddly check than measuring run lengths, which cannot tell
    a line's width from its length.
    """

    spacing = small_phantom.spacing_um_zyx
    kam = pipelines.masked_kam(
        small_phantom.field, small_phantom.mask,
        pipelines.isotropic_footprint(spacing, 1.0),
    )
    kam_labels = (small_phantom.labels // 3) + 1
    info = figures.render_figure(
        tmp_path / "figure",
        field=small_phantom.field,
        kam=kam,
        truth_labels=small_phantom.labels,
        kam_labels=kam_labels,
        flood_fill_labels=small_phantom.labels,
        spacing_um_zyx=spacing,
        upsample=4,
        scale_bar_um=4.0,
        dpi=200,
    )
    assert (tmp_path / "figure.png").exists()
    assert (tmp_path / "figure.pdf").exists()
    # The (y, x) slice is already isotropic at 0.4 um, so both factors match.
    assert info["display_factors_yx"] == [4, 4]

    layer = info["layer"]
    factors = tuple(info["display_factors_yx"])
    reference = info["colour_reference_deg"]
    background = figures.upsample_nearest(
        figures.feature_rgb(small_phantom.field[layer], reference), factors
    )
    expected = {
        "Angular feature map": background,
        "Ground truth": figures.overlay(
            background,
            figures.single_pixel_boundaries(
                figures.upsample_nearest(
                    small_phantom.labels[layer][..., None], factors
                )[..., 0]
            ),
        ),
        "KAM threshold": figures.overlay(
            background,
            figures.single_pixel_boundaries(
                figures.upsample_nearest(kam_labels[layer][..., None], factors)[..., 0]
            ),
        ),
    }

    checked = 0
    for title, box in zip(info["panel_titles"], info["panel_boxes_ltwh"]):
        if title not in expected:
            continue
        panel = figures.read_panel(tmp_path / "figure.png", box)
        assert panel.shape == expected[title].shape, title
        # The scale bar and its label are drawn inside the axes, so only the
        # upper part of each panel is pure image.
        keep = int(panel.shape[0] * 0.25)
        drawn = panel[keep:]
        wanted = expected[title][keep:]
        # 8-bit PNG quantisation is the only permitted difference.
        assert np.max(np.abs(drawn - wanted)) <= 1.5 / 255.0, title
        # Black is exact, so the boundary pixels must match one for one.
        assert np.array_equal(
            np.all(drawn <= 1.0 / 255.0, axis=-1),
            np.all(wanted == 0.0, axis=-1),
        ), title
        checked += 1
    assert checked == 3


def test_figure_is_pixel_exact_in_size(tmp_path, small_phantom):
    import matplotlib.image as mpimg

    spacing = small_phantom.spacing_um_zyx
    kam = pipelines.masked_kam(
        small_phantom.field, small_phantom.mask,
        pipelines.isotropic_footprint(spacing, 1.0),
    )
    info = figures.render_figure(
        tmp_path / "figure",
        field=small_phantom.field,
        kam=kam,
        truth_labels=small_phantom.labels,
        kam_labels=small_phantom.labels,
        flood_fill_labels=small_phantom.labels,
        spacing_um_zyx=spacing,
        upsample=3,
        dpi=150,
    )
    image = mpimg.imread(str(tmp_path / "figure.png"))
    assert [image.shape[1], image.shape[0]] == info["figure_pixels_wh"]


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def test_cli_defaults_come_from_the_phantom_module():
    """A stale CLI default must not silently replace the phantom's own."""

    args = benchmark.build_parser().parse_args(["--out-dir", "/tmp/unused"])
    assert benchmark.phantom_config_from_args(args) == PhantomConfig()

    args = benchmark.build_parser().parse_args(
        ["--out-dir", "/tmp/unused", "--n-cells", "42"]
    )
    config = benchmark.phantom_config_from_args(args)
    assert config.n_cells == 42
    # Everything not passed still tracks PhantomConfig.
    assert config.shape_zyx == PhantomConfig().shape_zyx
    assert config.spacing_um_zyx == PhantomConfig().spacing_um_zyx


def test_grid_covers_every_combination():
    config = benchmark.SearchConfig(
        local_thresholds_deg=(0.01, 0.02),
        footprint_tolerances=(0.1,),
        footprint_radii_um=(0.9,),
        min_cell_sizes=(20, 60),
        kam_radii_um=(0.6, 1.0),
    )
    points = benchmark.grid_points(config)
    assert len(points) == 2 * 1 * 1 * 2 * 2
    assert all(set(point) == set(benchmark.PARAMETER_NAMES) for point in points)
    assert len({tuple(sorted(point.items())) for point in points}) == len(points)


def test_refinement_extends_an_axis_off_its_boundary(monkeypatch, small_phantom):
    """A best value on an end must pull the axis outward, not be accepted."""

    config = benchmark.SearchConfig(
        local_thresholds_deg=(0.01, 0.02, 0.04),
        footprint_tolerances=(0.1,),
        footprint_radii_um=(1.3,),
        min_cell_sizes=(20,),
        kam_radii_um=(1.0,),
        max_refinement_rounds=12,
    )

    # The optimum sits at 0.0025, below the starting range, so the loop has to
    # extend the axis downwards and then settle on an interior value.
    state = {}

    def tracking_run(phantom, kam, point, cfg, random_seed):
        state["local"] = point["local_threshold_deg"]
        return phantom.labels, phantom.labels

    def scored(phantom, labels):
        penalty = np.log(state["local"] / 0.0025) ** 2
        return {"ari": float(1.0 / (1.0 + penalty)), "vi_total_bits": 1.0,
                "vi_split_bits": 0.5, "vi_merge_bits": 0.5,
                "boundary_assd_um": 0.5, "n_cells_truth": 10,
                "n_cells_pred": 10, "cell_count_error": 0,
                "percolating_marker_volume_fraction": 0.0, "cells_split": 0,
                "cells_unseeded": 0}

    monkeypatch.setattr(benchmark, "run_flood_fill_point", tracking_run)
    monkeypatch.setattr(benchmark, "_score", scored)

    rows, best, summary = benchmark.refine_flood_fill(
        small_phantom, np.zeros(small_phantom.labels.shape, dtype=np.float32),
        config, verbose=False,
    )
    assert best["local_threshold_deg"] < 0.01, "axis was never extended"
    assert summary["axes"]["local_threshold_deg"][0] < 0.01
    # Extended far enough that the optimum is no longer on an end.
    assert not summary["on_grid_boundary"]["local_threshold_deg"]
    assert best["local_threshold_deg"] == pytest.approx(0.0025, rel=0.6)
    assert summary["unsearched_axes"], "single-value axes reported as unsearched"


def test_refinement_terminates_with_an_interior_optimum(monkeypatch, small_phantom):
    """A clean interior peak must converge and leave no boundary flag set."""

    config = benchmark.SearchConfig(
        local_thresholds_deg=(0.005, 0.01, 0.02, 0.04),
        footprint_tolerances=(0.1,),
        footprint_radii_um=(0.9, 1.3, 1.8),
        min_cell_sizes=(50, 100, 200),
        kam_radii_um=(0.6, 1.0, 1.6),
        max_refinement_rounds=8,
        refine_ratio=1.5,
        refine_min_size_step=40,
    )
    state = {}

    def tracking_run(phantom, kam, point, cfg, random_seed):
        state["point"] = point
        return phantom.labels, phantom.labels

    def scored(phantom, labels):
        p = state["point"]
        # Smooth quadratic peak in log-space, interior on every axis.
        penalty = (
            (np.log(p["local_threshold_deg"] / 0.01)) ** 2
            + (np.log(p["footprint_tolerance"] / 0.1)) ** 2
            + (np.log(p["footprint_radius_um"] / 1.3)) ** 2
            + (np.log(p["min_cell_size"] / 100.0)) ** 2
            + (np.log(p["kam_radius_um"] / 1.0)) ** 2
        )
        return {"ari": float(1.0 / (1.0 + penalty)), "vi_total_bits": 1.0,
                "vi_split_bits": 0.5, "vi_merge_bits": 0.5,
                "boundary_assd_um": 0.5, "n_cells_truth": 10,
                "n_cells_pred": 10, "cell_count_error": 0,
                "percolating_marker_volume_fraction": 0.0, "cells_split": 0,
                "cells_unseeded": 0}

    monkeypatch.setattr(benchmark, "run_flood_fill_point", tracking_run)
    monkeypatch.setattr(benchmark, "_score", scored)

    rows, best, summary = benchmark.refine_flood_fill(
        small_phantom, np.zeros(small_phantom.labels.shape, dtype=np.float32),
        config, verbose=False,
    )
    assert summary["converged"], "refinement hit the round limit"
    assert not summary["any_on_boundary"], summary["on_grid_boundary"]
    assert best["footprint_radius_um"] == pytest.approx(1.3, rel=0.3)


def test_repeat_candidates_breaks_ties_on_vi_then_assd_then_spread(
    monkeypatch, small_phantom
):
    """Within the ARI tolerance the tie-breakers must decide, in order."""

    config = benchmark.SearchConfig(
        repeat_seeds=(1, 2, 3, 4, 5), selection_tolerance=0.01
    )
    points = [
        {"local_threshold_deg": 0.01, "footprint_tolerance": 0.1,
         "footprint_radius_um": 1.3, "min_cell_size": 100, "kam_radius_um": 1.0},
        {"local_threshold_deg": 0.02, "footprint_tolerance": 0.1,
         "footprint_radius_um": 1.3, "min_cell_size": 100, "kam_radius_um": 1.0},
    ]
    # The first point wins on raw ARI by less than the tolerance but has worse
    # VI, so the second must be selected.
    table = {0.01: (0.900, 2.0), 0.02: (0.895, 1.0)}
    state = {}

    def tracking_run(phantom, kam, point, cfg, random_seed):
        state["local"] = point["local_threshold_deg"]
        return phantom.labels, phantom.labels

    def scored(phantom, labels):
        ari, vi = table[state["local"]]
        return {"ari": ari, "vi_total_bits": vi, "vi_split_bits": 0.5,
                "vi_merge_bits": 0.5, "boundary_assd_um": 0.5,
                "n_cells_truth": 10, "n_cells_pred": 10, "cell_count_error": 0,
                "percolating_marker_volume_fraction": 0.0, "cells_split": 0,
                "cells_unseeded": 0}

    monkeypatch.setattr(benchmark, "run_flood_fill_point", tracking_run)
    monkeypatch.setattr(benchmark, "_score", scored)

    rows = []
    selected = benchmark.repeat_candidates(
        small_phantom, np.zeros(small_phantom.labels.shape, dtype=np.float32),
        points, config, rows, verbose=False,
    )
    assert selected["local_threshold_deg"] == pytest.approx(0.02)
    assert selected["n_tied_within_tolerance"] == 2
    assert selected["n_repeats"] == 5
    assert len({row["random_seed"] for row in rows}) == 5


def test_search_records_failures_without_stopping(monkeypatch, small_phantom):
    config = benchmark.SearchConfig(
        local_thresholds_deg=(0.01, 0.02, 0.04),
        footprint_tolerances=(0.5,), footprint_radii_um=(0.9,),
        min_cell_sizes=(20,), kam_radii_um=(1.0,), max_refinement_rounds=2,
    )

    def fake_run(phantom, kam, point, cfg, random_seed):
        if point["local_threshold_deg"] < 0.015:
            raise RuntimeError("Flood fill produced no accepted markers.")
        return phantom.labels, phantom.labels

    monkeypatch.setattr(benchmark, "run_flood_fill_point", fake_run)
    rows, best, summary = benchmark.refine_flood_fill(
        small_phantom,
        np.zeros(small_phantom.labels.shape, dtype=np.float32),
        config,
        verbose=False,
    )
    failures = [row for row in rows if row["status"] == "error"]
    assert failures and "no accepted markers" in failures[0]["error"]
    assert best["local_threshold_deg"] >= 0.015


def test_acceptance_gate_flags_a_pathological_phantom(tmp_path, monkeypatch):
    """The benchmark must say so when flood fill cannot even seed the cells."""

    config = benchmark.SearchConfig(
        local_thresholds_deg=(0.04,), footprint_tolerances=(0.5,),
        footprint_radii_um=(0.9,), min_cell_sizes=(10,), kam_radii_um=(1.0,),
        kam_percentiles=(30.0,),
        failure_mode_percentiles=(30.0,), n_finalists=1, repeat_seeds=(1,),
        max_refinement_rounds=1, max_seed_attempts=800,
        stagnation_tolerance=200, max_unseeded_fraction=0.10,
    )
    metrics = benchmark.run_benchmark(
        SMALL, config, tmp_path, figure_upsample=2, figure_dpi=120, verbose=False
    )
    acceptance = metrics["acceptance"]
    assert acceptance["max_unseeded_fraction"] == 0.10
    assert 0.0 <= acceptance["unseeded_fraction"] <= 1.0
    assert acceptance["phantom_accepted"] == (
        acceptance["unseeded_fraction"] <= 0.10
    )


def test_end_to_end_writes_every_documented_output(tmp_path):
    config = benchmark.SearchConfig(
        local_thresholds_deg=(0.04, 0.08),
        footprint_tolerances=(0.5,),
        footprint_radii_um=(0.9,),
        min_cell_sizes=(10,),
        kam_radii_um=(1.0,),
        kam_percentiles=(20.0, 30.0),
        failure_mode_percentiles=(20.0, 50.0),
        n_finalists=2,
        repeat_seeds=(1, 2),
        max_refinement_rounds=1,
        max_seed_attempts=1500,
        stagnation_tolerance=400,
    )
    metrics = benchmark.run_benchmark(
        SMALL, config, tmp_path, figure_upsample=2, figure_dpi=120, verbose=False
    )

    for name in (
        "search.csv",
        "selected_parameters.json",
        "metrics.json",
        "benchmark_figure.png",
        "benchmark_figure.pdf",
        "intradomain_spread_cdf.png",
        "kam_failure_modes.csv",
        "kam_failure_modes.png",
        "phantom_and_labels.npz",
    ):
        assert (tmp_path / name).exists(), name

    assert set(metrics["flood_fill"]["parameters"]) == set(benchmark.PARAMETER_NAMES)
    assert metrics["flood_fill"]["n_repeats"] == 2
    assert "std" in metrics["flood_fill"]["repeated"]["ari"]
    assert metrics["phantom"]["n_cells"] > 0
    assert metrics["kam_threshold"]["ari"] <= 1.0

    import csv

    with (tmp_path / "search.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["stage"] for row in rows} == {"search", "repeat", "kam_baseline"}
    assert "refinement" in metrics
    assert "intradomain_spread_deg" in metrics["phantom"]
