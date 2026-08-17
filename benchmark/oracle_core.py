#!/usr/bin/env python3
"""Cached phantom, cached KAM fields, and one fully-scored flood-fill run.

This is the evaluation kernel the oracle search calls tens of thousands of
times, so two things are cached rather than recomputed:

* the **phantom**, which takes ~14 s to build and is fixed for the whole study;
* the **KAM field**, one per distinct watershed footprint.

Two of the searched parameters are not continuous, and treating them as if they
were wastes most of the search budget:

**Physical radii.**  ``isotropic_footprint`` rasterises a physical sphere onto
the anisotropic voxel grid, so the footprint is a step function of the radius.
Over 0.3--4.0 um there are only 122 distinct footprints.  Both radius axes are
therefore *canonicalised*: a radius is replaced by the representative of its
equivalence class, and the class interval is recorded, so "the optimum is at
1.30 um" can be reported honestly as "the optimum is the 63-voxel footprint,
which any radius in [1.282, 1.340] um produces".

**Footprint tolerance.**  The C++ kernel accepts a voxel when at least
``ceil(footprint_tolerance * valid_neighbours)`` of its in-mask neighbours pass
the misorientation test.  The tolerance only enters through that ceiling, so it
is a staircase too -- but not a single one: ``valid_neighbours`` falls at the
volume border and, more importantly, wherever neighbouring voxels have already
been claimed by an earlier region.  A tolerance therefore defines a *vector* of
integer requirements, one per possible neighbour count, and two tolerances are
equivalent only if the whole vector matches.  That vector is what is
canonicalised and recorded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = BENCHMARK_DIR / "oracle_results" / "cache"

#: Radius range over which the footprint equivalence classes are enumerated.
RADIUS_RANGE_UM = (0.30, 4.00)
RADIUS_PROBE_STEP_UM = 0.001

PARAMETER_NAMES = (
    "local_threshold_deg",
    "global_threshold_deg",
    "footprint_tolerance",
    "footprint_radius_um",
    "min_cell_size",
    "kam_radius_um",
)


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class Config:
    """One flood-fill operating point.

    ``global_threshold_deg <= 0`` disables the running-mean test, which is a
    genuinely different regime rather than a large value: the C++ kernel skips
    the test entirely instead of applying a loose one.
    """

    local_threshold_deg: float
    global_threshold_deg: float
    footprint_tolerance: float
    footprint_radius_um: float
    min_cell_size: int
    kam_radius_um: float

    def key(self) -> str:
        return "|".join(
            (
                f"{self.local_threshold_deg:.10g}",
                f"{self.global_threshold_deg:.10g}",
                f"{self.footprint_tolerance:.10g}",
                f"{self.footprint_radius_um:.10g}",
                f"{int(self.min_cell_size):d}",
                f"{self.kam_radius_um:.10g}",
            )
        )

    def hash(self) -> str:
        """Stable SHA-256 identity of the canonical parameter tuple."""

        return hashlib.sha256(self.key().encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, float]:
        out = asdict(self)
        out["min_cell_size"] = int(out["min_cell_size"])
        return out


#: Fixed run controls, not searched.  ``max_seed_attempts`` is generous enough
#: that no configuration in the study is limited by it; the search records the
#: marker count so a truncated run would be visible.
MAX_SEED_ATTEMPTS = 8000
STAGNATION_TOLERANCE = 2000
WATERSHED_CONNECTIVITY = 1

GLOBAL_DISABLED = -1.0


# ------------------------------------------------------------ radius classes


def _footprint_signature(footprint: np.ndarray) -> str:
    return f"{footprint.shape}:{hashlib.md5(footprint.tobytes()).hexdigest()}"


def enumerate_radius_classes(
    spacing_um_zyx: Sequence[float],
    low_um: float = RADIUS_RANGE_UM[0],
    high_um: float = RADIUS_RANGE_UM[1],
    step_um: float = RADIUS_PROBE_STEP_UM,
) -> list[dict[str, Any]]:
    """Equivalence classes of physical radius that give the same footprint.

    Returned in increasing radius, each with its interval, its representative
    (the geometric centre of the interval, so it is not an interval endpoint
    where floating-point rounding could tip it into the neighbouring class) and
    the number of voxels in the footprint.
    """

    import pipelines

    classes: list[dict[str, Any]] = []
    previous = None
    for radius in np.arange(float(low_um), float(high_um) + 0.5 * step_um, step_um):
        footprint = pipelines.isotropic_footprint(spacing_um_zyx, float(radius))
        signature = _footprint_signature(footprint)
        if signature != previous:
            classes.append(
                {
                    "low_um": float(radius),
                    "high_um": float(radius),
                    "n_voxels": int(footprint.sum()),
                    "shape": tuple(int(v) for v in footprint.shape),
                }
            )
            previous = signature
        else:
            classes[-1]["high_um"] = float(radius)
    for index, entry in enumerate(classes):
        entry["index"] = index
        entry["radius_um"] = float(
            math.sqrt(entry["low_um"] * entry["high_um"])
        )
    return classes


class RadiusClasses:
    """Snap a physical radius onto its footprint equivalence class.

    Snapping is by the footprint the radius actually rasterises to, not by the
    interval endpoints found on the probe grid.  Interval lookup gets ~1.4% of
    radii wrong, because a class boundary is only located to within one probe
    step and a radius landing inside that step is assigned to its neighbour --
    which would silently evaluate a different footprint from the one recorded.
    """

    def __init__(self, spacing_um_zyx: Sequence[float], cache_dir: Path | None = None):
        self.spacing = tuple(float(v) for v in spacing_um_zyx)
        path = None
        if cache_dir is not None:
            path = Path(cache_dir) / "radius_classes.json"
        if path is not None and path.exists():
            self.classes = json.loads(path.read_text())
        else:
            self.classes = enumerate_radius_classes(self.spacing)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self.classes, indent=2))
        self._by_signature: dict[str, int] = {}
        for entry in self.classes:
            self._by_signature[self._signature(entry["radius_um"])] = entry["index"]

    def _signature(self, radius_um: float) -> str:
        import pipelines

        return _footprint_signature(
            pipelines.isotropic_footprint(self.spacing, float(radius_um))
        )

    def index_of(self, radius_um: float) -> int:
        """Index of the class whose footprint ``radius_um`` rasterises to.

        A radius whose footprint was not seen on the probe grid -- possible for
        a class thinner than one probe step -- registers a new class rather than
        being forced into a neighbouring one.
        """

        signature = self._signature(radius_um)
        index = self._by_signature.get(signature)
        if index is not None:
            return index
        import pipelines

        footprint = pipelines.isotropic_footprint(self.spacing, float(radius_um))
        entry = {
            "low_um": float(radius_um),
            "high_um": float(radius_um),
            "radius_um": float(radius_um),
            "n_voxels": int(footprint.sum()),
            "shape": tuple(int(v) for v in footprint.shape),
            "index": len(self.classes),
            "registered_at_runtime": True,
        }
        self.classes.append(entry)
        self._by_signature[signature] = entry["index"]
        return entry["index"]

    def snap(self, radius_um: float) -> float:
        """Representative radius of the class containing ``radius_um``."""

        return float(self.classes[self.index_of(radius_um)]["radius_um"])

    def representatives(self, low_um: float, high_um: float) -> list[float]:
        return [
            float(c["radius_um"])
            for c in self.classes
            if low_um <= c["radius_um"] <= high_um
        ]

    def describe(self, radius_um: float) -> dict[str, Any]:
        return dict(self.classes[self.index_of(radius_um)])


# ------------------------------------------------- footprint-tolerance ladder


def neighbour_requirements(
    footprint_tolerance: float, n_neighbours: int
) -> np.ndarray:
    """``ceil(tolerance * n)`` for every ``n`` from 1 to ``n_neighbours``.

    This vector is what the algorithm actually sees.  The full-footprint entry
    is the headline number, but the smaller counts are reached constantly:
    ``valid_neighbours`` is the count of neighbours still *in the mask*, and the
    C++ driver clears the mask over every region it accepts, so a voxel growing
    alongside an established cell is tested against a reduced count.
    """

    n = np.arange(1, int(n_neighbours) + 1, dtype=np.float64)
    return np.ceil(float(footprint_tolerance) * n - 1e-9).astype(np.int64)


def tolerance_bands(n_neighbours: int) -> list[dict[str, Any]]:
    """Tolerance intervals giving the same requirement at the *full* footprint.

    ``(r - 1) / n < tolerance <= r / n`` gives requirement ``r``, so there are
    exactly ``n_neighbours`` bands.  This is the headline staircase and the one
    the analysis groups by.

    It is **not** an exact equivalence.  Two tolerances in the same band still
    differ at reduced neighbour counts -- at the volume border, and wherever
    neighbouring voxels have already been claimed -- so they are not guaranteed
    to produce the same segmentation.  The tolerance axis is therefore searched
    continuously and only *reported* by band;
    :func:`oracle_probes.tolerance_band_equivalence` measures how much of the
    within-band variation is real.
    """

    n = int(n_neighbours)
    return [
        {
            "requirement_full_footprint": r,
            "low": float((r - 1) / n),
            "high": float(r / n),
            "tolerance": float((r - 0.5) / n),
            "n_neighbours": n,
        }
        for r in range(1, n + 1)
    ]


# -------------------------------------------------------------------- caching


@dataclass
class Workspace:
    """Everything an evaluation needs, loaded once per process."""

    labels: np.ndarray
    field: np.ndarray
    mask: np.ndarray
    spacing_um_zyx: tuple[float, float, float]
    cache_dir: Path
    radius_classes: RadiusClasses
    _kam: dict[float, np.ndarray]

    def kam(self, radius_um: float) -> np.ndarray:
        """KAM field for a watershed radius, from memory, disk, or computed."""

        import pipelines

        radius = self.radius_classes.snap(radius_um)
        if radius in self._kam:
            return self._kam[radius]
        index = self.radius_classes.index_of(radius)
        path = self.cache_dir / f"kam_class_{index:03d}.npy"
        if path.exists():
            field = np.load(path, mmap_mode="r")
        else:
            field = pipelines.masked_kam(
                self.field,
                self.mask,
                pipelines.isotropic_footprint(self.spacing_um_zyx, radius),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, field)
        # Keep no more than two fields.  Ordinary search workers need one, and
        # an analysis comparison may briefly need two; an unbounded cache was
        # the principal source of the crashed run's memory growth.
        self._kam[radius] = field
        while len(self._kam) > 2:
            oldest = next(iter(self._kam))
            if oldest == radius and len(self._kam) > 1:
                oldest = next(k for k in self._kam if k != radius)
            del self._kam[oldest]
        return field

    def clear_kam_cache(self) -> None:
        self._kam.clear()

    def footprint(self, radius_um: float) -> np.ndarray:
        import pipelines

        return pipelines.isotropic_footprint(
            self.spacing_um_zyx, self.radius_classes.snap(radius_um)
        )


def load_workspace(cache_dir: Path | str = DEFAULT_CACHE) -> Workspace:
    """Load (building and caching if needed) the fixed phantom.

    The phantom and its latent angular field are frozen for this study; nothing
    here may regenerate them with a different configuration.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "phantom.npz"
    if not path.exists():
        from phantom import PhantomConfig, generate_phantom

        phantom = generate_phantom(PhantomConfig())
        np.savez_compressed(
            path,
            labels=phantom.labels,
            latent=phantom.latent_field,
            field=phantom.field,
            wall_width_um=phantom.wall_width_um,
            spacing=np.asarray(phantom.spacing_um_zyx),
        )
    data = np.load(path)
    labels = np.ascontiguousarray(data["labels"])
    field = np.ascontiguousarray(data["latent"])
    spacing = tuple(float(v) for v in data["spacing"])
    return Workspace(
        labels=labels,
        field=field,
        mask=np.ones(labels.shape, dtype=bool),
        spacing_um_zyx=spacing,
        cache_dir=cache_dir,
        radius_classes=RadiusClasses(spacing, cache_dir),
        _kam={},
    )


# ----------------------------------------------------------------- evaluation


def canonical(config: Config, workspace: Workspace) -> Config:
    """Snap both radii onto their footprint classes; clamp the rest."""

    return Config(
        local_threshold_deg=float(config.local_threshold_deg),
        global_threshold_deg=(
            GLOBAL_DISABLED
            if float(config.global_threshold_deg) <= 0
            else float(config.global_threshold_deg)
        ),
        footprint_tolerance=float(config.footprint_tolerance),
        footprint_radius_um=workspace.radius_classes.snap(config.footprint_radius_um),
        min_cell_size=int(config.min_cell_size),
        kam_radius_um=workspace.radius_classes.snap(config.kam_radius_um),
    )


def segment(
    config: Config, workspace: Workspace, random_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Flood-fill markers and the watershed-refined partition they produce."""

    import pipelines

    footprint = workspace.footprint(config.footprint_radius_um)
    return pipelines.run_flood_fill(
        workspace.field,
        workspace.mask,
        workspace.kam(config.kam_radius_um),
        footprint,
        local_threshold_deg=float(config.local_threshold_deg),
        global_threshold_deg=float(config.global_threshold_deg),
        footprint_tolerance=float(config.footprint_tolerance),
        min_cell_size=int(config.min_cell_size),
        max_seed_attempts=MAX_SEED_ATTEMPTS,
        stagnation_tolerance=STAGNATION_TOLERANCE,
        random_seed=int(random_seed),
        watershed_connectivity=WATERSHED_CONNECTIVITY,
    )


def evaluate(
    config: Config,
    workspace: Workspace,
    random_seed: int,
    *,
    with_boundary: bool = True,
) -> dict[str, Any]:
    """Score one operating point at all three levels, before and after watershed.

    The ``marker_`` prefixed entries describe the identification step alone and
    the unprefixed ones the refined partition, which is what separates an
    identification error from a watershed error: the watershed can move a
    boundary and reassign a voxel, but it cannot merge two markers or invent a
    new one, so any difference in *count* is already present in the markers.
    """

    import oracle_metrics as om

    started = _now()
    labels, markers = segment(config, workspace, random_seed)
    segment_seconds = _now() - started

    row: dict[str, Any] = {
        **config.as_dict(),
        "random_seed": int(random_seed),
        "global_threshold_active": bool(config.global_threshold_deg > 0),
    }
    footprint = workspace.footprint(config.footprint_radius_um)
    n_neighbours = int(footprint.sum())
    requirements = neighbour_requirements(config.footprint_tolerance, n_neighbours)
    row["footprint_n_voxels"] = n_neighbours
    row["neighbour_requirement_full"] = int(requirements[-1])
    row["neighbour_requirement_fraction"] = float(
        requirements[-1] / max(n_neighbours, 1)
    )
    row["kam_footprint_n_voxels"] = int(
        workspace.footprint(config.kam_radius_um).sum()
    )

    row.update({f"marker_{k}": v for k, v in om.marker_errors(markers, workspace.labels).items()})
    if with_boundary:
        rows_, cols_, counts_, _, pred_sizes_ = om.contingency(
            workspace.labels, markers
        )
        dominant = om.dominant_map(
            rows_, cols_, counts_, int(pred_sizes_.size), by="pred"
        )
        row.update(
            {
                f"marker_{k}": v
                for k, v in om.interface_precision(
                    workspace.labels, markers, workspace.spacing_um_zyx, dominant
                ).items()
            }
        )
    row.update(
        om.evaluate_partition(
            workspace.labels,
            labels,
            workspace.spacing_um_zyx,
            with_boundary=with_boundary,
        )
    )
    row["cells_added_by_watershed"] = int(
        row["n_cells_pred"] - row["marker_marker_count"]
    )
    row["segment_seconds"] = segment_seconds
    row["eval_seconds"] = _now() - started
    row["status"] = "ok"
    return row


def _now() -> float:
    import time

    return time.perf_counter()


def limit_threads() -> None:
    """Pin the numeric libraries to one thread each.

    The search parallelises over configurations, so a worker that also spreads
    its own BLAS over 16 cores would oversubscribe the machine by that factor.
    """

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
