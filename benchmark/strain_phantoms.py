#!/usr/bin/env python3
"""Phantoms across the strain series, from the measured DFXM trend.

Source of the trend
-------------------
Zelenika, Cretton, Frankus, Borgi, Grumsen, Yildirim, Detlefs, Winther and
Poulsen, "Observing formation and evolution of dislocation cells during plastic
deformation", *Scientific Reports* **15**, 8655 (2025),
doi:10.1038/s41598-025-88262-3.  The paper reports, for a 99.9999 % pure
aluminium single crystal loaded along [111]:

* cell size is **log-normal** at every strain, with the mean falling **linearly**
  with strain and the standard deviation scaling with the mean, so the shape of
  the distribution is preserved;
* misorientation between adjacent cells follows a **chi distribution** whose
  scale ``sigma`` grows **linearly** with strain while the shape ``k`` stays
  approximately constant.

The tabulated values below are the ones already used by
``overnight_continue.py`` and are treated here as ground truth, per the study
owner.  **The 6.2 % entry is an extrapolation**: the paper's measurements span
0.6 % to 4.6 % strain, so 6.2 % lies beyond the measured range and is included
because it matches the strain of the experimental volume held separately.

What varies and what does not
-----------------------------
Across the series only the microstructure changes -- cell size (hence cell
count), size dispersion, and the misorientation distribution.  Wall geometry,
intracell structure and noise are held at the primary phantom's values so that
any change in the best segmentation parameters is attributable to the evolving
microstructure rather than to a changed rendering model.

Usage::

    python strain_phantoms.py build            # generate and cache all four
    python strain_phantoms.py summary          # report what was realised
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_OUT = HERE / "phantoms"

#: strain %  ->  (mean cell diameter um, sd um, chi shape k, chi scale sigma deg)
#: From the Sci Rep 15, 8655 (2025) trend; 6.2 % is extrapolated beyond the
#: paper's measured 0.6-4.6 % range.
STRAIN_TREND = {
    "2p4": (2.4, 5.02, 2.91, 1.60, 0.14),
    "3p5": (3.5, 4.87, 2.60, 1.59, 0.22),
    "4p6": (4.6, 4.59, 2.29, 1.72, 0.28),
    "6p2": (6.2, 4.20, 2.20, 1.70, 0.36),
}
EXTRAPOLATED = {"6p2"}
MEASURED_STRAIN_RANGE_PERCENT = (0.6, 4.6)

#: Matched to the primary phantom so results are directly comparable.
SHAPE_ZYX = (24, 160, 160)
SPACING_UM_ZYX = (1.0, 0.4, 0.4)
BASE_SEED = 2026081900


def config_for(strain_key: str, realization: int = 0):
    """The :class:`PhantomConfig` for one strain level."""

    from phantom import PhantomConfig

    strain, mean_d, sd_d, chi_k, chi_scale = STRAIN_TREND[strain_key]
    volume_um3 = float(np.prod(np.asarray(SHAPE_ZYX) * np.asarray(SPACING_UM_ZYX)))
    sphere_volume = math.pi * mean_d ** 3 / 6.0
    n_cells = max(24, int(round(volume_um3 / sphere_volume)))

    # Diameter-to-volume conversion is an explicit equivalent-sphere
    # approximation, not a claim that the measured 2-D diameter distribution is
    # a 3-D volume distribution.  sigma_log(V) = 3 sigma_log(d).
    sigma_log_d = math.sqrt(math.log1p((sd_d / mean_d) ** 2))
    log_volume_sigma = min(3.0 * sigma_log_d, 1.15)

    seed = BASE_SEED + 100 * list(STRAIN_TREND).index(strain_key) + realization
    config = PhantomConfig(
        shape_zyx=SHAPE_ZYX,
        spacing_um_zyx=SPACING_UM_ZYX,
        n_cells=n_cells,
        log_volume_sigma=log_volume_sigma,
        misorientation_k=chi_k,
        misorientation_sigma_deg=chi_scale,
        seed=seed,
    )
    meta = {
        "strain_key": strain_key,
        "strain_percent": strain,
        "realization": realization,
        "target_mean_cell_diameter_um": mean_d,
        "target_cell_diameter_sd_um": sd_d,
        "misorientation_k": chi_k,
        "misorientation_sigma_deg": chi_scale,
        "n_cells_requested": n_cells,
        "log_volume_sigma": log_volume_sigma,
        "volume_um3": volume_um3,
        "extrapolated_beyond_measured_range": strain_key in EXTRAPOLATED,
        "measured_strain_range_percent": list(MEASURED_STRAIN_RANGE_PERCENT),
        "trend_source": "Zelenika et al., Sci Rep 15, 8655 (2025), "
                        "doi:10.1038/s41598-025-88262-3",
        "diameter_to_volume_assumption": "equivalent-sphere approximation",
        "config": asdict(config),
    }
    return config, meta


#: The study's primary phantom.  Cell size and misorientation are both taken
#: from the 6.2 % strain point, so the volume is internally consistent: 4.20 um
#: cells with chi sigma 0.36 deg, as measured together at that strain.  Setting
#: one from the measurements and leaving the other at some other value produces
#: a microstructure the material never shows, and tunes every method on it.
#:
#: It carries its own seed so that it is an independent realisation, not a copy
#: of ``strain_6p2_r0``.  Keeping them distinct matters: the primary phantom is
#: where parameters are chosen, and the strain series is where the resulting
#: trend is tested, so they must not be the same volume.
PRIMARY_STRAIN_KEY = "6p2"
PRIMARY_SEED = 2026081999
PRIMARY_NAME = "primary_6p2_consistent"


def primary_path(out: Path = DEFAULT_OUT, realization: int = 0) -> Path:
    suffix = "" if realization == 0 else f"_r{realization}"
    return out / f"{PRIMARY_NAME}{suffix}.npz"


def build_primary(out: Path = DEFAULT_OUT, force: bool = False,
                  realization: int = 0) -> Path:
    """Generate the self-consistent primary phantom.

    ``realization`` offsets the seed to produce an independent volume of the
    same microstructure.  Realisation 0 is the primary phantom itself, name
    and seed unchanged, so existing results stay reproducible; the others
    exist to put an uncertainty on the headline table, which otherwise rests
    on a single volume.
    """

    from phantom import PhantomConfig, generate_phantom, neighbour_misorientations_deg

    path = primary_path(out, realization)
    if path.exists() and not force:
        print(f"  {path.name} already present")
        return path

    config, meta = config_for(PRIMARY_STRAIN_KEY)
    config = PhantomConfig(**{**asdict(config),
                              "seed": PRIMARY_SEED + realization})
    meta["realization"] = realization
    meta["config"] = asdict(config)
    meta["role"] = "primary phantom for the capped benchmark"
    meta["independent_realisation_of"] = f"strain_{PRIMARY_STRAIN_KEY}_r0"
    meta["calibration"] = (
        "cell size and misorientation both from the 6.2 % strain point of "
        "Zelenika et al., Sci Rep 15, 8655 (2025): 4.20 um cells, chi sigma "
        "0.36 deg"
    )
    print(f"  generating {path.name}: {meta['n_cells_requested']} cells at "
          f"{meta['target_mean_cell_diameter_um']} um, chi sigma "
          f"{meta['misorientation_sigma_deg']} deg ...", flush=True)
    phantom = generate_phantom(config)

    voxel = float(np.prod(phantom.spacing_um_zyx))
    volumes = np.bincount(phantom.labels.ravel())[1:] * voxel
    empty = int(np.count_nonzero(volumes == 0))
    volumes = volumes[volumes > 0]
    diameters = 2.0 * (3.0 * volumes / (4.0 * np.pi)) ** (1.0 / 3.0)
    misorientations = neighbour_misorientations_deg(phantom)
    meta["realised"] = {
        "n_cells": int(volumes.size), "max_label": int(phantom.labels.max()),
        "empty_labels": empty,
        "mean_cell_volume_um3": float(volumes.mean()),
        "diameter_from_mean_volume_um": float(
            2.0 * (3.0 * volumes.mean() / (4.0 * np.pi)) ** (1.0 / 3.0)),
        "equivalent_sphere_diameter_mean_um": float(diameters.mean()),
        "equivalent_sphere_diameter_median_um": float(np.median(diameters)),
        "equivalent_sphere_diameter_log_sd": float(np.std(np.log(diameters))),
        "neighbour_misorientation_mean_deg": float(np.mean(misorientations)),
        "neighbour_misorientation_median_deg": float(np.median(misorientations)),
        "neighbour_misorientation_p90_deg": float(np.percentile(misorientations, 90)),
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, labels=phantom.labels, latent=phantom.latent_field,
        field=phantom.field, wall_width_um=phantom.wall_width_um,
        spacing=np.asarray(phantom.spacing_um_zyx),
    )
    (out / f"{path.stem}.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n")
    r = meta["realised"]
    print(f"    -> {r['n_cells']} cells, d {r['diameter_from_mean_volume_um']:.2f} um "
          f"(target {meta['target_mean_cell_diameter_um']:.2f}), "
          f"misorientation {r['neighbour_misorientation_mean_deg']:.3f} deg")
    return path


def phantom_path(out: Path, strain_key: str, realization: int = 0) -> Path:
    return out / f"strain_{strain_key}_r{realization}.npz"


def build(strain_key: str, out: Path, realization: int = 0, force: bool = False) -> Path:
    """Generate one strain phantom and cache it in the benchmark's npz layout."""

    from phantom import (generate_phantom, ground_truth_region_means_deg,
                         neighbour_misorientations_deg)

    path = phantom_path(out, strain_key, realization)
    if path.exists() and not force:
        print(f"  {path.name} already present")
        return path

    config, meta = config_for(strain_key, realization)
    print(f"  generating {path.name}: {meta['n_cells_requested']} cells, "
          f"chi sigma {meta['misorientation_sigma_deg']} deg ...", flush=True)
    phantom = generate_phantom(config)

    voxel = float(np.prod(phantom.spacing_um_zyx))
    volumes = np.bincount(phantom.labels.ravel())[1:] * voxel
    # Requesting n cells does not guarantee n survive: a Laguerre site whose
    # weight loses everywhere contributes no voxels.  Empty labels must be
    # dropped before any size statistic, or log(0) poisons the dispersion.
    empty = int(np.count_nonzero(volumes == 0))
    volumes = volumes[volumes > 0]
    diameters = 2.0 * (3.0 * volumes / (4.0 * np.pi)) ** (1.0 / 3.0)
    misorientations = neighbour_misorientations_deg(phantom)

    meta["realised"] = {
        "n_cells": int(volumes.size),
        "max_label": int(phantom.labels.max()),
        "empty_labels": empty,
        "mean_cell_volume_um3": float(volumes.mean()),
        "diameter_from_mean_volume_um": float(
            2.0 * (3.0 * volumes.mean() / (4.0 * np.pi)) ** (1.0 / 3.0)
        ),
        "equivalent_sphere_diameter_mean_um": float(diameters.mean()),
        "equivalent_sphere_diameter_median_um": float(np.median(diameters)),
        "equivalent_sphere_diameter_log_sd": float(np.std(np.log(diameters))),
        "neighbour_misorientation_mean_deg": float(np.mean(misorientations)),
        "neighbour_misorientation_median_deg": float(np.median(misorientations)),
        "neighbour_misorientation_p90_deg": float(np.percentile(misorientations, 90)),
    }

    out.mkdir(parents=True, exist_ok=True)
    # Same key layout as oracle_results/cache/phantom.npz, so every existing
    # tool -- phantom_lab, capped_search -- reads these without modification.
    np.savez_compressed(
        path,
        labels=phantom.labels,
        latent=phantom.latent_field,
        field=phantom.field,
        wall_width_um=phantom.wall_width_um,
        spacing=np.asarray(phantom.spacing_um_zyx),
    )
    (out / f"strain_{strain_key}_r{realization}.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    realised = meta["realised"]
    print(f"    -> {realised['n_cells']} cells, "
          f"d(mean volume) {realised['diameter_from_mean_volume_um']:.2f} um "
          f"(target {meta['target_mean_cell_diameter_um']:.2f}), "
          f"neighbour misorientation mean {realised['neighbour_misorientation_mean_deg']:.3f} deg")
    return path


def repair(out: Path) -> None:
    """Recompute the size statistics in every metadata file from its npz.

    Needed when a build ran with an older copy of this module loaded: the
    earlier version reported ``labels.max()`` as the cell count and let empty
    labels reach ``log()``, which returns NaN dispersion.  The phantoms
    themselves are unaffected -- only the recorded statistics.
    """

    for path in sorted(out.glob("strain_*_r*.json")):
        meta = json.loads(path.read_text())
        npz = path.with_suffix(".npz")
        if not npz.exists():
            continue
        with np.load(npz) as data:
            labels = data["labels"]
            spacing = data["spacing"]
        voxel = float(np.prod(spacing))
        volumes = np.bincount(labels.ravel())[1:] * voxel
        empty = int(np.count_nonzero(volumes == 0))
        volumes = volumes[volumes > 0]
        diameters = 2.0 * (3.0 * volumes / (4.0 * np.pi)) ** (1.0 / 3.0)
        realised = meta.setdefault("realised", {})
        realised.update({
            "n_cells": int(volumes.size),
            "max_label": int(labels.max()),
            "empty_labels": empty,
            "mean_cell_volume_um3": float(volumes.mean()),
            "diameter_from_mean_volume_um": float(
                2.0 * (3.0 * volumes.mean() / (4.0 * np.pi)) ** (1.0 / 3.0)
            ),
            "equivalent_sphere_diameter_mean_um": float(diameters.mean()),
            "equivalent_sphere_diameter_median_um": float(np.median(diameters)),
            "equivalent_sphere_diameter_log_sd": float(np.std(np.log(diameters))),
        })
        path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        print(f"  repaired {path.name}: {realised['n_cells']} cells "
              f"({empty} empty labels dropped)")


def summary(out: Path):
    """Table of requested against realised microstructure, per strain."""

    import pandas as pd

    rows = []
    for strain_key in STRAIN_TREND:
        path = out / f"strain_{strain_key}_r0.json"
        if not path.exists():
            continue
        meta = json.loads(path.read_text())
        realised = meta.get("realised", {})
        rows.append({
            "strain_%": meta["strain_percent"],
            "extrapolated": meta["extrapolated_beyond_measured_range"],
            "target_d_um": meta["target_mean_cell_diameter_um"],
            "realised_d_um": round(realised.get("diameter_from_mean_volume_um", float("nan")), 3),
            "cells": realised.get("n_cells"),
            "chi_k": meta["misorientation_k"],
            "chi_sigma_deg": meta["misorientation_sigma_deg"],
            "realised_misor_mean_deg": round(
                realised.get("neighbour_misorientation_mean_deg", float("nan")), 4),
            "size_log_sd": round(realised.get("equivalent_sphere_diameter_log_sd", float("nan")), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "summary", "repair", "primary"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.action == "primary":
        print(f"building the consistent primary phantom into {args.out}")
        build_primary(args.out, force=args.force)
    elif args.action == "repair":
        print(f"repairing metadata in {args.out}")
        repair(args.out)
        print("\n" + summary(args.out).to_string(index=False))
    elif args.action == "build":
        print(f"building the strain series into {args.out}")
        for strain_key in STRAIN_TREND:
            build(strain_key, args.out, force=args.force)
        print("\n" + summary(args.out).to_string(index=False))
    else:
        print(summary(args.out).to_string(index=False))


if __name__ == "__main__":
    main()
