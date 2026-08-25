# Flood fill versus KAM thresholding — synthetic benchmark

Evidence that `disell`'s flood-fill segmentation should be used instead of KAM
thresholding on 3D DFXM volumes, measured where ground truth exists.

The experimental volumes have no ground truth, so the argument cannot be made
on them directly. It is made here, on phantoms built from the measured cell-size
and misorientation trend, and carried over by **mechanism** — the quantity that
predicts KAM's failure is kernel radius over cell diameter, and both are
measurable without ground truth.

Report: [`analysis/BENCHMARK.md`](analysis/BENCHMARK.md)

## The result in one line

Flood fill with the merge step recovers **2,024 of 2,534 cells** at ≥90 % purity
and completeness; KAM thresholding recovers **126**. Across 20 paired replicates
— flood fill blind on fixed defaults, KAM re-tuned on every volume — the gap is
**+0.581 ± 0.135**, every pair positive.

## Layout

| module | what it does |
|---|---|
| `strict_recovery.py` | the scoring metric: cells recovered at ≥90 % both ways, plus contamination and the split/fused split |
| `merge_cells.py` | orientation-gated merge of small cells, run after the watershed |
| `capped_search.py` | the staged, resumable search; both arms, all stages |
| `strain_phantoms.py` | phantom generation from the measured strain trend |
| `strain_parameter_study.py` | per-strain optima and how they move |
| `why_flood_fill.py` | the mechanism: interior collapse, faint walls, 2D vs 3D |
| `generalisation.py` | parameter breadth and cross-strain transfer |
| `replicates.py` | the replicate study and its paired statistics |
| `parameter_economy.py` | which axes actually have to be chosen |
| `seed_ablation.py` | that the seed changes labels but not the partition |
| `dimension_replicates.py` | the 2D-vs-3D claim on independent volumes |
| `benchmark_report.py` | assembles everything into `analysis/` |
| `phantom_lab.py` | interactive front end used by the notebook |
| `phantom_playground.ipynb` | explore segmentations and parameters by hand |

Inherited from the earlier study and still required: `phantom.py` (generation),
`pipelines.py` (both segmentation arms), `bench_metrics.py` and
`object_orientation_metrics.py` (scoring). Nothing else survives from it — the
one function still needed from `oracle_core` is inlined in `capped_search.py`.

Layout: `runs/` holds every search store (`primary`, `strain`, `replicates`,
`frozen`, `seeds`, `kamverify`, `kamverify_replicates`), `phantoms/` the
volumes, `analysis/` the outputs.

## Running it

Use `~/miniconda3/envs/main/bin/python`; plain `python` lacks `disell`.

```sh
python strain_phantoms.py build          # the strain series
python strain_phantoms.py primary        # the primary phantom
python capped_search.py markers  --workers 14
python capped_search.py kam      --workers 14
python capped_search.py merge    --workers 14
python capped_search.py baseline --workers 14
python capped_search.py finalise --workers 14
python capped_search.py report
python strain_parameter_study.py run --workers 14
python why_flood_fill.py all
python generalisation.py all --workers 14
python benchmark_report.py
```

Every stage is keyed by a configuration hash and skips work already present, so
a run can be interrupted and resumed.

## Deployable parameters

Fixed across all strains; only the local threshold is worth a sweep.

```
footprint_radius_um    1.15      plateau 1.08–1.35; 0.9 costs 0.64
kam_radius_um          1.20      barely matters anywhere in range
footprint_tolerance    0.04      zero cost at every strain
global_threshold_deg   -1 (off)  inert
min_cell_size          3         merged arm; >=40 costs 0.15 at high strain
merge_size_voxels      20        zero cost
merge_threshold_deg    0.05      near-zero cost
local_threshold_deg    0.0095    merged  <- the one knob; sweep it
                       0.0151    unmerged
```

## Caveats that belong in the paper

- The phantoms are **synthetic**. They are built from the measured trend but are
  not measured material.
- The **6.2 % level is extrapolated** beyond the 0.6–4.6 % range the source
  measurements cover.
- The staged search keeps only what leads at each stage, so it drops
  configurations a direct sweep still finds — at 4.6 % strain the frozen 1-D
  sweep beats the full search by 0.005. **Reported optima are lower bounds.**
- "It only needs one knob" is **not** a comparative advantage. Asked the same
  question, KAM is equally insensitive to freezing axes. What separates the arms
  is that an arbitrary configuration is near-optimal 62 % of the time for the
  merged arm and 0.22 % for KAM, and the ceilings themselves, 0.66 against 0.08.
  The KAM optimum is reachable but not findable.

## Data

Search stores and phantoms are large and reproducible from seeds, so they stay
local and gitignored. Only the compact products in
`analysis/` are tracked.

Every phantom the study uses is calibrated to the measured cell-size and
misorientation trend together. Nothing is measured on a volume built to some
other cell size: a microstructure the material never presents cannot support a
claim about the material, whichever method it favours.

## Tests

```sh
python -m pytest -q      # 38 tests
```

`test_comparison_audit.py` ties the report back to the stores, so run it after
regenerating anything.

## Source of the strain trend

Zelenika, Cretton, Frankus, Borgi, Grumsen, Yildirim, Detlefs, Winther and
Poulsen, *Observing formation and evolution of dislocation cells during plastic
deformation*, Scientific Reports **15**, 8655 (2025),
doi:10.1038/s41598-025-88262-3.
