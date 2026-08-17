# Synthetic 3D segmentation benchmark

> **Paper evidence boundary (2026-08-17):** this directory is the labelled
> synthetic benchmark. It is separate from the unlabelled experimental 6.2%
> DFXM volume. See [`PROVENANCE.md`](PROVENANCE.md) and
> [`provenance_ledger.json`](provenance_ledger.json).
>
> **Current selection policy:** recovered cell count is lexicographically
> primary, correct-cell identity F1 is secondary, and boundary displacement is
> reported as uncertainty. The earlier geometric-mean and cubic objectives are
> retained as historical analyses, not the paper's final selection rule.

> **Definitive manuscript algorithm (2026-08-13):** primary optimisation uses
> `disell.flood_fill_dfxm_two_stage` for size-prioritised markers, followed by
> exactly one KAM-guided watershed. `oracle_results/` and
> `continuation_results/flood_fill_random_order/` use `flood_fill_dfxm` and are
> retained only as random-order comparators. Commands and status are in
> [`two_stage_oracle_results/STATUS.md`](two_stage_oracle_results/STATUS.md).

## Versioned orientation-count continuation

Selection policy `object_orientation_count_geomean_v1` reads only the 8,000
`stage == "broad"` rows in `two_stage_oracle_f1_results/evaluations.jsonl`.
The schema-v3 source stays read-only; its SHA-256 and exact broad accounting are
pinned in the schema-v4 continuation manifest under
`two_stage_oracle_count_results/`. Old-policy adaptive rows are never imported.

Run the mandatory frozen-broad audit and deterministic adaptive-plan dry run:

```bash
cd /home/adam/Documents/Scripts/packages/disell
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  MPLCONFIGDIR=/tmp/disell-two-stage-count-mpl \
  /home/adam/miniconda3/envs/main/bin/python \
  benchmark/two_stage_oracle.py --frozen-broad-dry-run --workers 8
```

Launch or resume the adaptive-only continuation. It uses one coordinator,
eight spawned workers, one numerical thread per worker, worker recycling after
15 trials, and the existing RAM/RSS/swap safety gates:

```bash
cd /home/adam/Documents/Scripts/packages/disell
setsid --fork nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  MALLOC_ARENA_MAX=2 MPLCONFIGDIR=/tmp/disell-two-stage-count-mpl \
  /home/adam/miniconda3/envs/main/bin/python \
  benchmark/two_stage_oracle.py --continue-frozen-broad --workers 8 \
  >> benchmark/two_stage_oracle_count_results/coordinator.log 2>&1 \
  < /dev/null &
```

Monitor without changing state:

```bash
cat benchmark/two_stage_oracle_count_results/pid.json
cat benchmark/two_stage_oracle_count_results/progress.json
tail -n 40 benchmark/two_stage_oracle_count_results/coordinator.log
```

For a graceful interruption, send `SIGTERM` to the PID recorded in `pid.json`.
The coordinator stops submitting work, drains returned trials, validates each
envelope, and fsyncs it before bookkeeping. Each adaptive stage has an immutable
plan tied to the frozen-source hash, so relaunching the same command resumes only
missing new-policy hashes. Broad execution is disabled in this driver, and the
namespace lock rejects a second coordinator.

After primary completion, the downstream stages are deliberately separate and
are run in this order:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  conda run -n main python benchmark/two_stage_saltelli.py --n 512 --workers 8
conda run -n main python benchmark/two_stage_analyze.py
conda run -n main python benchmark/two_stage_paired_comparison.py \
  --workers 8 --configurations 32 --seeds 21
conda run -n main python benchmark/two_stage_strain_suite.py \
  --workers 8 --broad 256 --adaptive 128
```

`progress.json` is owned by the currently running coordinator. A read-only
sidecar maintains `scientific_progress.json` with separate durable-row,
superseded-v1, full-hash, semantic-trial, selectable-v2, and planned-v2
counters. Schema-v1 rows never enter scientific selection or modelling. Broad
coverage means 8,000 unique successful v2 `(config_key, seed)` trials; any
post-queue shortfall is filled deterministically before analysis.

The Saltelli stage is a separate balanced variance-decomposition design with
its own `saltelli_evaluations.jsonl`; it never consumes the mixed broad store. The paired
experiment writes under `continuation_results/algorithm_comparison`; the
36-phantom rerun writes under `continuation_results/two_stage_strain_suite` and
reads the preserved phantom archives without altering them.

## Cell-scale audit

The strain/difficulty continuation originally reported a constant
`cell_scale_proxy_um`. That value was not a cell measurement: the estimator
defined interfaces as the top 10% of adjacent differences, forcing a constant
face fraction on the fixed grid. The corrected label-free proxy uses an
adaptive log-gradient split and physical face areas. Independent ground-truth
3D equivalent-sphere and XY section-equivalent diameters, including explicit
edge-cell variants, are written by `cell_scale_audit.py`.

The audit reads existing phantom and trial checkpoints and does not repeat the
oracle searches. It compares no-size, corrected measurable-size, and explicitly
non-deployable ground-truth-size rules with grouped held-out evaluations. See
`continuation_results/CELL_SCALE_AUDIT.md` for the results and limitations.

## KAM failure analysis

`kam_failure_analysis.py` performs a resumable, strictly sequential comparison
of KAM as a boundary indicator, low-KAM connected-component markers,
KAM-marker watershed, saved flood-fill markers, and saved flood-fill watershed
partitions. It reuses the immutable primary phantom and one prespecified
realisation of all 12 strain×difficulty conditions. Vector-L2 KAM is recorded
as analytically equivalent to `sqrt(2)` times two-channel RMS and is not
duplicated in percentile-threshold trials.

Launch or resume:

```bash
cd /home/adam/Documents/Scripts/packages/disell
setsid --fork nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  MPLCONFIGDIR=/tmp/disell-kam-analysis-mpl \
  /home/adam/miniconda3/envs/main/bin/python \
  benchmark/kam_failure_analysis.py \
  >> benchmark/continuation_results/kam_analysis/kam_analysis.log 2>&1 \
  < /dev/null
```

The driver holds `kam_analysis.lock`, writes `kam_analysis.pid.json`, fsyncs
each trial to `kam_trials.jsonl`, and skips existing configuration keys.

Input-parity note: the preserved oracle benchmark defines its segmentation
field as cache key `latent` (`oracle_core.load_workspace` assigns this to
`workspace.field`). The KAM analysis uses that same key, so its direct
comparison is matched but idealized. Cache key `field` is the separately
blurred/noisy measurement field and was not segmented by either arm. See
`kam_analysis/input_parity_audit.json` for SHA-256 identities and the complete
12-phantom trace. A measured-field claim requires a future matched evaluation
of both algorithms, not a KAM-only rerun.

One fixed, deterministic 3D dislocation-cell phantom with known ground truth,
and a systematic search of the `adam_fix` `disell` flood-fill parameters
against it.

## The phantom

A Laguerre (power) tessellation on a `(24, 160, 160)` grid at `(1.0, 0.4, 0.4)` µm
(24 × 64 × 64 µm), 360 cells, rendered into a two-channel DFXM angular feature
field. Every property below is checked by a test rather than asserted:

| property | how it is built | realised value |
| --- | --- | --- |
| anisotropic voxels | fixed | `(1.0, 0.4, 0.4)` µm |
| log-normal cell volumes | target volumes drawn log-normal, then **realised** by fitting the power-diagram weights | σ(log V) = 0.79 for a target of 0.80, skew +0.08, KS *p* = 0.96; equivalent diameter 5.2/7.2/10.4 µm |
| chi-distributed misorientations, non-integer `k` | orientation states fitted on the cell-adjacency graph by rank-matching edge lengths to chi quantiles and minimising graph stress | fitted `k` = 1.70, σ = 0.360° against targets 1.70 and 0.360°; median 0.375° |
| heterogeneous intradomain variation | 86% smooth randomly directed affine drift + 18% weak non-radial curvature, both scaled by one per-cell log-normal amplitude clipped at 2σ; white noise 0.001° | `s_k` median 0.031°, p95 0.109°, p99 0.143°, max 0.267°, p95/median 3.5 |
| heterogeneous wall ridges | per-facet blur width, smoothly modulated, with a broadened fraction | wall KAM median 0.082°, 14.8× the interior (0.006°) |

The misorientation distribution is the important one. Independent 2-D Gaussian
cell states would force the edge magnitudes to be chi with exactly two degrees
of freedom; the states are instead fitted on the actual adjacency graph so a
non-integer `k = 1.7` is reproduced. Because `k < 2`, the distribution carries
real weight at low angle — 6.0% of facets below 0.10°, 1.7% below 0.05° — and
those are the walls a KAM threshold cannot see.

`boundary_normal_ridge_offsets_um` verifies that no angular ridge sits away from
a labelled interface: median absolute offset is exactly 0 µm. The labels and the
field agree, so any segmentation failure is the method's, not the phantom's.

### Intradomain variation

`s_k` is defined exactly as Eq. A2 of the paper, `sqrt(Var(chi) + Var(phi))`
per cell. **The experimental 0.184 deg is an `s_k`.** An earlier revision of
this benchmark mistakenly read it as a peak-to-peak range and inflated the
intradomain field by ~4.7x to match; that has been reverted.

A peak-to-peak range is still computed, as a **separately labelled diagnostic
only**. It is a different statistic with a different sampling behaviour, it is
not compared with `s_k` anywhere, and no conversion between the two is
reported.

| statistic | value |
| --- | --- |
| `s_k` p5 / p25 / p50 / p75 / p95 | 0.010 / 0.020 / **0.031** / 0.050 / 0.109 deg |
| `s_k` p99 / max | 0.143 / 0.267 deg |
| peak-to-peak range (separate diagnostic) | median 0.144 deg |

Structure of the intradomain field:

* **86% of the variance is a smooth, randomly directed affine drift** measured
  from the cell centroid, with 18% from a weak curvature (they overlap slightly
  because the split is measured by ablation).
* The curvature is **non-radial and untapered**. An earlier version tapered it
  to zero at cell boundaries to keep a per-cell amplitude continuous; that
  prints a dark ring around every cell. The residual is now flat against
  distance-to-boundary (shell medians 0.033 / 0.029 / 0.026 / 0.025 deg,
  Spearman correlation -0.10), which a test asserts.
* The per-cell log-normal amplitude is **clipped at 2σ**. Unclipped it put one
  cell at 24x the median `s_k`, which rendered as a saturated blob; the maximum
  is now 8.7x.

### Total orientation scale

| quantity | value |
| --- | --- |
| `s_crystal` = `sqrt(Var(chi) + Var(phi))` over the whole crystal | 0.333 deg |
| per-channel 1–99% ranges | 1.274 and 1.046 deg |
| `s_crystal` / median(`s_k`) | 10.8 |
| median adjacent-cell misorientation / median(`s_k`) | 12.2 |

The global spread was **not** broadened. Both ratios say the field is dominated
by between-cell variation by an order of magnitude, and a 1.0–1.3 deg
per-channel range across a 64 µm grain is the right order for this strain, so
the conditional instruction to broaden the long-wavelength distribution of cell
means did not apply.

The main figure uses a **fixed physical angular colour normalisation**
(`colour_reference_deg`, default 0.75 deg) rather than a percentile of the
data, so a colour means the same number of degrees between runs.

### Segmentation input

The benchmark segments `phantom.segmentation_field`, which is the **latent**
label-derived field: piecewise per cell, plus gradients, curvature and drift,
with sharp interfaces. The separately available `phantom.field` adds the
symmetric physical-space wall blurring and is not used for segmentation. This
keeps the demonstration clean — no detector model, no measurement blur, almost
no noise, so nothing can be blamed on a degraded measurement.

## Legacy random-order comparator search

`disell.flood_fill_dfxm` identification followed by KAM-guided watershed
refinement is retained for the algorithmic ablation only. It no longer supplies
the manuscript's primary optimisation or transferable rules. The legacy search
covered five parameters: local threshold, global threshold, footprint
tolerance, physical footprint radius and minimum cell size.

The search is **coarse-to-fine**, not a fixed grid. Round 1 evaluates a focused
768-point grid around the previous optimum. Each later round:

1. extends any axis whose best value sits on an end, by a fixed geometric factor
   of 2 -- deliberately *not* the local neighbour ratio, which shrinks as the
   axis is refined and makes the search creep outwards one tiny step per round;
2. bisects the gaps either side of the best value, until adjacent values differ
   by less than 1.30x (or 25 elements for the minimum cell size); and
3. evaluates the full product of each axis's best value and its immediate
   neighbours (3^5 = 243 points, mostly cached).

It stops when a round adds no new points, which can only happen once the best
point is interior on every axis **and** every immediate neighbouring value has
been evaluated. `selected_parameters.json` records the final axes, the per-round
history, whether the loop converged, and an explicit `on_grid_boundary` flag per
axis.

The best 15 distinct points are then repeated over **7 seed orders**. Selection
is by mean ARI; candidates within 0.002 mean ARI of the leader count as tied and
are separated by, in order, lower mean VI, lower mean boundary ASSD, and lower
seed-to-seed ARI spread.

A KAM-threshold baseline is swept over its threshold percentile and reported at
its own best ARI. Both arms use the same KAM field and the same watershed, so
what is compared is how the markers are identified.

## Running it

### Memory-safe random-order oracle search (preserved comparator)

The oracle search is strictly sequential.  It rejects `--processes` values
other than 1 and evaluates every segmentation in a fresh spawned process.  The
parent monitors process-tree RSS, available physical RAM, and swap.  By default
RSS is capped at the smaller of 40% of physical RAM and physical RAM minus
4 GiB; a lower explicit cap can be supplied.  Every completed trial is fsynced
to `evaluations.jsonl`, equivalent physical footprints are canonicalised before
hashing, and a file lock prevents duplicate drivers.

Run the mandatory one-trial profile plus 20-trial stability gate first:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
conda run --no-capture-output -n main \
python benchmark/oracle_smoke.py --out-dir benchmark/oracle_results
```

Resume (completed parameter/seed hashes are skipped):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
conda run --no-capture-output -n main python benchmark/oracle_search.py \
  --out-dir benchmark/oracle_results --processes 1 --broad 3000 \
  --deadline 2026-08-13T09:00
```

Run derived selection, tables, finalist volumes, and figures after the search:

```bash
conda run --no-capture-output -n main \
python benchmark/oracle_analyze.py --out-dir benchmark/oracle_results
```

The legacy command below reproduces the earlier five-parameter benchmark and
is separate from the resumable six-parameter oracle workflow.

```bash
python benchmark/synthetic_3d_benchmark.py --out-dir benchmark/results
```

### Detached strain/difficulty continuation

The completed fixed-primary oracle is immutable.  The continuation driver
reads it only to produce diagnostics, then stores each smaller strain/difficulty
phantom under `benchmark/continuation_results/phantoms/` with its own manifest,
append-only `evaluations.jsonl`, frozen refinement/repeat plans, and atomic
progress file.  It is strictly sequential and saves full labels only for the
balanced finalist of each phantom.

Launch or resume (the same command skips validated completed units):

```bash
cd /home/adam/Documents/Scripts/packages/disell
setsid nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  MPLCONFIGDIR=/tmp/disell-continuation-mpl \
  /home/adam/miniconda3/envs/main/bin/python \
  benchmark/overnight_continue.py \
  --deadline 2026-08-13T09:00:00+02:00 \
  > benchmark/continuation_results/continue.log 2>&1 < /dev/null &
```

Phone-friendly monitoring:

```bash
cd /home/adam/Documents/Scripts/packages/disell
tail -n 30 benchmark/continuation_results/continue.log
cat benchmark/continuation_results/overnight_continue.pid.json
cat benchmark/continuation_results/status.json
find benchmark/continuation_results/phantoms -name evaluations.jsonl -exec wc -l {} + | tail
ps -o pid,ppid,sid,stat,rss,etime,cmd -p "$(python -c 'import json; print(json.load(open("benchmark/continuation_results/overnight_continue.pid.json"))["pid"])')"
free -h
```

Graceful stop (the active trial may finish; at most that one trial is in
flight), followed by forceful stop only if necessary:

```bash
kill -TERM "$(python -c 'import json; print(json.load(open("benchmark/continuation_results/overnight_continue.pid.json"))["pid"])')"
kill -KILL "$(python -c 'import json; print(json.load(open("benchmark/continuation_results/overnight_continue.pid.json"))["pid"])')"
```

Resume with the launch command above.  The lock prevents duplicates; manifests
and frozen task plans prevent a resumed stage from silently changing identity.

### Audit and finalise completed continuation results

Complete only missing primary selected-solution seed diagnostics (existing
configuration/seed pairs are skipped and every new row is fsynced):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
conda run --no-capture-output -n main \
python benchmark/complete_primary_diagnostics.py
```

Evaluate or resume the saved leakage-free cross-validated rules without
rerunning a search, then regenerate audited tables, figures and reports:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
conda run --no-capture-output -n main \
python benchmark/evaluate_transferable_rules.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
MPLCONFIGDIR=/tmp/disell-finalise-mpl \
conda run --no-capture-output -n main \
python benchmark/finalise_continuation_results.py
```

The substantive final status is
[`continuation_results/OVERNIGHT_STATUS.md`](continuation_results/OVERNIGHT_STATUS.md),
not the earlier launch-time status text.

About 25 minutes. Outputs:

| file | contents |
| --- | --- |
| `search.csv` | every stage-1, stage-2 and KAM-baseline point, including failures |
| `selected_parameters.json` | the selected parameters, how they were selected, all finalist summaries |
| `metrics.json` | measured phantom properties and the final scores |
| `kam_failure_modes.csv` | percolation and fragmentation of KAM markers over a fine threshold sweep |
| `benchmark_figure.{png,pdf}` | the main figure |
| `kam_failure_modes.{png,pdf}` | the two KAM failure modes against threshold |
| `intradomain_spread_cdf.{png,pdf}` | CDFs of both intradomain spread statistics |
| `facet_recovery.{png,pdf}` | cell-pair separation against true facet misorientation |
| `phantom_and_labels.npz` | the arrays behind the figures |

## Figures

`benchmark_figure.{png,pdf}` — five panels over one shared background: (a) the
angular feature map, (b–d) the same map with the ground-truth, KAM-threshold and
flood-fill boundaries, and (e) the KAM field both arms act on.

Boundaries are opaque black and **exactly one output pixel** wide. That needs
two things, both of which a normal matplotlib overlay gets wrong:

* boundaries are found on the display raster *after* nearest-neighbour
  upsampling and written straight into it, rather than drawn as line artists
  that would be antialiased into grey; and
* the axes are positioned with explicit pixel arithmetic so one raster pixel is
  one device pixel. `constrained_layout` and `bbox_inches="tight"` both rescale
  the axes and are therefore not used.

`test_rendered_boundaries_are_exactly_one_output_pixel` reads the saved PNG back
and asserts each panel equals the composited raster pixel for pixel, with the
black boundary pixels matching one for one. That is stronger than measuring run
lengths, which cannot tell a line's width from its length.
`test_figure_is_pixel_exact_in_size` checks the saved image is the size the
layout intended.

`kam_failure_modes.{png,pdf}` — the two failure modes of a KAM threshold against
the threshold percentile, with the flood-fill markers as horizontal reference
lines. Both are measured on the raw markers, before any refinement, because a
watershed can only propagate percolation and fragmentation, never undo them.

## Tests

```bash
python -m pytest benchmark/test_synthetic_3d_benchmark.py
```

Covers phantom determinism, each distributional claim above, metric direction
and physical scaling, flood-fill reproducibility and mask handling, one-pixel
opaque boundary rendering, and that the selection really uses the mean over seed
orders rather than the best single run.

## What the run in `results/` shows

**Phantom accepted**: flood fill leaves 7 of 360 cells unseeded (1.9%) at its
optimum, against the 10% rejection limit.

### Flood-fill search

Coarse-to-fine over the genuinely active parameters, now including the
**KAM/watershed footprint radius**. Converged after 8 rounds, 1438 unique
points, 0 failures, 21 minutes, no axis on a boundary.

```
local_threshold_deg = 0.00949    ARI       = 0.888 ± 0.000   (7 seed orders)
footprint_tolerance = 0.168      VI total  = 0.462 ± 0.004 bits
footprint_radius_um = 1.3        ASSD      = 0.062 ± 0.001 µm
min_cell_size       = 56         cells     = 472 ± 4   (truth 360, +112)
kam_radius_um       = 1.265
```

Selection was mean ARI first, then VI, ASSD, absolute cell-count error, leaked
volume, split cells, unseeded cells.

**The near-optimal plateau**, rather than arbitrary values inside it. Five
candidates fall within 0.002 mean ARI:

| local | fp tol | radius | min size | kam radius | ARI | VI | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00949 | 0.168 | 1.3 | 38 | 1.265 | 0.8882 | 0.471 | 515 |
| 0.00949 | 0.168 | 1.3 | 56 | 1.265 | 0.8879 | 0.462 | 472 |
| 0.00949 | 0.200 | 1.3 | 19 | 1.265 | 0.8877 | 0.523 | 628 |
| 0.00949 | 0.200 | 1.3 | 56 | 1.265 | 0.8865 | 0.481 | 466 |
| 0.00949 | 0.200 | 1.3 | 75 | 1.265 | 0.8863 | 0.471 | 435 |

Local threshold, footprint radius and KAM radius are pinned across the whole
plateau; the footprint tolerance (0.168–0.20) and the minimum cell size
(19–75, giving 435–628 cells) are only weakly determined and should be read as
a range, not as tuned values.

**The footprint tolerance had to be put back into the search.** It was inactive
on the previous phantom, so it was initially held fixed at 0.025 — but the
re-check profile found ARI 0.884 at 0.20 against 0.841 at 0.025, a 0.043 loss
from assuming it stayed inactive. Only the global threshold remains held, and
its profile confirms it is genuinely flat: 0.888 at 0.17 and 0.888 from 0.30 to
1.2, falling only below 0.10.

### Errors against true facet misorientation

Fraction of ground-truth **cell pairs actually separated** (dominant predicted
label differs — a measure an over-segmenting method cannot win):

| facet misorientation | facets | flood fill | KAM threshold |
| --- | --- | --- | --- |
| 0.00–0.05° | 46 | 0.739 | 0.717 |
| 0.05–0.10° | 95 | 0.853 | 0.705 |
| 0.10–0.20° | 280 | **0.982** | 0.900 |
| 0.20–0.40° | 705 | **0.996** | 0.949 |
| > 0.40° | 963 | **0.999** | 0.965 |

This separates the two failure types the analysis was for. Flood fill recovers
98–100% of every facet above 0.10°, so its residual error is **not** a
parameter-search failure on resolvable boundaries — what it misses is
concentrated below 0.10°, where the angular contrast approaches zero and no
method has information to work with. The KAM threshold misses 3.5% of facets
even above 0.40° and 10% between 0.10 and 0.20°: those are recoverable
boundaries, so that is a method limitation rather than an information limit.

A first version of this metric measured the fraction of each facet along which
the prediction changes label anywhere, and reported flood fill at 1.000 in every
band including 0.00–0.05°. That was an artefact: a prediction with 472 domains
against 360 true cells changes label somewhere along essentially every facet.
The pair-separation measure above replaces it.

**KAM baseline** at its own best ARI, with its own radius searched (40th
percentile, radius 1.0 µm): ARI 0.705, VI 1.22 bits, ASSD 0.10 µm, 242 cells
(−118).

## Limitations

* One phantom and one seed. This measures parameter sensitivity and the KAM
  failure modes on a known structure; it does not estimate variability across
  microstructures.
* Segmentation runs on the latent field, not the measured one. That is
  deliberate — it removes any suspicion that KAM fails because of blur or noise
  — but it also means the absolute scores are optimistic for both methods, and
  the benchmark says nothing about centre-of-mass versus multi-peak analysis.
* The percolation result depends on the low-angle tail of the misorientation
  distribution and on the intradomain spread. It is a statement about a chi
  distribution with `k = 1.7`, `sigma = 0.36 deg` and `s_k` median 0.166 deg,
  not about KAM thresholding in general.
* Two of the five searched parameters are inactive at the optimum, so the
  search is really fitting three. A grid search cannot distinguish "optimal"
  from "irrelevant" without a profile like `plateau_probe.json`.
* The reported flood-fill optimum sits in a regime where the algorithm is
  seed-order independent. Its zero seed-to-seed spread is therefore a property
  of that operating point, not evidence that the method is generally stable.
* The KAM baseline is reported at its own best ARI, which is an oracle choice.
  So is the flood-fill selection. Both are upper bounds on what a user without
  ground truth would achieve, and neither should be used to justify parameters
  for experimental data.
