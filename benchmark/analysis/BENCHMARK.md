# Flood fill versus KAM thresholding on 3D DFXM volumes

All numbers are measured against synthetic phantoms with known labels.
The experimental volumes have no ground truth, so the case for using
this segmentation on them rests on the mechanism established here, not
on a score transferred from them.

Recovery is counted strictly: a true cell counts only when a predicted
cell is at least 90 % pure **and** at least 90 % complete for it.
Contamination is the fraction of labelled voxels sitting in a cell
other than their own; splitting a cell costs nothing, fusing two costs
in proportion to how wrong it is.

## 1. Each arm at its own optimum

Primary phantom, 2534 cells. Each arm is searched on its own refined grid: the flood fill over footprints of 1.0-1.45 um, KAM over kernels of 0.4-2.475 um and percentiles from 4 to 60 in half-point steps. The earlier 0.9 um radius floor was flood-fill reasoning -- a smaller kernel has no out-of-plane reach at 1.0 um z spacing -- and was never appropriate to KAM; given the range back, KAM still peaks at 1.08 um, so the floor cost it nothing. The flood fill is scored **after its merge step**, which is the arm as deployed. The parameter column identifies the exact winning configuration.

| arm | parameters | cells | recovered@90 | rate | contamination | fused | ARI |
|---|---|---|---|---|---|---|---|
| flood fill | r=1.15 um, tol=0.1, local=0.014652, min=3 | 2878 | 2008 | 0.792 | 0.0672 | 391 | 0.892 |
| flood fill + merge | r=1.345 um, tol=0.04, local=0.011721, min=3 | 2634 | 2024 | 0.799 | 0.0631 | 406 | 0.897 |
| KAM threshold | pct=22.5, r=1.08 um, min=1, conn=2 | 1702 | 138 | 0.054 | 0.2345 | 1990 | 0.718 |

### 1b. The same table on independent volumes

Section 1 is one phantom. These are independent
realisations of it under the section 7 protocol -- flood
fill blind on the defaults, KAM re-tuned on each volume.

| arm | n | recovered@90 rate | ARI |
|---|---|---|---|
| KAM threshold | 5 | 0.0510 +- 0.0036 | 0.6574 +- 0.0499 |
| flood fill | 5 | 0.7871 +- 0.0045 | 0.8839 +- 0.0317 |
| flood fill + merge | 5 | 0.7890 +- 0.0031 | 0.8839 +- 0.0317 |

Paired within each volume: flood fill +0.7361 +- 0.0046 (smallest +0.7294); flood fill + merge +0.7380 +- 0.0038 (smallest +0.7333).

## 2. The advantage is three-dimensional

Slice-wise segmentation followed by linking labels through z
is what one does without a volumetric algorithm. It is the
honest 2D baseline: an unlinked stack scores zero by
construction, since no single slice holds 90 % of a cell that
spans several layers. These 3D values are a separate
mechanism run, not the section 1 primary winner: the flood-fill
run uses its own configuration and recovers 2010 cells.

| arm | 2D + link | 3D | gain |
|---|---|---|---|
| KAM threshold | 2 | 126 | +124 |
| flood fill | 38 | 2010 | +1972 |

### 2b. The same claim on independent volumes

The 2D baseline is a construction rather than a search, so
a single number gives no sense of how much of it belongs to
the one volume it was measured on. Repeated on independent
realisations of the primary phantom, at the study defaults:

| arm | n | 2D+link recovered | 3D recovered | gain |
|---|---|---|---|---|
| KAM threshold | 6 | 3.8 +- 2.2 | 123.3 +- 6.7 | +119.5 +- 6.9 |
| flood fill | 6 | 31.3 +- 3.6 | 1944.5 +- 26.1 | +1913.2 +- 24.6 |

## 3. Where each method fails

The phantom broadens a fraction of its wall area, so a wider
wall is a fainter one with a weaker KAM ridge. If a method
fails because it needs a closed ridge, the interfaces it
fuses should have wider walls than those it keeps.

| arm | fused interfaces | wall width fused | intact | ratio | p |
|---|---|---|---|---|---|
| KAM threshold | 3401 | 0.346 um | 0.330 um | 1.05 | 1.6e-04 |
| flood fill | 240 | 0.335 um | 0.333 um | 1.01 | 4.4e-01 |

Interfaces that share a cell are not independent draws, so the
edge-level test above overstates its confidence. Resampling
cells instead (2000 bootstrap replicates over
2534 cells) gives the interval the
claim actually rests on:

| arm | width ratio | 95 % CI | median difference | 95 % CI | excludes parity |
|---|---|---|---|---|---|
| KAM threshold | 1.046 | [1.031, 1.060] | 15.3 nm | [10.2, 19.8] nm | yes |
| flood fill | 1.004 | [0.979, 1.041] | 1.3 nm | [-6.8, 13.7] nm | no |

The differential result survives the correction: KAM's
interval clears parity, the flood fill's straddles it.

## 3b. Why KAM fails: the low-KAM interior collapses

A KAM kernel of radius r raises KAM within r of any wall, so a
low-KAM component can only form in the inner core of a cell.
As cells shrink that core vanishes, and cells with no core at
all must share a component with a neighbour -- under-segmentation
that nothing downstream can undo. This is measurable on
experimental data, because it needs only the kernel radius and
the cell size, not ground truth.

Kernel radius 1.08 um throughout; only the cell size changes.

| phantom | cell diameter | r / d | median interior fraction | cells with no interior |
|---|---|---|---|---|
| strain 2p4 | 5.02 um | 0.21 | 0.087 | 22.9% |
| strain 3p5 | 4.87 um | 0.22 | 0.078 | 25.4% |
| strain 4p6 | 4.60 um | 0.23 | 0.061 | 28.7% |
| strain 6p2 | 4.21 um | 0.26 | 0.030 | 38.8% |
| primary (4.2 um cells) | 4.20 um | 0.26 | 0.033 | 35.8% |

## 4. Can the parameters be chosen without ground truth?

Each arm is measured against its **own** best, so this
compares tunability rather than accuracy. A method whose
accuracy collapses a step away from an exactly-tuned point
cannot be tuned on experimental data, however high that point
scores. Measured on the refined grids, pooled over the
four strain phantoms, each against its own best there.

| arm | configurations | mean best rate | within 95 % | within 90 % | within 80 % |
|---|---|---|---|---|---|
| flood fill | 20557 | 0.6644 | 16.50% | 37.95% | 66.13% |
| flood fill + merge | 22520 | 0.6666 | 16.85% | 38.02% | 67.00% |
| KAM threshold | 70512 | 0.0858 | 0.12% | 0.37% | 1.24% |

## 4b. Do the parameters transfer across strain?

Each strain's own optimum applied to every other strain.
Rows are where the parameters came from, columns where they
were applied; the diagonal is the matched case.

**KAM threshold**

| from \ to | 2p4 | 3p5 | 4p6 | 6p2 |
|---|---|---|---|---|
| 2p4 | 0.088 | 0.102 | 0.077 | 0.053 |
| 3p5 | 0.086 | 0.105 | 0.076 | 0.050 |
| 4p6 | 0.085 | 0.092 | 0.078 | 0.054 |
| 6p2 | 0.086 | 0.099 | 0.076 | 0.055 |

**flood fill**

| from \ to | 2p4 | 3p5 | 4p6 | 6p2 |
|---|---|---|---|---|
| 2p4 | 0.473 | 0.612 | 0.699 | 0.706 |
| 3p5 | 0.456 | 0.620 | 0.709 | 0.741 |
| 4p6 | 0.433 | 0.604 | 0.730 | 0.776 |
| 6p2 | 0.400 | 0.593 | 0.714 | 0.791 |

**flood fill + merge**

| from \ to | 2p4 | 3p5 | 4p6 | 6p2 |
|---|---|---|---|---|
| 2p4 | 0.477 | 0.632 | 0.728 | 0.745 |
| 3p5 | 0.457 | 0.636 | 0.730 | 0.773 |
| 4p6 | 0.436 | 0.612 | 0.742 | 0.796 |
| 6p2 | 0.398 | 0.594 | 0.715 | 0.800 |

## 5. How the parameters move with strain

Cell size falls and misorientation rises with strain
(Zelenika et al., Sci Rep 15, 8655 (2025)). 6.2 % is
extrapolated beyond the paper's measured 0.6-4.6 % range.

Read the local threshold column against section 5b: on the
refined grid, and with the flood fill scored after its merge,
it barely moves.

| strain_percent | arm | cell_diameter_um | chi_sigma_deg | footprint_radius_um | local_threshold_deg | percentile | min_cell_size | kam_radius_um | n_cells_pred | recovered_at_90 | rate | ari |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.4 | flood fill | 5.02 | 0.14 | 1.08 | 0.01048 | nan | 5 | 1.2 | 1999 | 715 | 0.4828 | 0.5497 |
| 2.4 | flood fill + merge | 5.02 | 0.14 | 1.08 | 0.01048 | nan | 5 | 1.2 | 1867 | 716 | 0.4835 | 0.5495 |
| 2.4 | KAM threshold | 5.02 | 0.14 | nan | nan | 30.5 | 1 | 1.08 | 1235 | 138 | 0.09318 | 0.6683 |
| 3.5 | flood fill | 4.87 | 0.22 | 1.15 | 0.01048 | nan | 5 | 1.2 | 2255 | 1033 | 0.6369 | 0.7529 |
| 3.5 | flood fill + merge | 4.87 | 0.22 | 1.08 | 0.01048 | nan | 5 | 1.2 | 2036 | 1042 | 0.6424 | 0.7719 |
| 3.5 | KAM threshold | 4.87 | 0.22 | nan | nan | 30 | 1 | 1.08 | 1421 | 179 | 0.1104 | 0.6907 |
| 4.6 | flood fill | 4.59 | 0.28 | 1.15 | 0.0131 | nan | 3 | 1.2 | 2372 | 1431 | 0.7438 | 0.877 |
| 4.6 | flood fill + merge | 4.59 | 0.28 | 1.15 | 0.01048 | nan | 3 | 1.2 | 2373 | 1427 | 0.7417 | 0.9019 |
| 4.6 | KAM threshold | 4.59 | 0.28 | nan | nan | 25 | 1 | 1.08 | 1351 | 156 | 0.08108 | 0.7302 |
| 6.2 | flood fill | 4.2 | 0.36 | 1.15 | 0.01638 | nan | 3 | 1.2 | 2636 | 2005 | 0.7941 | 0.904 |
| 6.2 | flood fill + merge | 4.2 | 0.36 | 1.15 | 0.01465 | nan | 3 | 1.2 | 2567 | 2017 | 0.7988 | 0.9058 |
| 6.2 | KAM threshold | 4.2 | 0.36 | nan | nan | 27.5 | 1 | 1.08 | 1973 | 148 | 0.05861 | 0.6911 |

### 5b. What one global value of that knob costs

Each arm has one axis anybody would actually sweep -- the
local threshold for the flood fill, the percentile for KAM.
Fixing it at a single value for every strain, against tuning
it per volume:

| arm | axis | per-strain optima | best single value | tuned | fixed | mean cost | worst |
|---|---|---|---|---|---|---|---|
| flood fill | local_threshold_deg | 0.010483, 0.010483, 0.013104, 0.016381 | 0.010483 | 0.6644 | 0.6582 | 0.0062 | 0.0139 |
| flood fill + merge | local_threshold_deg | 0.010483, 0.010483, 0.010483, 0.014652 | 0.010483 | 0.6666 | 0.6656 | 0.0010 | 0.0040 |
| KAM threshold | percentile | 30.5, 30, 25, 27.5 | 30.5 | 0.0858 | 0.0840 | 0.0018 | 0.0040 |

Two things to read here, and only one of them separates the
arms.

**The merge is what makes the threshold transferable.** After
it, one global threshold serves all four strains for 0.0010
mean and 0.0040 worst; the same fixed value costs the unmerged
arm six times as much, and its optimum still drifts upward
with strain. The arm as deployed needs no per-volume tuning at
all.

**An earlier reading of this was wrong and is worth recording.**
Measured on the unmerged arm under the retired admissibility
rule, the optimum threshold rose monotonically with strain and
fitted a clean power law in the misorientation width. That
trend was largely manufactured by the rule: it disqualified
over-segmenting configurations before the merge could act, and
so pushed the threshold up in proportion to how much each
phantom over-segmented. Assessed after the merge, the optimum
is the same value at 2.4, 3.5 and 4.6 % and rises only at
6.2 %.

**This does not separate the arms**, for the same reason
section 6 gives: KAM fixes its percentile for 2.1 % of its own
ceiling, against 0.15 % for the merged arm. Both arms tolerate
one frozen number. What separates them is section 4 and the
ceilings themselves.

## 6. How many parameters actually have to be chosen

A method with seven parameters and a method with one are not
equally usable on data without ground truth, even at equal
accuracy. Each axis below is frozen at a single global value
while every other axis is re-tuned per strain; the cost is
the loss in recovery rate at tau = 0.9, averaged over the
four strain phantoms.

The two flood-fill tables are read off the staged search's
finalists; the KAM one is read off its complete 9,108-point
grid, which is scored in full. The KAM figures are thus the
better measured of the three, which matters because they are
the ones that complicate the story. `best fixed` below means
one axis fixed while all other axes are re-tuned; it is not
the joint fallback default used by the blind replicate runs.

**flood fill**

| axis | values searched | best fixed (marginal) | mean cost | worst |
|---|---|---|---|---|
| footprint_tolerance | 3 | 0.04 | 0.0000 | 0.0000 |
| global_threshold_deg | 2 | -1 | 0.0000 | 0.0000 |
| kam_radius_um | 22 | 1.15 | 0.0007 | 0.0016 |
| min_cell_size | 7 | 20 | 0.0029 | 0.0115 |
| footprint_radius_um | 13 | 1.15 | 0.0063 | 0.0158 |
| local_threshold_deg | 9 | 0.013416 | 0.0089 | 0.0263 |

**flood fill + merge**

| axis | values searched | best fixed (marginal) | mean cost | worst |
|---|---|---|---|---|
| footprint_tolerance | 3 | 0.04 | 0.0000 | 0.0000 |
| global_threshold_deg | 2 | -1 | 0.0000 | 0.0000 |
| merge_size_voxels | 7 | 20 | 0.0000 | 0.0000 |
| min_cell_size | 5 | 3 | 0.0000 | 0.0000 |
| merge_threshold_deg | 6 | 0.05 | 0.0009 | 0.0018 |
| footprint_radius_um | 12 | 1.15 | 0.0010 | 0.0025 |
| local_threshold_deg | 4 | 0.013416 | 0.0130 | 0.0297 |

**KAM threshold**

| axis | values searched | best fixed (marginal) | mean cost | worst |
|---|---|---|---|---|
| kam_radius_um | 22 | 1.08 | 0.0000 | 0.0000 |
| min_cell_size | 6 | 3 | 0.0000 | 0.0000 |
| connectivity | 3 | 1 | 0.0003 | 0.0010 |
| percentile | 23 | 30 | 0.0016 | 0.0028 |

Only the local threshold is worth a sweep. Two axes cost
nothing at all in the unmerged arm and four in the merged
one, and the footprint radius -- the axis the cap was
imposed on -- costs 0.006 anywhere on its plateau.

**This does not separate the arms, and should not be read
as though it did.** Asked the same question, the KAM arm is
just as insensitive: its worst axis costs 0.0016. Measured
against each arm's own ceiling the three are the same to
within a percentage point -- 1.4 % for flood fill, 2.0 %
merged, 1.9 % for KAM. Every method here reaches its own
optimum with any single axis pinned, because the axes
compensate for one another.

What section 6 establishes is therefore a statement about
flood fill on its own -- it can be deployed on data with no
ground truth by choosing one number -- and not a comparative
claim. The comparison is carried by section 4, where an
arbitrary configuration is near-optimal 38.02% of the time for
the merged arm and 0.37% of the time for KAM, and by the
accuracy itself: these two ceilings are 0.67 and 0.09. The
KAM optimum is reachable but not findable.

Freezing axes one at a time does not license freezing them
together, and the staged search cannot answer that: its
finalist funnel keeps only what leads at each stage, so the
fully frozen combination never appears in its results. It was
therefore run directly (`capped_search.py --frozen`), pinning
every axis and sweeping the threshold alone.

| arm | strain | full search | frozen 1-D sweep | cost |
|---|---|---|---|---|
| flood fill | 2p4 | 0.4733 | 0.4747 | -0.0014 |
| flood fill | 3p5 | 0.6202 | 0.6252 | -0.0049 |
| flood fill | 4p6 | 0.7302 | 0.7412 | -0.0109 |
| flood fill | 6p2 | 0.7909 | 0.7889 | +0.0020 |
| flood fill | primary | 0.7928 | 0.7932 | -0.0004 |
| flood fill + merge | 2p4 | 0.4767 | 0.4760 | +0.0007 |
| flood fill + merge | 3p5 | 0.6356 | 0.6319 | +0.0037 |
| flood fill + merge | 4p6 | 0.7422 | 0.7469 | -0.0047 |
| flood fill + merge | 6p2 | 0.7996 | 0.7960 | +0.0036 |
| flood fill + merge | primary | 0.7987 | 0.7968 | +0.0020 |

Cost of the whole reduction -- six axes to one in the
unmerged arm, seven to one in the merged one: flood fill -0.0031 mean / 0.0020 worst; flood fill + merge 0.0010 mean / 0.0037 worst.

At 4.6 % strain the frozen sweep is *better* than the full
search. That is not a rounding artefact and not a point in
the defaults' favour so much as a caveat on the search: the
staged funnel discards configurations that a direct sweep
still finds. Reported optima are therefore lower bounds.

Two cliffs bound the plateau. A footprint radius of 0.9 um
costs 0.64 -- the effective floor is 1.08 um, *inside* the
searched range rather than below it -- and a min_cell_size of
40 or more costs 0.15 at 6.2 % strain, 0.59 at 80.

## 7. Does the comparison survive a different roll of the dice?

Sections 1-6 rest on one phantom realisation per strain
searched with one seed. Here each strain gets 5
further independent realisations and the comparison is
repeated on every one.

The protocol is deliberately asymmetric, and against the
conclusion. The flood-fill arm runs **blind**: every axis
pinned to the fallback defaults in `capped_search.py`, chosen on
r0 and never re-derived, so these phantoms are out of
sample. The KAM arm is **re-tuned on every replicate** over
828 configurations. If flood fill still wins, the gap cannot
be explained by unequal tuning effort.

The KAM arm's 828-point grid first has to be shown
sufficient. On r0, against the full 9,108-point grid:

| strain | full grid | reduced grid | difference |
|---|---|---|---|
| 2p4 | 0.0885 | 0.0885 | +0.0000 |
| 3p5 | 0.1048 | 0.1048 | +0.0000 |
| 4p6 | 0.0780 | 0.0780 | +0.0000 |
| 6p2 | 0.0554 | 0.0554 | +0.0000 |

Recovery rate at tau = 0.9, mean +- sd over replicates:

| arm | 2p4 | 3p5 | 4p6 | 6p2 |
|---|---|---|---|---|
| flood fill | 0.4722 +- 0.0127 | 0.6187 +- 0.0095 | 0.7341 +- 0.0141 | 0.7824 +- 0.0147 |
| flood fill + merge | 0.4828 +- 0.0138 | 0.6249 +- 0.0112 | 0.7405 +- 0.0156 | 0.7868 +- 0.0152 |
| KAM threshold | 0.0902 +- 0.0102 | 0.0892 +- 0.0086 | 0.0808 +- 0.0023 | 0.0499 +- 0.0019 |

Those runs still choose one number per phantom, the
local threshold. Fixing that too -- the fallback
default in `capped_search.py`, with no tuning on the
replicate whatsoever --
costs this much:

| arm | strain | threshold swept | threshold fixed | cost |
|---|---|---|---|---|
| flood fill | 2p4 | 0.4722 +- 0.0127 | 0.4053 +- 0.0161 | 0.0669 |
| flood fill | 3p5 | 0.6187 +- 0.0095 | 0.5863 +- 0.0079 | 0.0324 |
| flood fill | 4p6 | 0.7341 +- 0.0141 | 0.7262 +- 0.0139 | 0.0079 |
| flood fill | 6p2 | 0.7824 +- 0.0147 | 0.7824 +- 0.0147 | 0.0000 |
| flood fill + merge | 2p4 | 0.4828 +- 0.0138 | 0.4805 +- 0.0147 | 0.0023 |
| flood fill + merge | 3p5 | 0.6249 +- 0.0112 | 0.6219 +- 0.0110 | 0.0030 |
| flood fill + merge | 4p6 | 0.7405 +- 0.0156 | 0.7281 +- 0.0233 | 0.0124 |
| flood fill + merge | 6p2 | 0.7868 +- 0.0152 | 0.7501 +- 0.0320 | 0.0367 |

The cost is small on average but not evenly spread,
and the two arms need the threshold at opposite ends
of the range: unmerged it matters most at low strain
(0.066 at 2.4 %), merged at high strain (0.037 at
6.2 %). Sweeping one number is cheap, so sweep it --
but even wholly untuned the worst blind mean, 0.405,
is four times the best KAM arm managed with per-
replicate tuning.

Each replicate pairs the two arms on the *same*
phantom, so the difference is measured within a
realisation rather than across the noise between
them:

| arm minus KAM | pairs | mean gap | sd | smallest gap | all positive | paired t | p |
|---|---|---|---|---|---|---|---|
| flood fill | 20 | +0.5743 | 0.1366 | +0.3541 | yes | 18.8 | 9.8e-14 |
| flood fill + merge | 20 | +0.5812 | 0.1345 | +0.3655 | yes | 19.3 | 6.0e-14 |

The defaults were chosen on r0, so r0 could in
principle be flattered by them. It is not: the frozen
run on r0 sits inside the spread of the replicates it
never saw.

| strain | r0, full search | r0, frozen | replicates (blind) |
|---|---|---|---|
| 2p4 | 0.4767 | 0.4760 | 0.4828 +- 0.0138 |
| 3p5 | 0.6356 | 0.6319 | 0.6249 +- 0.0112 |
| 4p6 | 0.7422 | 0.7469 | 0.7405 +- 0.0156 |
| 6p2 | 0.7996 | 0.7960 | 0.7868 +- 0.0152 |

### 7b. Is that spread the microstructure or the algorithm?

The replicate spread confounds two things: the
microstructure changing between realisations, and the
flood fill's own stochasticity -- it grows regions from
seed points in an order that depends on a random seed.
Holding the phantom fixed at r0 and sweeping only the
seed separates them. The seed contributes nothing:

| arm | strain | seeds | mean | seed sd | seed range | replicate sd | seed / replicate |
|---|---|---|---|---|---|---|---|
| flood fill | 2p4 | 5 | 0.4747 | 0.0000 | 0.0000 | -- | -- |
| flood fill | 3p5 | 5 | 0.6252 | 0.0000 | 0.0000 | -- | -- |
| flood fill | 4p6 | 5 | 0.7412 | 0.0000 | 0.0000 | -- | -- |
| flood fill | 6p2 | 5 | 0.7889 | 0.0000 | 0.0000 | -- | -- |
| flood fill + merge | 2p4 | 5 | 0.4760 | 0.0000 | 0.0000 | -- | -- |
| flood fill + merge | 3p5 | 5 | 0.6324 | 0.0003 | 0.0006 | -- | -- |
| flood fill + merge | 4p6 | 5 | 0.7469 | 0.0000 | 0.0000 | -- | -- |
| flood fill + merge | 6p2 | 5 | 0.7960 | 0.0000 | 0.0000 | -- | -- |

A quantity that never moves is equally consistent with
the seed never reaching the algorithm, so that is checked
directly rather than assumed. The seed does reach it and
does change the output -- the label *numbering* differs on
every seed, unseeded runs included. What does not change
is the partition those labels describe:
`flood_fill_dfxm_two_stage` collects its seeds
deterministically and then sorts them by size.

**That invariance is conditional, and the condition is the
global threshold.** It is not a blanket property of the
algorithm. The global test compares each candidate voxel
against the region's *running* mean, whose value depends on
the order voxels were added, so wherever that test actually
binds the partition becomes order-dependent. Sweeping it on
the primary phantom with everything else fixed: off, 2.0,
0.873 and 0.30 deg all give a spread of 0 cells over five
seeds, while 0.10 deg gives a spread of 5. The defaults used
throughout this study disable the global threshold, which is
why the measured spread is zero.

The earlier synthetic benchmark, which enabled it at a tight
local threshold, correspondingly reported 358-362 cells over
five seed orders -- reproduced here exactly. Both
observations are consistent once the condition is stated.

Across 16 comparisons on four phantoms (seeds 1-3 and one seeded from `std::random_device`): ARI against seed 0 is exactly 1 in every case, cell counts match exactly, and the raw label arrays differ in every case.

So the error bars in section 7 are microstructure
variance, not the algorithm reshuffling its own output.
The single exception is the merged arm at 3.5 %, whose
seed sd is 0.0003 -- the merge step walks cells in label
order, so ties between equal-sized candidates can break
differently. It is two orders of magnitude below the
replicate spread.

## Provenance

- Phantoms: `strain_phantoms.py`, from the measured DFXM trend.
- Search: `capped_search.py`, both arms to their own optimum.
- Mechanism: `why_flood_fill.py`. Generalisation: `generalisation.py`.
- Seed ablation: `seed_ablation.py`, results in
  `runs/seeds/`.
- Replicates: `replicates.py`, results in
  `runs/replicates/`; the KAM grid check is in
  `runs/kamverify/`.
- Parameter economy: `parameter_economy.py`; the frozen searches
  are in `runs/frozen/` and the defaults they price
  are `capped_search.DEFAULTS`.
- Selection policy and its alternatives are recorded in
  `runs/primary/capped_report.json`.
* Sections 1, 4, 5 and 5b come from the refined sweeps in
  `runs/refined/` -- `refine.py kam|flood|merged`. Section 6 and
  the transfer matrix in 4b are still read off the coarse staged
  search, whose axes the refined grids do not reproduce.

