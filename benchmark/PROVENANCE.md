# Paper evidence provenance

The paper uses two distinct sources of evidence. They must not be combined into
one “6.2% segmentation” result.

## Synthetic benchmark

The primary benchmark is the fixed 360-cell phantom in
`oracle_results/cache/phantom.npz`. Its orientation statistics were motivated by
approximately 6% strain, but it is not measured material. The labels are known,
so this is the only evidence used to:

- select segmentation parameters;
- rank size-ordered, unordered, and pure-KAM methods;
- measure cell-count and identity recovery;
- quantify boundary displacement; and
- estimate how difficult parameter selection is.

The independent `continuation_results/phantoms/strain_*` suite tests transfer
across strain proxies and difficulty levels. The explicit `strain_6p2_*`
phantoms remain synthetic.

## Experimental 6.2% volume

The real 6.2% result comes from the 11-layer DFXM volume rooted at
`~/Documents/Data/4dcells/111_cells_6-2pct_mosa_2x_raw`. It has no cell-level
ground truth. It can show the measured partition and report descriptive domain
statistics, but it cannot establish segmentation accuracy or select benchmark
parameters.

The manuscript/notebook result (3411 flood-fill domains and 1811 KAM domains)
and the alternate `paper_outputs` result (1129 and 941) are separate
experimental runs. Gate 3 is deferred, so neither is regenerated or silently
declared canonical here.

## Selection rule

The synthetic benchmark is ranked in this order:

1. smallest absolute error in recovered cell count;
2. highest correct-cell identity F1;
3. highest orientation-correct identity F1;
4. partition agreement metrics.

Boundary displacement is an uncertainty output. A small boundary improvement
cannot compensate for a worse cell count or worse identity recovery.

The machine-readable version of this separation is
`provenance_ledger.json`.
