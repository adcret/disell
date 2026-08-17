# Synthetic method comparison

All methods use the same 360-cell synthetic orientation field.

| Method | Exact count | Cells | Identity F1 | ARI | Boundary ASSD (µm) |
|---|---:|---:|---:|---:|---:|
| size ordered | 3/5 | 360.0 | 0.833 | 0.886 | 0.041 |
| not size ordered | 4/20 | 359.0 | 0.840 | 0.829 | 0.075 |
| KAM | 1/1 | 360.0 | 0.725 | 0.778 | 0.080 |

The size-ordered method is selected because count recovery is primary. Boundary ASSD quantifies placement uncertainty and is not a selection term.
