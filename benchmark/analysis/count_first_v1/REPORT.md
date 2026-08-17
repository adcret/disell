# Count-first synthetic benchmark

The known cell count is the primary selection criterion. Correct-cell identity is secondary, while boundary displacement is reported as uncertainty.

## Selected configuration

The selected size-ordered configuration recovered the exact count in 3/5 seed orders. Its median absolute count error was 0 cells and its worst error was 2 cells. Median identity F1 was 0.833, median ARI was 0.886, and median boundary ASSD was 0.041 µm.

## Parameter-search difficulty

- Exact count: 1/4096 (0.024%).
- Within 1%: 15/4096 (0.366%).
- Within 5%: 95/4096 (2.319%).

These rates use the complete balanced Saltelli design. Algorithmically invalid settings remain in the denominator.
