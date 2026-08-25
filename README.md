# disell

`disell` provides the core algorithms for identifying and segmenting
dislocation cells in dark-field X-ray microscopy data, including flood fill,
KAM, watershed refinement, registration, metrics and layer handling.

The repository contains the installable package under `src/disell`, its C++
extension under `src/cpp`, tests, documentation and a small example notebook.
The synthetic phantom generator and paper-specific benchmark archive are kept
under [`disell_junk/`](disell_junk/).

Install the package with:

```bash
pip install .
```
