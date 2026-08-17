"""Make the benchmark modules importable when pytest collects this directory.

The benchmark is a set of standalone scripts rather than an installed package,
so its own directory has to be on ``sys.path`` for ``import phantom`` and
friends to resolve the same way they do when the CLI is run directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
