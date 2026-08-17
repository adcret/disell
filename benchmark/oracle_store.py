#!/usr/bin/env python3
"""Append-only, resumable store of every evaluated configuration.

One JSON object per line, one line per ``(configuration, seed order)`` pair.
Nothing is ever overwritten, so a search can be killed and restarted at any
point and will skip what it already knows.  The key is the exact canonical
configuration string plus the seed, which is why
:func:`oracle_core.canonical` snaps the radii *before* anything is stored --
otherwise two radii that rasterise to the same footprint would be evaluated
twice and recorded as different points.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


def _plain(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class Store:
    """Evaluated rows on disk, indexed in memory by ``(config key, seed)``."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.index: dict[tuple[str, int], int] = {}
        self._stream = None
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        valid: list[dict[str, Any]] = []
        corrupt = False
        with self.path.open() as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    corrupt = True
                    continue
                valid.append(row)
                self._register(row)
        if corrupt:
            # Preserve the damaged source verbatim, then atomically reconstruct
            # a clean checkpoint containing every complete trial recovered.
            stamp = time.strftime("%Y%m%dT%H%M%S")
            preserved = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
            self.path.rename(preserved)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            with temporary.open("w") as stream:
                for row in valid:
                    stream.write(json.dumps(_plain(row)) + "\n")
                stream.flush(); os.fsync(stream.fileno())
            temporary.replace(self.path)

    def _register(self, row: dict[str, Any]) -> None:
        key = (str(row["config_key"]), int(row["random_seed"]))
        if key in self.index:
            self.rows[self.index[key]] = row
            return
        self.index[key] = len(self.rows)
        self.rows.append(row)

    def has(self, config_key: str, seed: int) -> bool:
        return (str(config_key), int(seed)) in self.index

    def get(self, config_key: str, seed: int) -> dict[str, Any] | None:
        position = self.index.get((str(config_key), int(seed)))
        return None if position is None else self.rows[position]

    def append(self, rows: Iterable[dict[str, Any]]) -> int:
        """Write new rows through to disk immediately.

        ``flush`` plus ``fsync`` on every batch: the search runs for hours and
        the whole point of the store is that killing it costs at most the batch
        in flight.
        """

        added = 0
        # If interruption cut the last JSON object in half, terminate that
        # invalid line before appending.  _load deliberately ignores it.
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as check:
                check.seek(-1, os.SEEK_END)
                needs_newline = check.read(1) != b"\n"
            if needs_newline:
                with self.path.open("ab") as repair:
                    repair.write(b"\n"); repair.flush(); os.fsync(repair.fileno())
        with self.path.open("a") as stream:
            for row in rows:
                if self.has(row["config_key"], row["random_seed"]):
                    continue
                self._register(row)
                stream.write(json.dumps(_plain(row)) + "\n")
                added += 1
            stream.flush()
            os.fsync(stream.fileno())
        return added

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.rows)

    def ok_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.rows if row.get("status") == "ok"]

    def config_keys(self) -> set[str]:
        return {key for key, _ in self.index}

    def seeds_for(self, config_key: str) -> list[int]:
        return sorted(
            seed for key, seed in self.index if key == str(config_key)
        )

    def to_frame(self):
        """All rows as a pandas DataFrame, for the analysis stage."""

        import pandas as pd

        return pd.DataFrame(self.rows)
