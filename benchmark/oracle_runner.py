#!/usr/bin/env python3
"""Memory-bounded, strictly sequential evaluation of oracle configurations.

Each segmentation is evaluated in a fresh spawned process.  The parent samples
RSS, available RAM, and swap while it runs and can terminate that one trial
without damaging the append-only result store.  There is deliberately no pool:
at most one label volume and one trial's temporary arrays exist at a time.
"""

from __future__ import annotations

import gc
import hashlib
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

# These must be set before a spawned worker imports NumPy/SciPy.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import oracle_core as oc


@dataclass(frozen=True)
class MemoryPolicy:
    rss_limit_bytes: int
    min_available_fraction: float = 0.20
    max_swap_growth_bytes: int = 384 * 1024**2
    poll_seconds: float = 0.20

    @classmethod
    def safe_default(cls, requested_gib: float | None = None) -> "MemoryPolicy":
        import psutil
        total = int(psutil.virtual_memory().total)
        safe = min(int(0.40 * total), max(512 * 1024**2, total - 4 * 1024**3))
        if requested_gib is not None:
            safe = min(safe, int(float(requested_gib) * 1024**3))
        return cls(rss_limit_bytes=safe)


def _base_row(params: dict[str, Any], seed: int) -> dict[str, Any]:
    config = oc.Config(**params)
    return {
        "config_key": config.key(), "parameter_hash": config.hash(),
        "trial_hash": hashlib.sha256(
            f"{config.hash()}|seed={int(seed)}".encode()).hexdigest(),
        "random_seed": int(seed), **config.as_dict(),
    }


def _worker(params: dict[str, Any], seed: int, cache_dir: str,
            with_boundary: bool, result_queue) -> None:
    """Evaluate exactly one trial and return scalars through a small queue."""
    oc.limit_threads()
    row = _base_row(params, seed)
    try:
        workspace = oc.load_workspace(cache_dir)
        row.update(oc.evaluate(oc.Config(**params), workspace, int(seed),
                               with_boundary=with_boundary))
    except BaseException as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc(limit=6))
    finally:
        result_queue.put(row)
        gc.collect()


def _tree_rss(process, psutil_module) -> int:
    try:
        return int(process.memory_info().rss) + sum(
            int(child.memory_info().rss) for child in process.children(recursive=True)
        )
    except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
        return 0


def _evaluate_spawned(params: dict[str, Any], seed: int, cache_dir: Path | str,
                      with_boundary: bool, policy: MemoryPolicy) -> dict[str, Any]:
    import psutil
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    child = context.Process(target=_worker, args=(params, seed, str(cache_dir),
                                                  with_boundary, result_queue))
    swap_start = int(psutil.swap_memory().used)
    peak = 0
    reason = None
    child.start()
    observed = psutil.Process(child.pid)
    while child.is_alive():
        rss = _tree_rss(observed, psutil)
        peak = max(peak, rss)
        memory = psutil.virtual_memory()
        swap_growth = max(0, int(psutil.swap_memory().used) - swap_start)
        if rss > policy.rss_limit_bytes:
            reason = f"RSS limit exceeded ({rss} > {policy.rss_limit_bytes})"
        elif memory.available / memory.total < policy.min_available_fraction:
            reason = "available RAM fell below configured fraction"
        elif swap_growth > policy.max_swap_growth_bytes:
            reason = f"swap grew by {swap_growth} bytes"
        if reason:
            child.terminate()
            child.join(timeout=10)
            if child.is_alive():
                child.kill(); child.join()
            break
        child.join(timeout=policy.poll_seconds)
    child.join()
    try:
        row = result_queue.get(timeout=2) if reason is None else _base_row(params, seed)
    except queue.Empty:
        row = _base_row(params, seed)
        reason = reason or f"worker exited {child.exitcode} without a result"
    finally:
        result_queue.close(); result_queue.join_thread()
    row["peak_rss_bytes"] = int(peak)
    row["available_ram_bytes_after"] = int(psutil.virtual_memory().available)
    row["swap_used_bytes_after"] = int(psutil.swap_memory().used)
    if reason:
        row.update(status="memory_abort", error=reason)
    return row


def precompute_kam(radii_um: Sequence[float], cache_dir: Path | str,
                   processes: int = 1) -> int:
    """Compute missing KAM fields sequentially; ``processes`` is ignored."""
    workspace = oc.load_workspace(cache_dir)
    wanted = sorted({workspace.radius_classes.index_of(r) for r in radii_um})
    made = 0
    for index in wanted:
        path = Path(cache_dir) / f"kam_class_{index:03d}.npy"
        if path.exists():
            try:
                field = __import__("numpy").load(path, mmap_mode="r", allow_pickle=False)
                if field.shape == workspace.labels.shape and field.dtype.kind == "f":
                    del field
                    continue
            except Exception:
                stamp = time.strftime("%Y%m%dT%H%M%S")
                path.rename(path.with_name(f"{path.name}.corrupt-{stamp}"))
        workspace.kam(float(workspace.radius_classes.classes[index]["radius_um"]))
        workspace.clear_kam_cache()
        gc.collect()
        made += 1
    return made


def evaluate_batch(tasks: Sequence[tuple[dict[str, Any], int]],
                   cache_dir: Path | str, processes: int = 1, *,
                   with_boundary: bool = True, chunksize: int = 1,
                   progress_every: int = 25, label: str = "",
                   memory_limit_gib: float | None = None) -> Iterable[dict[str, Any]]:
    """Yield completed trials sequentially, with live memory enforcement."""
    del processes, chunksize
    if memory_limit_gib is None and os.environ.get("DISELL_MEMORY_LIMIT_GIB"):
        memory_limit_gib = float(os.environ["DISELL_MEMORY_LIMIT_GIB"])
    policy = MemoryPolicy.safe_default(memory_limit_gib)
    deadline = float(os.environ.get("DISELL_DEADLINE_EPOCH", "inf"))
    started = time.perf_counter()
    for index, (params, seed) in enumerate(tasks, start=1):
        if time.time() >= deadline:
            print(f"    {label}deadline reached; no new trial launched", flush=True)
            return
        row = _evaluate_spawned(params, seed, cache_dir, with_boundary, policy)
        yield row
        if row.get("status") == "memory_abort":
            print(f"    {label}stopped safely: {row['error']}", flush=True)
            return
        if progress_every and (index % progress_every == 0 or index == len(tasks)):
            elapsed = time.perf_counter() - started
            rate = index / max(elapsed, 1e-9)
            print(f"    {label}{index}/{len(tasks)}  {rate:.3f}/s  "
                  f"peak {row['peak_rss_bytes']/1024**3:.2f} GiB  "
                  f"eta {(len(tasks)-index)/max(rate,1e-9)/60:.1f} min", flush=True)
