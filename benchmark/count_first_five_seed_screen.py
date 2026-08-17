#!/usr/bin/env python3
"""Five-seed validation for the count-first size-ordered finalists."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from count_first_analysis import PARAMETERS

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "two_stage_oracle_count_results" / "evaluations.jsonl"
POOL = HERE / "analysis" / "count_first_v1" / "screening_pool.json"
NAMESPACE = HERE / "two_stage_oracle_count_results" / "count_first_five_seed_v1"
REPORT_DIR = HERE / "analysis" / "count_first_v1"
STAGE = "count_first_five_seed_v1"
SEEDS = (1, 2, 3, 4)
PINNED = (
    "python_source_sha256",
    "compiled_extension_sha256",
    "latent_sha256",
    "truth_sha256",
    "mask_sha256",
    "matching_source_sha256",
)


def plain(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def strict_source() -> tuple[bytes, list[dict]]:
    raw = SOURCE.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError("source store lacks final newline")
    return raw, [json.loads(line) for line in raw.splitlines()]


def load_targets() -> list[dict]:
    pool = json.loads(POOL.read_text())
    if len(pool) != 5:
        raise RuntimeError(f"expected five count-first finalists, found {len(pool)}")
    if len({row["config_hash"] for row in pool}) != len(pool):
        raise RuntimeError("duplicate finalist hash")
    return pool


def verify_seed0(targets: list[dict]) -> tuple[str, dict, dict[str, dict]]:
    raw, rows = strict_source()
    source_sha = hashlib.sha256(raw).hexdigest()
    audit = json.loads((HERE / "two_stage_oracle_count_results" / "implementation_audit.json").read_text())
    by_hash = {row.get("config_hash"): row for row in rows}
    seed0 = {}
    for target in targets:
        digest = target["config_hash"]
        row = by_hash.get(digest)
        if row is None or row.get("status_category") != "ok":
            raise RuntimeError(f"missing valid source row {digest}")
        if int(row.get("candidate_order_seed", 0)) != 0:
            raise RuntimeError(f"source row is not seed 0: {digest}")
        if int(row["n_cells_pred"]) != int(row["n_cells_true"]):
            raise RuntimeError(f"screening finalist is not exact count: {digest}")
        identity = row.get("configuration_identity") or {}
        for key in PINNED:
            if (row.get(key) or identity.get(key)) != audit.get(key):
                raise RuntimeError(f"provenance mismatch for {digest}: {key}")
        seed0[digest] = row
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError("source store changed during verification")
    return source_sha, audit, seed0


def make_manifest(source_sha: str, audit: dict, seed0: dict[str, dict]) -> dict:
    manifest = {
        "manifest_version": 1,
        "screen_namespace": NAMESPACE.name,
        "stage": STAGE,
        "analysis_policy_id": "count_first_identity_v1",
        "selection_priority": [
            "exact-count seed rate",
            "median absolute count error",
            "worst absolute count error",
            "median identity F1",
            "median orientation-correct identity F1 at 0.02 degrees",
            "median ARI",
        ],
        "source_store": str(SOURCE.relative_to(HERE)),
        "source_store_sha256": source_sha,
        "source_seed0_hashes": list(seed0),
        "source_seed0_reused": True,
        "scheduled_seeds": list(SEEDS),
        "new_rows_expected": len(seed0) * len(SEEDS),
        "algorithm_id": audit.get("algorithm_id", "size_prioritised_multiseed_v1"),
        "pinned_provenance": {key: audit[key] for key in PINNED},
    }
    path = NAMESPACE / "screen_manifest.json"
    if path.exists():
        current = json.loads(path.read_text())
        source = str(current.get("source_store", ""))
        if source.endswith(manifest["source_store"]):
            current["source_store"] = manifest["source_store"]
        if current != manifest:
            raise RuntimeError("existing count-first screen manifest differs")
        atomic(path, manifest)
    else:
        atomic(path, manifest)
    identity = {
        **audit,
        "algorithm_id": manifest["algorithm_id"],
        "scientific_metrics_version": 3,
        "matching_metrics_version": 3,
    }
    atomic(NAMESPACE / "implementation_audit.json", identity)
    return manifest


def summary(values) -> dict:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "values": clean,
        "n": len(clean),
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "minimum": min(clean),
        "maximum": max(clean),
    }


def candidate_report(rows: list[dict], digest: str, parameters: dict) -> dict:
    rows = sorted(rows, key=lambda row: int(row.get("candidate_order_seed", 0)))
    seeds = [int(row.get("candidate_order_seed", 0)) for row in rows]
    if seeds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"incomplete seed set for {digest}: {seeds}")
    rejected = [
        row for row in rows
        if row.get("status_category") != "ok"
        or row.get("candidate_pass_saturated")
        or row.get("final_pass_saturated")
    ]
    errors = [
        abs(int(row.get("n_cells_pred", -10**9)) - int(row.get("n_cells_true", 0)))
        if row.get("status_category") == "ok" else 10**9
        for row in rows
    ]
    metrics = {
        name: summary(row.get(name) for row in rows)
        for name in (
            "n_cells_pred",
            "identity_f1",
            "orientation_correct_f1_at_0p02deg",
            "ari",
            "vi_total_bits",
            "boundary_assd_um",
            "boundary_f1_at_0p4um",
        )
    }
    exact_seed_count = sum(error == 0 for error in errors)
    rank_key = (
        -exact_seed_count,
        statistics.median(errors),
        max(errors),
        -metrics["identity_f1"]["median"],
        -metrics["orientation_correct_f1_at_0p02deg"]["median"],
        -metrics["ari"]["median"],
        metrics["vi_total_bits"]["median"],
        digest,
    )
    return {
        "config_hash_seed0": digest,
        "parameters": parameters,
        "seeds": seeds,
        "hard_rejected": bool(rejected),
        "exact_seed_count": exact_seed_count,
        "absolute_count_errors": errors,
        "median_absolute_count_error": statistics.median(errors),
        "worst_absolute_count_error": max(errors),
        "metrics": metrics,
        "rank_key": list(rank_key),
    }


def make_report(
    manifest: dict,
    source_sha: str,
    seed0: dict[str, dict],
    screen_rows: list[dict],
) -> dict:
    candidates = []
    for digest, source_row in seed0.items():
        parameters = {name: source_row[name] for name in PARAMETERS}
        matched = [source_row] + [
            row for row in screen_rows
            if all(row.get(name) == value for name, value in parameters.items())
        ]
        candidates.append(candidate_report(matched, digest, parameters))
    accepted = [row for row in candidates if not row["hard_rejected"]]
    rejected = [row for row in candidates if row["hard_rejected"]]
    ranked = sorted(accepted, key=lambda row: tuple(row["rank_key"])) + rejected
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
    report = {
        "manifest": manifest,
        "source_store_sha256": source_sha,
        "hard_rejection_rule": "reject any non-ok or saturated seed",
        "boundary_policy": "boundary metrics are uncertainty outputs, not selection terms",
        "candidates": ranked,
        "winner": ranked[0],
    }
    atomic(REPORT_DIR / "five_seed_screen.json", report)
    lines = [
        "# Count-first five-seed screen",
        "",
        "| Rank | Exact seeds | Median count error | Worst count error | Identity F1 | ARI | Boundary ASSD (µm) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        metrics = row["metrics"]
        lines.append(
            f"| {row['rank']} | {row['exact_seed_count']}/5 | "
            f"{row['median_absolute_count_error']:.1f} | {row['worst_absolute_count_error']} | "
            f"{metrics['identity_f1']['median']:.3f} | {metrics['ari']['median']:.3f} | "
            f"{metrics['boundary_assd_um']['median']:.3f} |"
        )
    lines += [
        "",
        "Cell-count recovery determines the ranking. Boundary displacement is reported "
        "to quantify uncertainty and does not change the selected configuration.",
    ]
    (REPORT_DIR / "five_seed_screen.md").write_text("\n".join(lines) + "\n")
    difficulty = json.loads((REPORT_DIR / "count_first_report.json").read_text())[
        "parameter_difficulty"
    ]
    winner = ranked[0]
    final_lines = [
        "# Count-first synthetic benchmark",
        "",
        "The known cell count is the primary selection criterion. Correct-cell "
        "identity is secondary, while boundary displacement is reported as uncertainty.",
        "",
        "## Selected configuration",
        "",
        f"The selected size-ordered configuration recovered the exact count in "
        f"{winner['exact_seed_count']}/5 seed orders. Its median absolute count error "
        f"was {winner['median_absolute_count_error']:.0f} cells and its worst error "
        f"was {winner['worst_absolute_count_error']} cells. Median identity F1 was "
        f"{winner['metrics']['identity_f1']['median']:.3f}, median ARI was "
        f"{winner['metrics']['ari']['median']:.3f}, and median boundary ASSD was "
        f"{winner['metrics']['boundary_assd_um']['median']:.3f} µm.",
        "",
        "## Parameter-search difficulty",
        "",
    ]
    for label, key in (
        ("Exact count", "exact_count"),
        ("Within 1%", "within_1_percent"),
        ("Within 5%", "within_5_percent"),
    ):
        item = difficulty[key]
        final_lines.append(
            f"- {label}: {item['successes']}/{item['trials']} "
            f"({100 * item['fraction']:.3f}%)."
        )
    final_lines += [
        "",
        "These rates use the complete balanced Saltelli design. Algorithmically "
        "invalid settings remain in the denominator.",
    ]
    (REPORT_DIR / "REPORT.md").write_text("\n".join(final_lines) + "\n")
    return report


def main() -> None:
    targets = load_targets()
    source_sha, audit, seed0 = verify_seed0(targets)
    NAMESPACE.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(source_sha, audit, seed0)
    os.environ["DISELL_TWO_STAGE_OUT"] = str(NAMESPACE)
    import sys

    sys.path.insert(0, str(HERE))
    import two_stage_oracle as tso

    tso.OUT = NAMESPACE
    tso.IDENTITY = json.loads((NAMESPACE / "implementation_audit.json").read_text())
    parameters = [
        {name: seed0[target["config_hash"]][name] for name in PARAMETERS}
        for target in targets
    ]
    store = NAMESPACE / "evaluations.jsonl"
    if store.exists() and len(tso.strict_jsonl(store)[1]) == len(parameters) * len(SEEDS):
        persisted = tso.strict_jsonl(store)[1]
    else:
        tso.run_tasks(parameters, STAGE, SEEDS, 8, store)
        persisted = tso.strict_jsonl(store)[1]
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError("source store changed after screen")
    report = make_report(manifest, source_sha, seed0, persisted)
    atomic(NAMESPACE / "screen_complete.json", {
        "complete": len(persisted) == len(parameters) * len(SEEDS),
        "new_rows": len(persisted),
        "expected_new_rows": len(parameters) * len(SEEDS),
        "winner": report["winner"]["config_hash_seed0"],
    })
    print(json.dumps({
        "winner": report["winner"]["config_hash_seed0"],
        "exact_seed_count": report["winner"]["exact_seed_count"],
        "new_rows": len(persisted),
    }, indent=2))


if __name__ == "__main__":
    main()
