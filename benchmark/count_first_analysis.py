#!/usr/bin/env python3
"""Count-first selection and parameter-search difficulty analysis.

The synthetic benchmark has known ground truth. Selection is therefore
lexicographic: recover the cell count first, recover cell identities second,
and use partition quality only as later tie-breakers. Boundary metrics are
reported as uncertainty and never outrank cell-count or identity recovery.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
ORACLE_STORE = HERE / "two_stage_oracle_count_results" / "evaluations.jsonl"
SALTELLI_STORE = HERE / "two_stage_saltelli_v4_results" / "saltelli_evaluations.jsonl"
OUT = HERE / "analysis" / "count_first_v1"
PARAMETERS = (
    "local_threshold_deg",
    "global_threshold_deg",
    "footprint_tolerance",
    "footprint_radius_um",
    "min_cell_size",
    "kam_radius_um",
)
POLICY_ID = "count_first_identity_v1"


def provenance_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(HERE.resolve()))
    except ValueError:
        return str(path.resolve())


def strict_jsonl(path: Path) -> tuple[bytes, list[dict]]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"partial final JSONL line: {path}")
    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"non-object row at {path}:{line_number}")
        rows.append(row)
    return raw, rows


def finite(row: dict, key: str, default: float) -> float:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def absolute_count_error(row: dict) -> int:
    try:
        return abs(int(row["n_cells_pred"]) - int(row["n_cells_true"]))
    except (KeyError, TypeError, ValueError):
        return 2**31 - 1


def count_first_key(row: dict) -> tuple:
    """Deterministic single-run ranking, in scientific priority order."""
    return (
        absolute_count_error(row),
        -finite(row, "identity_f1", -math.inf),
        -finite(row, "orientation_correct_f1_at_0p02deg", -math.inf),
        -finite(row, "ari", -math.inf),
        finite(row, "vi_total_bits", math.inf),
        finite(row, "boundary_assd_um", math.inf),
        str(row.get("config_hash", "")),
    )


def selectable(row: dict) -> bool:
    if row.get("status_category") != "ok":
        return False
    if int(row.get("candidate_order_seed", row.get("seed", 0)) or 0) != 0:
        return False
    return all(key in row for key in ("n_cells_pred", "n_cells_true", *PARAMETERS))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - radius, centre + radius


def success_summary(rows: Iterable[dict], relative_limit: float | None) -> dict:
    rows = list(rows)
    success = 0
    for row in rows:
        if row.get("status_category") != "ok":
            continue
        error = absolute_count_error(row)
        true = int(row["n_cells_true"])
        if relative_limit is None:
            accepted = error == 0
        else:
            accepted = error / true <= relative_limit
        success += int(accepted)
    low, high = wilson(success, len(rows))
    return {
        "successes": success,
        "trials": len(rows),
        "fraction": success / len(rows) if rows else None,
        "wilson_95_low": low,
        "wilson_95_high": high,
    }


def projection(row: dict, rank: int) -> dict:
    return {
        "rank": rank,
        "config_hash": row["config_hash"],
        **{name: row[name] for name in PARAMETERS},
        "n_cells_pred": int(row["n_cells_pred"]),
        "n_cells_true": int(row["n_cells_true"]),
        "absolute_count_error": absolute_count_error(row),
        "identity_f1": row.get("identity_f1"),
        "orientation_f1_0p02": row.get("orientation_correct_f1_at_0p02deg"),
        "ari": row.get("ari"),
        "vi_total_bits": row.get("vi_total_bits"),
        "boundary_assd_um": row.get("boundary_assd_um"),
        "boundary_f1_at_0p4um": row.get("boundary_f1_at_0p4um"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    oracle_store: Path = ORACLE_STORE,
    saltelli_store: Path = SALTELLI_STORE,
    output: Path = OUT,
) -> dict:
    oracle_raw, oracle_rows = strict_jsonl(oracle_store)
    saltelli_raw, saltelli_rows = strict_jsonl(saltelli_store)
    ranked_source = sorted((row for row in oracle_rows if selectable(row)), key=count_first_key)
    if not ranked_source:
        raise RuntimeError("no selectable size-ordered benchmark rows")
    ranking = [projection(row, index) for index, row in enumerate(ranked_source, 1)]
    exact = [row for row in ranking if row["absolute_count_error"] == 0]
    screening_pool = exact[:5]
    difficulty = {
        "design": "balanced Saltelli matrix over the declared six-parameter bounds",
        "denominator_includes_expected_algorithmic_invalid": True,
        "exact_count": success_summary(saltelli_rows, None),
        "within_1_percent": success_summary(saltelli_rows, 0.01),
        "within_5_percent": success_summary(saltelli_rows, 0.05),
        "interpretation": (
            "These fractions estimate how rarely untuned parameter vectors recover "
            "the target count over the prespecified search box. They are not a "
            "probability for parameter choices drawn from another distribution."
        ),
    }
    report = {
        "policy_id": POLICY_ID,
        "policy": [
            "minimum absolute cell-count error",
            "maximum identity F1",
            "maximum orientation-correct identity F1 at 0.02 degrees",
            "maximum ARI",
            "minimum VI",
            "minimum boundary ASSD",
            "deterministic configuration hash",
        ],
        "boundary_policy": (
            "Boundary displacement is reported with uncertainty but is not allowed "
            "to outrank cell-count or identity recovery."
        ),
        "oracle_store": provenance_path(oracle_store),
        "oracle_store_sha256": hashlib.sha256(oracle_raw).hexdigest(),
        "oracle_rows": len(oracle_rows),
        "selectable_seed0_rows": len(ranking),
        "exact_count_rows": len(exact),
        "saltelli_store": provenance_path(saltelli_store),
        "saltelli_store_sha256": hashlib.sha256(saltelli_raw).hexdigest(),
        "parameter_difficulty": difficulty,
        "screening_pool": screening_pool,
        "provisional_winner": ranking[0],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "count_first_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output / "count_first_ranking.json").write_text(
        json.dumps(ranking, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    write_csv(output / "count_first_ranking.csv", ranking)
    (output / "screening_pool.json").write_text(
        json.dumps(screening_pool, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    exact_difficulty = difficulty["exact_count"]
    one_difficulty = difficulty["within_1_percent"]
    five_difficulty = difficulty["within_5_percent"]
    winner = ranking[0]
    lines = [
        "# Count-first synthetic benchmark",
        "",
        "The known cell count is the primary selection criterion. Identity recovery is "
        "the first tie-breaker; boundary placement is reported separately.",
        "",
        "## Provisional seed-0 winner",
        "",
        f"- Cells: {winner['n_cells_pred']}/{winner['n_cells_true']}",
        f"- Identity F1: {winner['identity_f1']:.3f}",
        f"- ARI: {winner['ari']:.3f}",
        f"- Boundary ASSD: {winner['boundary_assd_um']:.3f} µm",
        "",
        "This is provisional until the five-seed screen is complete.",
        "",
        "## How difficult is parameter selection?",
        "",
        f"- Exact count: {exact_difficulty['successes']}/{exact_difficulty['trials']} "
        f"({100 * exact_difficulty['fraction']:.3f}%).",
        f"- Within 1%: {one_difficulty['successes']}/{one_difficulty['trials']} "
        f"({100 * one_difficulty['fraction']:.3f}%).",
        f"- Within 5%: {five_difficulty['successes']}/{five_difficulty['trials']} "
        f"({100 * five_difficulty['fraction']:.3f}%).",
        "",
        "These rates come from the complete balanced Saltelli design and include "
        "algorithmically invalid settings in the denominator.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    report = run(output=args.output)
    print(json.dumps({
        "winner": report["provisional_winner"]["config_hash"],
        "exact_count_rows": report["exact_count_rows"],
        "difficulty": report["parameter_difficulty"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
