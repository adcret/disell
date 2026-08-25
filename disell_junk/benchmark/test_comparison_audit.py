"""Read-only consistency checks for the synthetic comparison outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import capped_search as cs
import generalisation
import replicates


HERE = Path(__file__).resolve().parent


def test_primary_report_winners_match_refined_stores():
    """Section 1 must be re-derivable from the store it claims to come from.

    The winners moved from the coarse staged search to the refined sweeps
    (``runs/refined/``), so this follows them; the point of the check is that
    the report is never hand-edited away from its evidence.
    """

    report = json.loads((HERE / "analysis" /
                         "benchmark.json").read_text())
    stems = {"flood fill": "flood", "flood fill + merge": "merged",
             "KAM threshold": "kam"}
    for arm, expected in report["winners"].items():
        rows = [r for r in cs.read_rows(
                    HERE / "runs/refined" / f"{stems[arm]}_primary.jsonl")
                if r.get("status") == "ok" and not cs._disqualified(r)]
        actual = min(rows, key=cs.strict_recovery_key)
        assert actual["recovered_at_90"] == expected["recovered_at_90"]
        assert actual["n_cells_pred"] == expected["n_cells_pred"]


def test_strain_transfer_diagonals_match_strain_searches():
    payload = json.loads((HERE / "analysis" /
                          "generalisation.json").read_text())
    rows = payload["transfer"]["rows"]
    for arm in {r["arm"] for r in rows}:
        for strain in ("2p4", "3p5", "4p6", "6p2"):
            diagonal = next(r for r in rows
                            if r["arm"] == arm
                            and r["source_strain"] == strain
                            and r["target_strain"] == strain)
            final = [r for r in cs.read_rows(
                HERE / "runs/strain" / strain / "final.jsonl")
                     if r.get("status") == "ok"
                     and (r.get("arm") or "flood fill") == arm]
            winner = min(final, key=cs.strict_recovery_key)
            assert np.isclose(diagonal["recovery_rate_at_90"],
                              winner["recovery_rate_at_90"])


def test_breadth_records_coverage_scope():
    report = generalisation.breadth()
    assert report["evaluation_scope"]["flood fill"]["exhaustive"]
    assert report["evaluation_scope"]["KAM threshold"]["exhaustive"]
    assert not report["evaluation_scope"]["flood fill + merge"]["exhaustive"]


def test_kam_verification_reads_complete_baseline_grid():
    rows = replicates.verification()
    assert len(rows) == 4
    assert all(row["full_grid"] >= row["reduced_grid"] - 1e-12
               for row in rows)


def test_seed_audit_uses_latent_field():
    source = (HERE / "seed_ablation.py").read_text()
    assert 'data["latent"]' in source
    assert 'data["field"]' not in source[source.index("def determinism"):]
