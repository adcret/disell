import json

from count_first_analysis import (
    absolute_count_error,
    count_first_key,
    run,
    success_summary,
)


def row(pred, *, identity=0.8, ari=0.8, status="ok", digest="a"):
    return {
        "status_category": status,
        "candidate_order_seed": 0,
        "config_hash": digest,
        "n_cells_pred": pred,
        "n_cells_true": 100,
        "identity_f1": identity,
        "orientation_correct_f1_at_0p02deg": identity,
        "ari": ari,
        "vi_total_bits": 0.5,
        "boundary_assd_um": 0.1,
        "boundary_f1_at_0p4um": 0.9,
        "local_threshold_deg": 0.01,
        "global_threshold_deg": 0.5,
        "footprint_tolerance": 0.1,
        "footprint_radius_um": 1.0,
        "min_cell_size": 50,
        "kam_radius_um": 1.0,
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))


def test_count_error_outranks_partition_quality():
    exact = row(100, identity=0.4, ari=0.4, digest="exact")
    inexact = row(101, identity=1.0, ari=1.0, digest="inexact")
    assert absolute_count_error(exact) == 0
    assert count_first_key(exact) < count_first_key(inexact)


def test_identity_breaks_exact_count_ties():
    lower = row(100, identity=0.7, digest="lower")
    higher = row(100, identity=0.8, digest="higher")
    assert count_first_key(higher) < count_first_key(lower)


def test_difficulty_denominator_includes_invalid_rows():
    rows = [
        row(100),
        row(101),
        row(105),
        row(100, status="expected_algorithmic_invalid"),
    ]
    assert success_summary(rows, None)["successes"] == 1
    assert success_summary(rows, None)["trials"] == 4
    assert success_summary(rows, 0.01)["successes"] == 2
    assert success_summary(rows, 0.05)["successes"] == 3


def test_run_writes_count_first_outputs(tmp_path):
    oracle = tmp_path / "oracle.jsonl"
    saltelli = tmp_path / "saltelli.jsonl"
    output = tmp_path / "out"
    write_jsonl(oracle, [row(101, identity=1.0, digest="inexact"), row(100, identity=0.8, digest="exact")])
    write_jsonl(saltelli, [row(100), row(101), row(110, status="expected_algorithmic_invalid")])

    report = run(oracle, saltelli, output)

    assert report["provisional_winner"]["config_hash"] == "exact"
    assert report["parameter_difficulty"]["exact_count"]["trials"] == 3
    assert (output / "count_first_ranking.csv").exists()
    assert (output / "REPORT.md").exists()
