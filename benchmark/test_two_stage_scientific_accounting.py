import json

import two_stage_accounting as tsa
import two_stage_saltelli as saltelli
import two_stage_v2_completion as completion


def row(config, seed, version, status="ok", stage="broad", digest=None):
    return {
        "algorithm_id": tsa.ALGORITHM_ID,
        "scientific_metrics_version": version,
        "status_category": status,
        "config_key": config,
        "candidate_order_seed": seed,
        "config_hash": digest or f"{config}-{seed}-v{version}-{status}",
        "stage": stage,
    }


def test_live_counts_separate_storage_semantics_and_selectable_v2():
    rows = [
        row("a", 0, 1, digest="v1-a"),
        row("a", 0, 2, digest="v2-a-old"),
        row("a", 0, 2, digest="v2-a-new"),
        row("b", 0, 2, status="expected_algorithmic_invalid"),
        row("c", 2, 1),
    ]
    counts = tsa.accounting(rows)
    assert counts["durable_rows"] == 5
    assert counts["superseded_v1_rows"] == 2
    assert counts["unique_full_hashes"] == 5
    assert counts["unique_trials"] == 3
    assert counts["selectable_v2_trials"] == 1
    assert counts["broad_selectable_v2_trials"] == 1
    assert counts["v1_v2_semantic_trial_overlap"] == 1
    assert tsa.unique_rows(rows, tsa.is_selectable_v2) == [rows[2]]


def test_v1_never_becomes_selectable_even_when_status_is_ok():
    assert not tsa.is_selectable_v2(row("a", 0, 1))
    assert tsa.is_selectable_v2(row("a", 0, 2))


def test_later_ceiling_validation_does_not_erase_broad_coverage():
    broad = row("a", 0, 2, stage="broad", digest="limit-700k")
    ceiling = row(
        "a", 0, 2, stage="candidate_iteration_ceiling_validation",
        digest="limit-1m",
    )
    counts = tsa.accounting([broad, ceiling])
    assert counts["selectable_v2_trials"] == 1
    assert counts["broad_selectable_v2_trials"] == 1


def fake_identity():
    return {
        "python_source_sha256": "python",
        "compiled_extension_sha256": "extension",
        "latent_sha256": "latent",
        "truth_sha256": "truth",
        "mask_sha256": "mask",
        "matching_source_sha256": "matching",
    }


def parameters():
    return {
        "local_threshold_deg": 0.01,
        "global_threshold_deg": -1.0,
        "footprint_tolerance": 0.5,
        "footprint_radius_um": 1.0,
        "min_cell_size": 50,
        "kam_radius_um": 1.0,
    }


def test_saltelli_matrix_coordinates_have_independent_identities():
    saltelli.IDENTITY = fake_identity()
    a, payload_a = saltelli.design_identity(
        {"role": "A", "index": 0, "axis": None}, parameters()
    )
    ab, payload_ab = saltelli.design_identity(
        {"role": "AB", "index": 0, "axis": 0}, parameters()
    )
    assert a != ab
    assert payload_a["experiment"] == "independent_balanced_saltelli_v2"
    assert payload_a["result_schema_version"] == 4
    assert payload_a["scientific_metrics_version"] == 3
    assert payload_ab["saltelli_role"] == "AB"


def test_higher_iteration_limit_has_a_distinct_full_identity():
    completion.tso.IDENTITY = fake_identity()
    lower, _ = completion.limit_identity(parameters(), 0, 700_000)
    higher, payload = completion.limit_identity(parameters(), 0, 1_000_000)
    assert lower != higher
    assert payload["max_iterations"] == 1_000_000


def test_observed_614152_is_audited_but_not_near_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(completion, "OUT", tmp_path)
    completion.tso.MAX_ITERATIONS = 700_000
    trial = {
        **row("competitive", 0, 2),
        **parameters(),
        "orientation_correct_cells_at_0p02deg": 350,
        "candidate_pass_iterations": 614_152,
        "candidate_pass_saturated": False,
    }
    flagged, report = completion.saturation_audit([trial])
    assert flagged == []
    assert report["maximum"] == 614_152
    assert report["observed_maximum_fraction_of_limit"] < 0.9
    saved = json.loads((tmp_path / "candidate_iteration_distribution_audit.json").read_text())
    assert saved["approach_threshold_iterations"] == 630_000


def test_competitive_trial_at_90_percent_triggers_revalidation(tmp_path, monkeypatch):
    monkeypatch.setattr(completion, "OUT", tmp_path)
    completion.tso.MAX_ITERATIONS = 700_000
    trial = {
        **row("competitive", 0, 2),
        **parameters(),
        "orientation_correct_cells_at_0p02deg": 350,
        "candidate_pass_iterations": 630_000,
        "candidate_pass_saturated": False,
    }
    flagged, _ = completion.saturation_audit([trial])
    assert [item["config_hash"] for item in flagged] == [trial["config_hash"]]
