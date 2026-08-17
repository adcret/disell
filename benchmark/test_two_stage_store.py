import json
import datetime as dt
from pathlib import Path

import numpy as np
import pytest

import two_stage_store as store_module
from two_stage_store import ResultStore, json_safe, normalize_result


IDENTITY = {
    "algorithm": "size_prioritised_multiseed_v1",
    "python_source_sha256": "source",
    "compiled_extension_sha256": "extension",
    "latent_sha256": "latent",
    "truth_sha256": "truth",
}


def base(**updates):
    row = {
        "config_hash": "hash", "configuration_identity": IDENTITY,
        "candidate_order_seed": 3, "status": "ok", "worker_pid": 10,
        "worker_peak_rss_bytes": 123, "started": "a", "finished": "b",
        "elapsed_seconds": 1.0, "candidate_pass_saturated": False,
        "final_pass_saturated": False,
    }
    row.update(updates)
    return row


def test_successful_trial():
    row = normalize_result(base())
    assert row["status_category"] == "ok"


@pytest.mark.parametrize("message", ["No valid seeds found", "Two-stage flood fill produced no accepted markers"])
def test_expected_algorithmic_invalid_paths(message):
    row = normalize_result(base(status="error", error=f"RuntimeError: {message}"))
    assert row["status_category"] == "expected_algorithmic_invalid"


def test_candidate_and_final_saturation_are_distinct():
    assert normalize_result(base(candidate_pass_saturated=True))["status_category"] == "candidate_saturated"
    assert normalize_result(base(final_pass_saturated=True))["status_category"] == "final_saturated"


def test_worker_exception_is_infrastructure_failure():
    row = normalize_result(base(status="error", error_type="ValueError", error_message="boom"))
    assert row["status_category"] == "worker_exception"


def test_missing_optional_diagnostic_warns_but_remains_valid():
    row = base(); del row["worker_peak_rss_bytes"]
    normalized = normalize_result(row)
    assert normalized["status_category"] == "ok"
    assert normalized["worker_peak_rss_bytes"] is None
    assert any("worker_peak_rss_bytes" in x for x in normalized["schema_warnings"])


def test_missing_required_identity_is_schema_error():
    row = base(); del row["config_hash"]
    normalized = normalize_result(row)
    assert normalized["status_category"] == "coordinator/schema_error"


def test_nonfinite_metric_is_persistable_but_not_scientifically_selectable(tmp_path):
    normalized = normalize_result(base(ari=float("nan")))
    assert normalized["status_category"] == "coordinator/schema_error"
    assert normalized["error_type"] == "NonFiniteResult"
    assert normalized["nonfinite_fields"] == ["result.ari"]
    assert normalized["nonfinite_values"] == ["NaN"]
    assert "ari" not in normalized
    path = tmp_path / "rows.jsonl"
    assert ResultStore(path).append(normalized)[0]
    assert json.loads(path.read_text())["status_category"] == "coordinator/schema_error"


@pytest.mark.parametrize(("value", "name"), [
    (float("nan"), "NaN"),
    (float("inf"), "Infinity"),
    (float("-inf"), "-Infinity"),
    (np.float32("nan"), "NaN"),
    (np.float64("inf"), "Infinity"),
])
def test_top_level_python_and_numpy_nonfinite_values(value, name):
    safe, diagnostics = json_safe(value, "metric")
    assert safe is None
    assert diagnostics["nonfinite_fields"] == ["metric"]
    assert diagnostics["nonfinite_values"] == [name]


def test_nested_dict_list_and_array_nonfinite_paths():
    value = {
        "scientific_metrics": {"boundary_f1": float("nan")},
        "diagnostics": [1, float("inf")],
        "array": np.asarray([[1, -np.inf]], dtype=np.float32),
    }
    safe, diagnostics = json_safe(value)
    assert diagnostics["nonfinite_fields"] == [
        "result.scientific_metrics.boundary_f1",
        "result.diagnostics[1]",
        "result.array[0][1]",
    ]
    assert diagnostics["nonfinite_values"] == ["NaN", "Infinity", "-Infinity"]
    json.dumps(safe, allow_nan=False)


def test_numpy_scalars_arrays_and_timestamp_metadata_are_json_safe():
    safe, diagnostics = json_safe({
        "integer": np.int64(7), "boolean": np.bool_(True),
        "array": np.asarray([1, 2], dtype=np.int32),
        "timestamp": dt.datetime(2026, 8, 13, tzinfo=dt.timezone.utc),
    })
    assert safe == {
        "integer": 7, "boolean": True, "array": [1, 2],
        "timestamp": "2026-08-13T00:00:00+00:00",
    }
    assert not diagnostics["nonfinite_fields"]
    json.dumps(safe, allow_nan=False)


def test_expected_invalid_with_nonfinite_diagnostic_becomes_schema_error():
    row = normalize_result(base(
        config_hash="invalid-nan", status="error",
        error="RuntimeError: No valid seeds found", diagnostics={"ratio": np.nan},
    ))
    assert row["status_category"] == "coordinator/schema_error"
    assert row["error_type"] == "NonFiniteResult"
    assert row["nonfinite_fields"] == ["result.diagnostics.ratio"]


def test_worker_exception_with_nonfinite_telemetry_becomes_schema_error():
    row = normalize_result(base(
        config_hash="exception-nan", status="error", error_type="ValueError",
        error_message="boom", worker_peak_rss_bytes=np.inf,
    ))
    assert row["status_category"] == "coordinator/schema_error"
    assert row["worker_peak_rss_bytes"] is None
    assert row["nonfinite_values"] == ["Infinity"]


def test_failure_constructing_minimal_schema_error_is_contained(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"; store = ResultStore(path)
    monkeypatch.setattr(store_module, "normalize_result", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("normalise")))
    monkeypatch.setattr(store_module, "minimal_schema_error", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("minimal")))
    added, row = store.append(base())
    assert not added and row["error_type"] == "PersistenceFailure"
    assert not path.exists() or not path.read_bytes()
    assert path.with_name(path.name + ".coordinator_errors.log").exists()


def test_failure_during_strict_serialisation_persists_minimal_error(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"; store = ResultStore(path)
    real_dumps = store_module.json.dumps; calls = {"count": 0}
    def fail_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TypeError("forced strict serialisation failure")
        return real_dumps(*args, **kwargs)
    monkeypatch.setattr(store_module.json, "dumps", fail_once)
    added, row = store.append(base(config_hash="serialization"))
    assert added
    assert row["status_category"] == "coordinator/schema_error"
    assert json.loads(path.read_text())["error_type"] == "TypeError"


def test_failure_during_minimal_strict_serialisation_is_nonfatal(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"; store = ResultStore(path)
    monkeypatch.setattr(store_module.json, "dumps", lambda *_a, **_k: (_ for _ in ()).throw(TypeError("always")))
    added, row = store.append(base(config_hash="unpersisted"))
    assert not added and row["error_type"] == "PersistenceFailure"
    assert not path.exists() or not path.read_bytes()


def test_resume_after_normalise_before_append(tmp_path):
    path = tmp_path / "rows.jsonl"
    normalized = normalize_result(base(config_hash="normalised"))
    resumed = ResultStore(path)
    assert resumed.append(normalized)[0]
    assert not ResultStore(path).append(normalized)[0]


def test_minimal_safe_error_persistence_and_coordinator_continuation(tmp_path):
    path = tmp_path / "rows.jsonl"; store = ResultStore(path)
    bad = base(config_hash="bad", scientific={"boundary_f1": np.nan})
    good = base(config_hash="good")
    assert store.append(bad)[0]
    assert store.append(good)[0]
    loaded = [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
              for line in path.read_text().splitlines()]
    assert [row["status_category"] for row in loaded] == [
        "coordinator/schema_error", "ok",
    ]
    assert ResultStore(path).counts()["coordinator/schema_error"] == 1


def test_crash_after_append_recovers_from_disk_without_duplicate(tmp_path):
    path = tmp_path / "rows.jsonl"; store = ResultStore(path)
    with pytest.raises(RuntimeError, match="simulated"):
        store.append(base(), crash_after_append=True)
    resumed = ResultStore(path)
    assert len(resumed.rows) == 1
    added, _ = resumed.append(base())
    assert not added and len(path.read_text().splitlines()) == 1


def test_partial_final_line_is_preserved_and_quarantined(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(base()) + "\n" + '{"config_hash":')
    store = ResultStore(path)
    assert len(store.rows) == 1 and len(store.quarantined) == 2
    assert len(path.read_text().splitlines()) == 1
    assert all(Path(x).exists() for x in store.quarantined)


def test_duplicate_full_hash_persists_exactly_once(tmp_path):
    path = tmp_path / "rows.jsonl"; store = ResultStore(path)
    assert store.append(base())[0]
    assert not store.append(base())[0]
    assert len(path.read_text().splitlines()) == 1


def test_invalid_result_does_not_poison_following_recycled_work(tmp_path):
    path = tmp_path / "rows.jsonl"; store = ResultStore(path)
    invalid = base(config_hash="invalid", status="error", error="RuntimeError: No valid seeds found")
    valid = base(config_hash="next")
    store.append(invalid); store.append(valid)
    assert store.counts()["expected_algorithmic_invalid"] == 1
    assert store.counts()["ok"] == 1
    assert len(ResultStore(path).rows) == 2
