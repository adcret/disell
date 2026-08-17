"""Crash-safe result envelopes and append-only storage for the two-stage oracle."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

# Version 3 is the shared-store default. Policy-specific writers may request a
# newer schema explicitly; this avoids relabelling unrelated benchmark rows.
SCHEMA_VERSION = 3
REQUIRED_IDENTITY = (
    "config_hash", "algorithm_id", "algorithm_source_hash", "extension_hash",
    "input_field_hash", "truth_hash", "seed",
)
REQUIRED_ENVELOPE = REQUIRED_IDENTITY + (
    "schema_version", "status", "status_category", "error_type",
    "error_message", "worker_pid", "worker_peak_rss_bytes", "started",
    "finished", "elapsed_seconds", "candidate_pass_saturated",
    "final_pass_saturated",
)
VALID_CATEGORIES = {
    "ok", "expected_algorithmic_invalid", "candidate_saturated",
    "final_saturated", "worker_exception", "coordinator/schema_error",
}


def _nonfinite_name(value):
    value = float(value)
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def json_safe(value, path="result"):
    """Recursively convert supported values and describe every unsafe value."""
    diagnostics = {"nonfinite_fields": [], "nonfinite_values": [],
                   "converted_fields": [], "unsupported_fields": []}

    def convert(item, current):
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, np.bool_):
            diagnostics["converted_fields"].append(current)
            return bool(item)
        if isinstance(item, np.integer):
            diagnostics["converted_fields"].append(current)
            return int(item)
        if isinstance(item, (float, np.floating)):
            number = float(item)
            if not math.isfinite(number):
                diagnostics["nonfinite_fields"].append(current)
                diagnostics["nonfinite_values"].append(_nonfinite_name(number))
                return None
            return number
        if isinstance(item, np.ndarray):
            diagnostics["converted_fields"].append(current)
            return [convert(child, f"{current}[{index}]")
                    for index, child in enumerate(item.tolist())]
        if isinstance(item, dict):
            output = {}
            for key, child in item.items():
                safe_key = str(key)
                if not isinstance(key, str):
                    diagnostics["converted_fields"].append(f"{current}.<key:{key!r}>")
                output[safe_key] = convert(child, f"{current}.{safe_key}")
            return output
        if isinstance(item, (list, tuple)):
            if isinstance(item, tuple):
                diagnostics["converted_fields"].append(current)
            return [convert(child, f"{current}[{index}]")
                    for index, child in enumerate(item)]
        if isinstance(item, (dt.datetime, dt.date, dt.time, Path, np.datetime64)):
            diagnostics["converted_fields"].append(current)
            return item.isoformat() if hasattr(item, "isoformat") else str(item)
        if isinstance(item, np.generic):
            diagnostics["converted_fields"].append(current)
            return convert(item.item(), current)
        diagnostics["unsupported_fields"].append(current)
        return f"<{type(item).__module__}.{type(item).__qualname__}>"

    return convert(value, path), diagnostics


def minimal_schema_error(raw, identity=None, *, error_type="SchemaError",
                         error_message="result schema validation failed",
                         diagnostics=None, config_hash=None,
                         schema_version=SCHEMA_VERSION):
    """Build a JSON-safe error row without copying untrusted result content."""
    safe_raw, found = json_safe(dict(raw or {}))
    safe_identity, identity_found = json_safe(dict(identity or {}), "identity")
    diagnostics = dict(diagnostics or found)
    for key in ("nonfinite_fields", "nonfinite_values", "converted_fields",
                "unsupported_fields"):
        diagnostics.setdefault(key, [])
        diagnostics[key] = list(diagnostics[key]) + list(identity_found.get(key, []))
    get = safe_raw.get
    digest = config_hash or get("config_hash")
    envelope = {
        "schema_version": int(schema_version),
        "config_hash": digest,
        "config_key": get("config_key"),
        "configuration_identity": safe_identity,
        "algorithm_id": get("algorithm_id") or get("algorithm") or safe_identity.get("algorithm"),
        "algorithm_source_hash": get("algorithm_source_hash") or safe_identity.get("python_source_sha256"),
        "extension_hash": get("extension_hash") or safe_identity.get("compiled_extension_sha256"),
        "input_field_hash": get("input_field_hash") or safe_identity.get("latent_sha256"),
        "truth_hash": get("truth_hash") or safe_identity.get("truth_sha256"),
        "stage": get("stage"),
        "seed": get("seed", get("candidate_order_seed")),
        "candidate_order_seed": get("candidate_order_seed", get("seed")),
        "status": "error",
        "status_category": "coordinator/schema_error",
        "error_type": str(error_type),
        "error_message": str(error_message)[:4096],
        "nonfinite_fields": diagnostics["nonfinite_fields"],
        "nonfinite_values": diagnostics["nonfinite_values"],
        "converted_fields": diagnostics["converted_fields"],
        "unsupported_fields": diagnostics["unsupported_fields"],
        "worker_pid": get("worker_pid"),
        "worker_peak_rss_bytes": get("worker_peak_rss_bytes"),
        "started": get("started"),
        "finished": get("finished"),
        "elapsed_seconds": get("elapsed_seconds", get("runtime_seconds")),
        "candidate_pass_saturated": bool(get("candidate_pass_saturated", False)),
        "final_pass_saturated": bool(get("final_pass_saturated", False)),
        "scientific_metrics_version": get("scientific_metrics_version", safe_identity.get("matching_metrics_version")),
        "selection_policy_id": get("selection_policy_id", safe_identity.get("selection_policy_id")),
        "schema_warnings": [str(error_message)[:4096]],
    }
    safe, final_diagnostics = json_safe(envelope)
    if final_diagnostics["nonfinite_fields"] or final_diagnostics["unsupported_fields"]:
        # Telemetry is nullable in the error envelope. Preserve the original
        # paths above, then null only those untrusted optional envelope fields.
        for key in ("worker_pid", "worker_peak_rss_bytes", "started", "finished",
                    "elapsed_seconds"):
            safe[key] = None
    return safe


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _error_parts(row):
    kind = row.get("error_type")
    message = row.get("error_message") or row.get("error")
    if message and not kind and ":" in str(message):
        kind, message = str(message).split(":", 1)
        message = message.strip()
    return kind, message


def normalize_result(raw, identity=None, *, rehash_identity=False,
                     schema_version=SCHEMA_VERSION):
    """Return one complete envelope; malformed identity becomes schema_error."""
    row, diagnostics = json_safe(dict(raw or {}))
    embedded = dict(row.get("configuration_identity") or {})
    safe_identity, identity_diagnostics = json_safe(dict(identity or {}), "identity")
    identity = {**safe_identity, **embedded}
    for key in diagnostics:
        diagnostics[key].extend(identity_diagnostics[key])
    if diagnostics["nonfinite_fields"] or diagnostics["unsupported_fields"]:
        issue = "non-finite result values" if diagnostics["nonfinite_fields"] else "unsupported result values"
        return minimal_schema_error(
            row, identity, error_type="NonFiniteResult" if diagnostics["nonfinite_fields"] else "UnsupportedResultType",
            error_message=f"{issue}: " + ", ".join(diagnostics["nonfinite_fields"] or diagnostics["unsupported_fields"]),
            diagnostics=diagnostics, schema_version=schema_version,
        )
    row["configuration_identity"] = identity
    if rehash_identity and identity:
        old_hash = row.get("config_hash")
        text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        row["config_hash"] = hashlib.sha256(text.encode()).hexdigest()
        if old_hash and old_hash != row["config_hash"]:
            row["legacy_config_hash"] = old_hash
    warnings = list(row.get("schema_warnings") or [])
    if diagnostics["converted_fields"]:
        warnings.append("JSON-normalised fields: " + ", ".join(diagnostics["converted_fields"]))
    aliases = {
        "algorithm_id": row.get("algorithm_id") or row.get("algorithm") or identity.get("algorithm"),
        "algorithm_source_hash": row.get("algorithm_source_hash") or identity.get("python_source_sha256"),
        "extension_hash": row.get("extension_hash") or identity.get("compiled_extension_sha256"),
        "input_field_hash": row.get("input_field_hash") or identity.get("latent_sha256"),
        "truth_hash": row.get("truth_hash") or identity.get("truth_sha256"),
        "seed": row.get("seed", row.get("candidate_order_seed")),
    }
    row.update({k: v for k, v in aliases.items() if v is not None})
    row["schema_version"] = int(schema_version)
    kind, message = _error_parts(row)
    row["error_type"], row["error_message"] = kind, message
    candidate = bool(row.get("candidate_pass_saturated", False))
    final = bool(row.get("final_pass_saturated", False))
    row["candidate_pass_saturated"], row["final_pass_saturated"] = candidate, final

    missing_identity = [k for k in REQUIRED_IDENTITY if row.get(k) is None]
    if missing_identity:
        row["status"] = "error"
        row["status_category"] = "coordinator/schema_error"
        row["error_type"] = "SchemaError"
        row["error_message"] = "missing required identity: " + ", ".join(missing_identity)
        warnings.append(row["error_message"])
    elif candidate:
        row["status"], row["status_category"] = "incomplete", "candidate_saturated"
    elif final:
        row["status"], row["status_category"] = "incomplete", "final_saturated"
    else:
        error_text = " ".join(str(x or "") for x in (kind, message, row.get("error"))).lower()
        if "no valid seeds" in error_text or "no accepted markers" in error_text or "produced no accepted markers" in error_text:
            row["status"], row["status_category"] = "invalid", "expected_algorithmic_invalid"
        elif row.get("status") == "ok" and not kind and not message:
            row["status"], row["status_category"] = "ok", "ok"
        elif row.get("status_category") not in VALID_CATEGORIES:
            row["status"], row["status_category"] = "error", "worker_exception"

    optional_defaults = {
        "worker_pid": None, "worker_peak_rss_bytes": None, "started": None,
        "finished": None, "elapsed_seconds": row.get("runtime_seconds"),
    }
    for key, value in optional_defaults.items():
        if key not in row:
            row[key] = value
            warnings.append(f"legacy/worker result omitted {key}")
    if row["worker_peak_rss_bytes"] is not None:
        try: row["worker_peak_rss_bytes"] = int(row["worker_peak_rss_bytes"])
        except (TypeError, ValueError):
            warnings.append("invalid worker_peak_rss_bytes")
            row["worker_peak_rss_bytes"] = None
    row["schema_warnings"] = sorted(set(warnings))
    # A complete envelope always has every required key, even where legacy
    # telemetry is explicitly unavailable.
    for key in REQUIRED_ENVELOPE:
        row.setdefault(key, None)
    return row


def validate_result(row):
    missing = [key for key in REQUIRED_ENVELOPE if key not in row]
    if missing:
        raise ValueError("missing envelope fields: " + ", ".join(missing))
    if row["status_category"] not in VALID_CATEGORIES:
        raise ValueError(f"invalid status_category {row['status_category']!r}")
    if row["status_category"] != "coordinator/schema_error":
        missing_identity = [k for k in REQUIRED_IDENTITY if row.get(k) is None]
        if missing_identity:
            raise ValueError("missing identity: " + ", ".join(missing_identity))
    return row


class ResultStore:
    """Single-writer fsynced JSONL whose in-memory state is rebuilt from disk."""
    def __init__(self, path, *, repair_partial=True, default_identity=None,
                 rehash_identity=False, schema_version=SCHEMA_VERSION):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.hashes = set()
        self.duplicates_on_disk = []
        self.quarantined = []
        self.default_identity = dict(default_identity or {})
        self.rehash_identity = bool(rehash_identity)
        self.schema_version = int(schema_version)
        self._load(repair_partial)

    def _load(self, repair):
        if not self.path.exists():
            return
        valid_raw = []
        malformed = []
        with self.path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                try:
                    valid_raw.append((line_number, json.loads(
                        raw, parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f"non-standard JSON constant {value}")))))
                except Exception as exc: malformed.append((line_number, raw, str(exc)))
        if malformed:
            stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
            preserved = self.path.with_name(f"{self.path.name}.malformed-{stamp}")
            quarantine = self.path.with_name(f"{self.path.name}.quarantine-{stamp}.jsonl")
            shutil.copy2(self.path, preserved)
            with quarantine.open("wb") as out:
                for line, raw, error in malformed:
                    out.write(json.dumps({"line": line, "error": error, "raw": raw.decode(errors="replace")}).encode() + b"\n")
                out.flush(); os.fsync(out.fileno())
            self.quarantined = [str(quarantine), str(preserved)]
            if repair:
                temporary = self.path.with_suffix(self.path.suffix + ".repair")
                with temporary.open("w") as out:
                    for _, raw in valid_raw: out.write(json.dumps(raw, sort_keys=True) + "\n")
                    out.flush(); os.fsync(out.fileno())
                temporary.replace(self.path)
        for line, raw in valid_raw:
            identity = {**self.default_identity, **dict(raw.get("configuration_identity") or {})}
            row = normalize_result(raw, identity, rehash_identity=self.rehash_identity,
                                   schema_version=self.schema_version)
            digest = row.get("config_hash")
            if digest in self.hashes:
                self.duplicates_on_disk.append({"line": line, "config_hash": digest})
                continue
            self.rows.append(row)
            if digest is not None: self.hashes.add(digest)

    def _emergency(self, message):
        path = self.path.with_name(self.path.name + ".coordinator_errors.log")
        bounded = str(message).replace("\x00", "?")[:8192]
        try:
            with path.open("a") as stream:
                stream.write(f"{utc_now()} {bounded}\n")
                stream.flush(); os.fsync(stream.fileno())
        except Exception:
            # Persistence of the scientific JSONL remains protected even when
            # the separate diagnostic sink is unavailable.
            pass

    def append(self, raw, identity=None, *, crash_after_append=False):
        """Persist one row without allowing malformed worker data to escape."""
        candidate = raw
        try:
            row = validate_result(normalize_result(
                candidate, identity, schema_version=self.schema_version))
            payload = json.dumps(row, sort_keys=True, allow_nan=False)
        except Exception as exc:
            try:
                row = validate_result(minimal_schema_error(
                    candidate, identity, error_type=type(exc).__name__,
                    error_message=f"result normalisation/validation failed: {exc}",
                    schema_version=self.schema_version,
                ))
                payload = json.dumps(row, sort_keys=True, allow_nan=False)
            except Exception as emergency_exc:
                digest = candidate.get("config_hash") if isinstance(candidate, dict) else None
                self._emergency(
                    f"unpersisted trial config_hash={digest!r}; "
                    f"primary={type(exc).__name__}: {exc}; "
                    f"minimal={type(emergency_exc).__name__}: {emergency_exc}"
                )
                return False, {
                    "schema_version": self.schema_version, "config_hash": digest,
                    "status": "error", "status_category": "coordinator/schema_error",
                    "error_type": "PersistenceFailure",
                    "error_message": "minimal schema-error persistence failed; retry required",
                }
        digest = row.get("config_hash")
        if digest in self.hashes:
            return False, row
        descriptor = None
        offset = self.path.stat().st_size if self.path.exists() else 0
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
            encoded = (payload + "\n").encode("utf-8")
            written = 0
            while written < len(encoded):
                count = os.write(descriptor, encoded[written:])
                if count <= 0:
                    raise OSError("zero-byte JSONL append")
                written += count
            os.fsync(descriptor)
        except Exception as exc:
            if descriptor is not None:
                try:
                    os.ftruncate(descriptor, offset); os.fsync(descriptor)
                except Exception as rollback_exc:
                    self._emergency(
                        f"append rollback failed config_hash={digest!r}; "
                        f"{type(rollback_exc).__name__}: {rollback_exc}"
                    )
            self._emergency(
                f"append failed before durable bookkeeping config_hash={digest!r}; "
                f"{type(exc).__name__}: {exc}"
            )
            return False, row
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if crash_after_append:
            raise RuntimeError("simulated crash after append")
        self.rows.append(row)
        if digest is not None: self.hashes.add(digest)
        return True, row

    def counts(self):
        out = {category: 0 for category in VALID_CATEGORIES}
        for row in self.rows: out[row["status_category"]] += 1
        return out


def migrate_legacy_file(path, default_identity):
    """Preserve v1 verbatim, then atomically install normalized v2 records."""
    path = Path(path)
    raw_rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip(): continue
            raw_rows.append((line_number, json.loads(line)))
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    preserved = path.with_name(f"{path.name}.schema-v1-{stamp}")
    shutil.copy2(path, preserved)
    normalized = []
    seen = set()
    for line_number, raw in raw_rows:
        identity = {**dict(default_identity), **dict(raw.get("configuration_identity") or {})}
        # The truth identity was missing in schema v1; add it before deriving
        # the v2 full hash. The old hash remains recoverable in every row.
        if default_identity.get("truth_sha256"):
            identity["truth_sha256"] = default_identity["truth_sha256"]
        # Preserve the scientific meaning of completed v1 scalar metrics.
        # They remain auditable, but their hashes cannot collide with the v2
        # Euclidean two-channel matching definition.
        error_text = str(raw.get("error") or raw.get("error_message") or "").lower()
        if "no valid seeds" in error_text or "no accepted markers" in error_text:
            # No scientific matching metrics exist for an empty-marker trial;
            # it is fully reusable under v2 and need not be rerun.
            identity["matching_metrics_version"] = 2
            if default_identity.get("matching_source_sha256"):
                identity["matching_source_sha256"] = default_identity["matching_source_sha256"]
        else:
            identity.setdefault("matching_metrics_version", 1)
        raw.setdefault("scientific_metrics_version", 1)
        row = normalize_result(raw, identity, rehash_identity=True)
        row["migrated_from_line"] = line_number
        digest = row.get("config_hash")
        if digest in seen:
            row["status"] = "error"
            row["status_category"] = "coordinator/schema_error"
            row["error_type"] = "DuplicateHash"
            row["error_message"] = f"duplicate full hash during migration: {digest}"
        else:
            seen.add(digest)
        normalized.append(row)
    temporary = path.with_suffix(path.suffix + ".schema-v2")
    with temporary.open("w") as stream:
        for row in normalized:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return {"preserved": str(preserved), "rows": len(normalized),
            "unique_hashes": len(seen)}
