#!/usr/bin/env python3
"""Migrate the stopped v1 two-stage store and write a forensic audit."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from two_stage_store import ResultStore, migrate_legacy_file

HERE = Path(__file__).resolve().parent
OUT = HERE / "two_stage_oracle_results"


def main():
    audit = json.loads((OUT / "implementation_audit.json").read_text())
    identity = {
        "algorithm": "size_prioritised_multiseed_v1",
        "python_source_sha256": audit["python_source_sha256"],
        "compiled_extension_sha256": audit["compiled_extension_sha256"],
        "latent_sha256": audit["latent_sha256"],
        "truth_sha256": audit["truth_sha256"],
        "mask_sha256": audit["mask_sha256"],
        "matching_source_sha256": audit["matching_source_sha256"],
    }
    result = migrate_legacy_file(OUT / "evaluations.jsonl", identity)
    store = ResultStore(OUT / "evaluations.jsonl")
    categories = store.counts()
    report = {
        **result,
        "schema_version": 2,
        "valid": categories["ok"],
        "expected_algorithmic_invalid": categories["expected_algorithmic_invalid"],
        "candidate_saturated": categories["candidate_saturated"],
        "final_saturated": categories["final_saturated"],
        "worker_exception": categories["worker_exception"],
        "coordinator_schema_error": categories["coordinator/schema_error"],
        "duplicates": len(store.duplicates_on_disk),
        "quarantined": store.quarantined,
        "jsonl_sha256": hashlib.sha256((OUT / "evaluations.jsonl").read_bytes()).hexdigest(),
    }
    (OUT / "schema_migration_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    with (OUT / "completed_hashes.txt").open("w") as stream:
        for digest in sorted(store.hashes): stream.write(digest + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
