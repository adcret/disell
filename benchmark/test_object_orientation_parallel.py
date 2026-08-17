import hashlib
import json

import oracle_core as oc
import run_object_orientation_analysis as runner


def test_task_identity_binds_canonical_parameters_and_seed():
    config=oc.Config(.01,-1,.2,1.0,50,1.2)
    task=runner.make_task(config,"test",7)
    assert task["config_key"]==config.key()
    assert task["config_hash"]==hashlib.sha256(config.key().encode()).hexdigest()
    assert task["task_id"]==f"{task['config_hash']}:7"


def test_coordinator_append_is_one_complete_fsynced_json_line(tmp_path):
    path=tmp_path/"checkpoint.jsonl"
    row={"config_key":"a|b","random_seed":0,"status":"ok","value":1.25}
    runner.append(path,row)
    raw=path.read_bytes()
    assert raw.endswith(b"\n")
    assert json.loads(raw)==row


def test_comparison_ignores_only_nondeterministic_telemetry():
    row={"metric":1.0,"runtime_seconds":2.0,"available_ram_bytes_after":3,
         "swap_used_bytes_after":4,"config_key":"x"}
    assert runner.comparable(row)=={"metric":1.0,"config_key":"x"}


def test_safety_limits():
    safe={"aggregate_rss_bytes":1,"available_ram_bytes":20*1024**3,
          "available_ram_fraction":.8,"swap_growth_bytes":0}
    assert runner.unsafe_reason(safe) is None
    assert "10 GiB" in runner.unsafe_reason({**safe,"aggregate_rss_bytes":runner.HARD_TREE_RSS})
    assert "256 MiB" in runner.unsafe_reason({**safe,"swap_growth_bytes":runner.MAX_SWAP_GROWTH+1})
