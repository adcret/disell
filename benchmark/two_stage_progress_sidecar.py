#!/usr/bin/env python3
"""Read-only live scientific accounting sidecar for the active coordinator."""
from __future__ import annotations
import argparse,fcntl,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent; OUT=HERE/'two_stage_oracle_results'; sys.path.insert(0,str(HERE))
from audit_two_stage_live import atomic
from two_stage_accounting import accounting

def coordinator_alive(pid):
    path=Path(f'/proc/{pid}/cmdline')
    try: command=path.read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except OSError: return False
    return 'benchmark/two_stage_oracle.py' in command

def snapshot():
    source=OUT/'evaluations.jsonl'; rows=[]; malformed=[]
    with source.open('rb') as stream:
        for line_number,line in enumerate(stream,1):
            try:
                if not line.endswith(b'\n'): raise ValueError('partial line')
                rows.append(json.loads(line))
            except Exception as exc: malformed.append({'line':line_number,'error':str(exc)})
    legacy=json.loads((OUT/'progress.json').read_text()) if (OUT/'progress.json').exists() else {}
    return {**accounting(rows),'malformed_records':malformed,'coordinator_stage':legacy.get('stage'),'coordinator_pending':legacy.get('pending_count'),'coordinator_in_flight':legacy.get('in_flight_count'),'worker_count':legacy.get('worker_count'),'trials_per_second':legacy.get('trials_per_second'),'eta_seconds':legacy.get('eta_seconds'),'aggregate_process_tree_rss_bytes':legacy.get('aggregate_process_tree_rss_bytes'),'available_ram_bytes':legacy.get('available_ram_bytes'),'swap_growth_bytes':legacy.get('swap_growth_bytes'),'updated_epoch':time.time()}

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--interval',type=float,default=20.); parser.add_argument('--once',action='store_true'); args=parser.parse_args(); lock=(OUT/'scientific_progress.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('scientific progress sidecar already running')
    pid_meta=json.loads((OUT/'pid.json').read_text()); atomic(OUT/'scientific_progress.pid.json',{'pid':os.getpid(),'coordinator_pid':pid_meta['pid'],'started_epoch':time.time(),'interval_seconds':args.interval})
    while True:
        value=snapshot(); atomic(OUT/'scientific_progress.json',value)
        if args.once or not coordinator_alive(int(pid_meta['pid'])): break
        time.sleep(max(args.interval,1.))
if __name__=='__main__': main()
