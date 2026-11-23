"""Bounded one-job enforcement of current governor rules; only this job is stopped."""
import json
import os
import signal
import sys
import time
from pathlib import Path

from policy import L, evaluate, verify_rules
from observer import latest
sys.path.insert(0, str(L / 'official_baseline_v14'))
from common import append
from dense_guard import process_table, descendants, stop_job


def release(process, identities, info, start, report):
    table = process_table()
    identities.update((pid, table[pid]['started_ticks'])
                      for pid in descendants(process.pid, table) if pid in table)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, birth in identities:
            table = process_table()
            if pid in table and table[pid]['started_ticks'] == birth and table[pid]['state'] != 'Z':
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        until = time.monotonic() + (3 if sig == signal.SIGTERM else 1)
        while time.monotonic() < until:
            table = process_table()
            if not any(pid in table and table[pid]['started_ticks'] == birth and table[pid]['state'] != 'Z'
                       for pid, birth in identities):
                break
            time.sleep(.1)
    process.wait(timeout=3)
    # Fresh XML avoids waiting for the observer's next sample at cleanup.
    from gpu_policy import inventory
    fleet = inventory(); table = process_table()
    remaining = [pid for c in fleet['cards'] for pid in c['pids']
                 if pid in table and (pid, table[pid]['started_ticks']) in identities]
    append(report, dict(cleanup_seconds_after_start=round(time.monotonic() - start, 3),
                        own_gpu_released=not remaining, remaining_own_gpu_pids=remaining,
                        other_jobs_never_signalled=True))
    assert not remaining, remaining


def guarded_command(cmd, info, report, *, gpu, deadline=21600, on_sample=None,
                    max_memory_mib=24563, prepared_process=None, **retired):
    """Old cumulative33/extra40 stops are retired by the user's explicit choice."""
    assert prepared_process is not None
    verify_rules(); process = prepared_process; start = time.monotonic()
    owned = set(); last = None; reason = None
    try:
        while process.poll() is None:
            verify_rules()
            fleet = latest()
            if fleet['monotonic'] == last:
                time.sleep(.2); continue
            last = fleet['monotonic']; table = process_table()
            tree = descendants(process.pid, table)
            owned.update((pid, table[pid]['started_ticks']) for pid in tree if pid in table)
            card = next(c for c in fleet['cards'] if c['uuid'] == info['uuid'])
            job = [p for c in fleet['cards'] for p in c['processes'] if (p['pid'], p['birth']) in owned]
            verdict = evaluate(fleet, owned)
            d = card['device_rolling180']; recent = card['device_recent60']
            sample = dict(schema='governor180_v15', seconds=round(time.monotonic()-start, 2),
                          wall_time=fleet['wall_time'], observer_monotonic=fleet['monotonic'],
                          server_window_seconds=180, means_before_MAX=True,
                          thresholds=verdict['limits'], server_verdict=verdict,
                          fleet=fleet, GPU_uuid=info['uuid'], gpu_pids=[p['pid'] for p in job],
                          process_identities=[dict(pid=p['pid'], birth=p['birth'], age=p['age']) for p in job],
                          process_rolling180={str(p['pid']):p['rolling180'] for p in job},
                          sm=card['sm'], bandwidth=card['bandwidth'], memory_mib=card['memory_mib'],
                          total_memory_mib=card['total_mib'], max_memory_mib=max_memory_mib,
                          # Compatibility inputs for frozen host's RESOURCE-only callback.
                          time_weighted_mean_utilization=d['util'], rolling_300s_utilization=d['util'],
                          mean_utilization=d['util'], process_mean_utilization={str(p['pid']):p['rolling180']['util'] for p in job},
                          recent_60s_utilization=recent['util'],
                          process_recent_60s_utilization={str(p['pid']):p['recent60']['util'] for p in job},
                          cpu={str(p['pid']):p['rolling180']['cpu'] for p in job},
                          rss_bytes={str(p['pid']):p['rss'] for p in job},
                          process_memory_mib={str(p['pid']):p['memory_mib'] for p in job},
                          account_gpu_count=fleet['own_gpu_count'], account_gpu_limit=3,
                          available_gpu_indices=fleet['available_indices'],
                          recent60_optimization_target=40, project_extra_session_limit_seconds=deadline)
            if verdict['reasons']:
                reason = '; '.join(verdict['reasons'])
            if card['memory_mib'] > max_memory_mib:
                reason = 'Project24563MiB device memory ceiling'
            if time.monotonic() - start >= min(deadline, 36 * 3600):
                reason = 'Project session time limit'
            if time.monotonic() - start > 60 and not job:
                reason = 'GPU process missing from verified job tree'
            if on_sample and not reason:
                reason = on_sample(sample)
            append(report, sample)
            if reason:
                append(report, dict(stopped=reason)); raise RuntimeError(reason)
            time.sleep(.2)
        code = process.wait()
        if code:
            raise RuntimeError('GPU wrapper exited with code ' + str(code))
    finally:
        release(process, owned, info, start, report)
