"""Original strict guard plus fail-closed verified terminal-PID handling.
Parent source SHA256 31b6e3566d8e910eef1d6419dbe9acdee2a6c8026af6020c5cd829b5e1a4bdcd
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import gpu_doctor
import run as frozen_run
import user_container as container
from gpu_utilization_window import MIN_UTILIZATION, UtilizationWindow


def process_table():
    table = {}
    ticks = os.sysconf("SC_CLK_TCK")
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            age = max(0.01, uptime - int(fields[19]) / ticks)
            cpu = 100.0 * (int(fields[11]) + int(fields[12])) / ticks / age
            rss = int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
            table[int(entry.name)] = {"ppid": int(fields[1]), "cpu": cpu, "rss": rss,
                                      "started_ticks": int(fields[19]), "state": fields[0]}
        except (OSError, ValueError, IndexError):
            continue
    return table


def descendants(root, table):
    result = {root}
    while True:
        added = {pid for pid, row in table.items() if row["ppid"] in result} - result
        if not added:
            return result
        result |= added


def classify_gpu_pids(root, selected, card_pids):
    # Query /proc AFTER the two NVIDIA snapshots. A prepared parent can fork
    # between those snapshots; an older process tree must not label its fresh
    # CUDA child as an unrelated occupant. Unknown/disappeared PIDs remain
    # foreign, and account ownership alone never proves job ownership.
    table = process_table()
    ours = descendants(root, table)
    observed = set(selected) | set(card_pids)
    return table, ours, sorted(observed & ours), sorted(observed - ours)


def stop_job(process):
    table = process_table()
    known = {pid: table[pid]["started_ticks"] for pid in descendants(process.pid, table) if pid in table}

    def alive():
        current = process_table()
        return {pid for pid, began in known.items() if pid in current
                and current[pid]["started_ticks"] == began and current[pid]["state"] != "Z"}

    for pid in alive():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    until = time.monotonic() + 3
    while alive() and time.monotonic() < until:
        time.sleep(0.1)
    for pid in alive():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=2)
    return known


def record_release(process, info, start, report, known=None):
    if known is None:
        known = stop_job(process) if process.poll() is None else {}
    # Include all account-owned compute PIDs on the selected card in the check,
    # so an orphaned worker cannot be missed after its parent exits.
    selected = []
    ours = []
    for attempt in range(3):
        rows = csv.reader(io.StringIO(gpu_doctor.query(["--query-compute-apps=gpu_uuid,pid"])))
        selected = [int(r[1]) for r in rows if len(r) >= 2 and r[0].strip() == info["uuid"]]
        table = process_table()
        ours = [pid for pid in selected if pid in table]
        if not ours:
            break
        if attempt < 2:
            time.sleep(0.2)
    result = dict(cleanup_seconds_after_start=round(time.monotonic()-start, 3),
                  own_gpu_released=not ours, remaining_own_gpu_pids=ours)
    with report.open("a") as log:
        log.write(json.dumps(result)+"\n")
    print("BASELINE_CLEANUP", json.dumps(result), flush=True)
    if ours:
        raise RuntimeError("GPU release verification failed: " + str(ours))


def finish_deferred_cleanup(token):
    from finish_scratch import finish
    root = container.PROJECT/'multi_event_gate_v1/outputs/frozen_event_eval/stack_bowls_two/demo_randomized/stacking2/Z_zero'
    completed = []
    for queue in root.glob('eval_*/sessions/process_*/storage_cleanup.json'):
        if queue.is_symlink():
            continue
        record = json.loads(queue.read_text())
        if record.get('launcher_token') == token:
            completed.append(dict(queue=str(queue), record=finish(queue)))
    print('BASELINE_CPU_SCRATCH_CLEANUP', json.dumps(completed), flush=True)
    return completed


def validate_memory_limit(limit_mib, total_mib):
    if type(limit_mib) is not int or not 4096 <= limit_mib < total_mib:
        raise ValueError('GPU memory limit must be an integer >=4096MiB and below device capacity')
    return limit_mib


def guarded_command(cmd, info, report, *, gpu, deadline=1800, on_sample=None, cleanup=None,
                    max_memory_mib=22528, min_cumulative_utilization=MIN_UTILIZATION,
                    steady_min_utilization=None, steady_window_seconds=60, prepared_process=None, before_monitor=None,
                    child_exit_proof=None, child_exit_identity=None):
    if not 33 <= min_cumulative_utilization <= 100:raise ValueError('Cumulative floor cannot be below server33%')
    if steady_min_utilization is not None and not 33 <= steady_min_utilization <= 100:raise ValueError('Invalid steady utilization floor')
    if steady_window_seconds not in (60,300):raise ValueError('Steady window must be60 or300 seconds')
    if (child_exit_proof is None)!=(child_exit_identity is None):
        raise ValueError('Both child-exit proof path and fresh attempt identity required')
    total_mib = int(gpu_doctor.query(['-i', str(gpu), '--query-gpu=memory.total']).strip())
    validate_memory_limit(max_memory_mib, total_mib)
    print('GPU_MEMORY_LIMIT', json.dumps(dict(limit_mib=max_memory_mib, total_mib=total_mib,
          headroom_mib=total_mib-max_memory_mib, allocation_oom_still_possible=True)), flush=True)
    start = time.monotonic()
    start_unix_us = time.time_ns()//1000
    from native_process_sampler_v8.device_window import DeviceWindow
    device_window = DeviceWindow(start_unix_us, info["uuid"])
    samples = []
    from native_fast_inventory_v1.inventory import inventory
    from gpu_policy import MAX_GPUS
    from native_process_sampler_v8.metrics import process_utilization,diagnostics
    from gpu_priority import ArrivalOrder
    arrival_order = ArrivalOrder()
    process_util_sums = {}
    verified_gpu_pids = set()
    verified_exit_identities = {}
    utilization_window = UtilizationWindow()
    recent_window = UtilizationWindow(60)
    process = prepared_process if prepared_process is not None else subprocess.Popen(cmd, start_new_session=True)
    reason = None
    known = None
    try:
        if before_monitor is not None:before_monitor()
        with report.open("x") as log:
            while process.poll() is None:
                elapsed = time.monotonic() - start
                table = process_table()
                ours = descendants(process.pid, table)
                row = next(csv.reader(io.StringIO(gpu_doctor.query([
                    "-i", str(gpu), "--query-gpu=utilization.gpu,utilization.memory,memory.used"
                ]))))
                sm, bandwidth, memory = [int(value.strip()) for value in row]
                samples.append(max(sm, bandwidth))
                apps = csv.reader(io.StringIO(gpu_doctor.query(["--query-compute-apps=gpu_uuid,pid,used_gpu_memory"])))
                app_rows = list(apps)
                per_process_memory = {int(r[1]): int(r[2].strip()) for r in app_rows if len(r)>=3 and r[0].strip()==info["uuid"] and r[2].strip().isdigit()}
                selected = [int(row[1]) for row in app_rows if len(row) >= 2 and row[0].strip() == info["uuid"]]
                fleet=inventory()
                card=next((c for c in fleet.get('cards',[]) if c.get('uuid')==info['uuid']),{})
                table,ours,active,foreign=classify_gpu_pids(process.pid,selected,card.get('pids',[]))
                verified_gpu_pids.update(active)
                from child_exit import verified_identity,retired
                pending_exit_identities = {
                    pid:verified_identity(pid,root_pid=process.pid,table=table)
                    for pid in active}
                from terminal import stale_after_verified_normal_exit
                terminal_stale = sorted(foreign) if stale_after_verified_normal_exit(process.poll(),foreign,verified_gpu_pids) else []
                pid_util=process_utilization(gpu)
                metric_samples=[diagnostics()]
                missing_initial=[pid for pid in active if pid not in pid_util]
                retry_records=[]
                # Confirm an unavailable pmon sample at most twice. A recovered
                # PID still contributes ZERO for this sample, never a favorable
                # replacement reading; sustained absence keeps the stop rule.
                unresolved=set(missing_initial)
                for _ in range(2 if unresolved else 0):
                    retry=process_utilization(gpu)
                    metric_samples.append(diagnostics())
                    retry_records.append({pid:retry.get(pid) for pid in missing_initial})
                    for pid in list(unresolved):
                        if pid in retry:
                            pid_util[pid]=0
                            unresolved.remove(pid)
                    if not unresolved:break
                # NVIDIA/proc queries can outlive the already finished root.
                # Never infer a new arrival conflict for a job that has exited.
                # Nonzero root exits still fail below; record_release independently
                # rejects any remaining account-owned GPU process, including orphans.
                terminal_code = process.poll()
                if terminal_code is not None:
                    log.write(json.dumps(dict(root_completed_during_telemetry=True,
                        root_exit_code=terminal_code,
                        verified_GPU_pids=sorted(verified_gpu_pids),
                        verified_child_identities=verified_exit_identities,
                        no_active_runtime_metric_exemption=True)) + "\n")
                    log.flush()
                    break
                child_exit_note = dict(accepted=False,reason='normal_child_exit_proof_not_configured')
                retired_children = set()
                if child_exit_proof is not None:
                    retired_children,child_exit_note=retired(
                        child_exit_proof,child_exit_identity,verified_exit_identities)
                terminal_stale=sorted(set(terminal_stale)|(set(foreign)&retired_children))
                active=[pid for pid in active if pid not in retired_children]
                priority=arrival_order.observe(
                    elapsed,active,sorted(set(foreign)-set(terminal_stale)))
                # Only a prior sample with a directly available per-PID metric
                # can authorize retirement on a later iteration. First-unseen,
                # recovered/missing or unknown PIDs never get this exemption.
                for pid in active:
                    candidate=pending_exit_identities.get(pid)
                    if candidate is not None and pid in pid_util and pid not in missing_initial:
                        previous=verified_exit_identities.get(pid)
                        if previous is not None and previous['child_started_ticks']!=candidate['child_started_ticks']:
                            raise ValueError('A verified CUDA PID was reused')
                        if previous is None:
                            verified_exit_identities[pid]=candidate
                for pid in active:
                    if pid in pid_util:process_util_sums[pid]=process_util_sums.get(pid,0)+pid_util[pid]
                # Include the full startup window; missing samples contribute 0.
                process_means={pid:process_util_sums.get(pid,0)/len(samples) for pid in active}
                elapsed = time.monotonic() - start
                cumulative, rolling = utilization_window.add(
                    elapsed, max(sm, bandwidth), {pid: pid_util.get(pid, 0) for pid in active})
                _, recent = recent_window.add(elapsed, max(sm, bandwidth), {pid:pid_util.get(pid,0) for pid in active})
                process_means = {pid: min(process_means[pid], cumulative.get(pid, 0)) for pid in active}
                legacy_device = dict(mean=sum(samples)/len(samples),cumulative=cumulative['device'],rolling300=rolling['device'],recent60=recent['device'])
                dense_device = device_window.update(metric_samples[-1]['raw_NVML'],time.time_ns()//1000,elapsed,steady_required=bool(active)and elapsed>=280)
                cumulative = dict(cumulative,device=dense_device['cumulative_lower_bound'])
                rolling = dict(rolling,device=dense_device['rolling300_lower_bound'])
                recent = dict(recent,device=dense_device['rolling60_lower_bound'])
                telemetry = {"seconds": round(elapsed, 2), "sm": sm, "bandwidth": bandwidth,
                             "device_continuous":dense_device,"device_legacy_sparse_diagnostic_only":legacy_device,
                             "memory_mib": memory, "max_memory_mib": max_memory_mib, "total_memory_mib": total_mib,
                             "mean_utilization": dense_device["sample_mean"],
                             "gpu_arrival_order": priority,
                             "terminal_absent_previously_verified_pids": terminal_stale,
                             "raw_foreign_pids": sorted(foreign),
                             "normal_child_exit_proof": child_exit_note,
                             "normal_child_retired_pids": sorted(retired_children),
                             "normal_child_previously_metric_verified_pids": sorted(verified_exit_identities),
                             "process_utilization": {pid:pid_util.get(pid) for pid in active},
                             "process_metric_samples": metric_samples,
                             "unavailable_components_counted_as_zero": True,
                             "process_utilization_initial_missing_pids": missing_initial,
                             "process_utilization_retries": retry_records,
                             "recovered_missing_samples_counted_as_zero": True,
                             "process_mean_utilization": process_means,
                             "time_weighted_mean_utilization": cumulative['device'],
                             "rolling_300s_utilization": rolling['device'],
                             "process_rolling_300s_utilization": {pid: rolling.get(pid, 0) for pid in active},
                             "minimum_mean_utilization": min_cumulative_utilization,
                             "steady_minimum_utilization": steady_min_utilization, "steady_check_after_seconds": 300,
                             "steady_window_seconds": steady_window_seconds,
                             "recent_60s_utilization": recent["device"],
                             "process_recent_60s_utilization": {pid:recent.get(pid,0) for pid in active},
                             "account_gpu_count": fleet["own_gpu_count"], "account_gpu_limit": MAX_GPUS, "available_gpu_indices": fleet["available_indices"],
                             "gpu_pids": active, "process_memory_mib": per_process_memory, "cpu": {pid: table[pid]["cpu"] for pid in active if pid in table},
                             "rss_bytes": {pid: table[pid]["rss"] for pid in active if pid in table}}
                log.write(json.dumps(telemetry) + "\n")
                log.flush()
                if fleet['own_gpu_count']>MAX_GPUS:
                    reason = "Account exceeded the all-hours three-GPU limit; stopping this job"
                if priority["stop_for_prior_or_unverified_occupant"]:
                    reason = "GPU occupied before our verified arrival or arrival order unverified; stopping only our job"
                if any(table[pid]["rss"] > 128 * 2**30 for pid in ours if pid in table):
                    reason = "A process exceeded the 128 GiB RSS limit"
                if memory > max_memory_mib:
                    reason = f"GPU device memory exceeded configured limit: {memory} > {max_memory_mib} MiB"
                # Check before five minutes, leaving time to release resources.
                # Never restart or generate filler work to evade these rules.
                if elapsed >= 280 and active:
                    if any(pid not in pid_util or process_means[pid] < min_cumulative_utilization
                           or rolling.get(pid, 0) < min_cumulative_utilization for pid in active):
                        reason = f"Per-process cumulative/rolling GPU utilization is unverified or below {min_cumulative_utilization}%"
                    if min(telemetry["mean_utilization"], cumulative['device'], rolling['device']) < min_cumulative_utilization or memory < 4096 or any(per_process_memory.get(pid,0)<4096 for pid in active):
                        reason = "Sustained GPU utilization/VRAM threshold would not be met"
                    if any(not 40 <= table[pid]["cpu"] <= 1600 for pid in active if pid in table):
                        reason = "Sustained CPU utilization threshold would not be met"
                if elapsed >= 300 and active and steady_min_utilization is not None:
                    selected_window = recent if steady_window_seconds==60 else rolling
                    if selected_window['device'] < steady_min_utilization or any(selected_window.get(pid,0) < steady_min_utilization for pid in active):
                        reason = f'After5min, recent{steady_window_seconds}s device/process utilization below {steady_min_utilization}%'
                if elapsed >= deadline:
                    reason = "Guarded evaluation reached its bounded time limit"
                if on_sample and not reason:
                    reason = on_sample(telemetry)
                if reason:
                    log.write(json.dumps({"stopped": reason}) + "\n")
                    log.flush()
                    break
                time.sleep(3)
        if reason:
            known = stop_job(process)
            raise RuntimeError(reason)
        code = process.wait()
        if code:
            raise SystemExit(code)
        print("BASELINE_PROCESS_COMPLETED", flush=True)
    finally:
        record_release(process, info, start, report, known)
        if cleanup:
            cleanup(reason, process.returncode)


