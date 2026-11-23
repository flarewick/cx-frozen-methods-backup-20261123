"""Run exactly one baseline acceptance episode under the server resource rules."""
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
                    steady_min_utilization=None, steady_window_seconds=60, prepared_process=None, before_monitor=None):
    if not 33 <= min_cumulative_utilization <= 100:raise ValueError('Cumulative floor cannot be below server33%')
    if steady_min_utilization is not None and not 33 <= steady_min_utilization <= 100:raise ValueError('Invalid steady utilization floor')
    if steady_window_seconds not in (60,300):raise ValueError('Steady window must be60 or300 seconds')
    total_mib = int(gpu_doctor.query(['-i', str(gpu), '--query-gpu=memory.total']).strip())
    validate_memory_limit(max_memory_mib, total_mib)
    print('GPU_MEMORY_LIMIT', json.dumps(dict(limit_mib=max_memory_mib, total_mib=total_mib,
          headroom_mib=total_mib-max_memory_mib, allocation_oom_still_possible=True)), flush=True)
    start = time.monotonic()
    samples = []
    from gpu_policy import inventory,MAX_GPUS
    from gpu_metrics_strict import process_utilization,diagnostics
    from gpu_priority import ArrivalOrder
    arrival_order = ArrivalOrder()
    process_util_sums = {}
    metric_missing_since = {}
    metric_last_verified = {}
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
                priority=arrival_order.observe(elapsed,active,foreign)
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
                for pid in active:
                    if pid in pid_util:process_util_sums[pid]=process_util_sums.get(pid,0)+pid_util[pid]
                # Include the full startup window; missing samples contribute 0.
                process_means={pid:process_util_sums.get(pid,0)/len(samples) for pid in active}
                elapsed = time.monotonic() - start
                metric_gap_exhausted=[]
                for pid in active:
                    if pid in unresolved:
                        since=metric_missing_since.setdefault(pid,metric_last_verified.get(pid,0.0))
                        pid_util[pid]=0
                        if elapsed>=280 and elapsed-since>9:metric_gap_exhausted.append(pid)
                    else:
                        metric_last_verified[pid]=elapsed;metric_missing_since.pop(pid,None)
                # Missing readings conservatively erase the WHOLE preceding
                # sampling interval; never carry forward a favorable PID rate.
                for pid in missing_initial:
                    utilization_window.previous[pid]=0
                    recent_window.previous[pid]=0
                cumulative, rolling = utilization_window.add(
                    elapsed, max(sm, bandwidth), {pid: pid_util.get(pid, 0) for pid in active})
                _, recent = recent_window.add(elapsed, max(sm, bandwidth), {pid:pid_util.get(pid,0) for pid in active})
                process_means = {pid: min(process_means[pid], cumulative.get(pid, 0)) for pid in active}
                telemetry = {"seconds": round(elapsed, 2), "sm": sm, "bandwidth": bandwidth,
                             "memory_mib": memory, "max_memory_mib": max_memory_mib, "total_memory_mib": total_mib,
                             "mean_utilization": sum(samples) / len(samples),
                             "gpu_arrival_order": priority,
                             "process_utilization": {pid:pid_util.get(pid) for pid in active},
                             "process_metric_samples": metric_samples,
                             "metric_missing_conservative_zero_pids": missing_initial,
                             "metric_unresolved_seconds": {pid:elapsed-since for pid,since in metric_missing_since.items()if pid in active},
                             "metric_gap_limit_seconds": 9,
                             "whole_missing_interval_counted_as_zero": True,
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
                if metric_gap_exhausted:
                    reason = "Per-process GPU metrics unavailable beyond bounded 9s interval; stopping this job"
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


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--profile-rollout", action="store_true")
    parser.add_argument("--shared-profile-scenes", type=int, choices=(1,3), default=1, help="Three scenes per worker use separate original seed groups for bounded resource profiling")
    parser.add_argument("--shared-model-pair", action="store_true", help="Two independent same-seed diagnostic processes sharing frozen weights")
    parser.add_argument("--defer-scratch-cleanup", action="store_true")
    parser.add_argument("--cache-mode", choices=("off", "export", "import"), default="off")
    parser.add_argument("--cache-bundle", type=Path)
    parser.add_argument("--gemm-backends", choices=("original", "ATEN"), default="original",
                        help="Independent compiled variant with a smaller GEMM candidate set")
    parser.add_argument("--fast-load", action="store_true",
                        help="Use the independently verified checkpoint-loading optimization")
    parser.add_argument("--compile-threads", type=int, choices=(1, 2, 4, 8, 16), default=16,
                        help="CPU compiler parallelism; preserves the frozen compiled backend")
    args = parser.parse_args()
    if args.gemm_backends != "original" and not args.fast_load:
        parser.error("Independent compiler variants require the profiled fast-load wrapper")
    if args.cache_mode != "off" and (not args.fast_load or args.gemm_backends != "ATEN"):
        parser.error("Compiler cache verification currently requires --fast-load --gemm-backends ATEN")
    if (args.cache_mode == "import") != bool(args.cache_bundle):
        parser.error("--cache-bundle is required exactly with --cache-mode import")
    if args.defer_scratch_cleanup and args.cache_mode == "off":
        parser.error("Deferred cleanup requires the deployment cache wrapper")
    if args.shared_model_pair and (args.cache_mode != "import" or not args.defer_scratch_cleanup):
        parser.error("Shared pair requires validated cache import and CPU-deferred cleanup")
    if args.shared_profile_scenes != 1 and not args.shared_model_pair:
        parser.error("Shared scene profiling requires --shared-model-pair")
    cleanup_token = uuid.uuid4().hex if args.defer_scratch_cleanup else None
    bundle = None
    if args.cache_bundle:
        from bundle_store import validated_store_path
        bundle = validated_store_path(args.cache_bundle)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    marker = container.WORK / ("gpu-doctor-" + str(args.gpu) + ".json")
    if not marker.is_file():
        raise RuntimeError("Run the bounded GPU doctor on this GPU first")
    info = gpu_doctor.gpu_snapshot(args.gpu)
    certificate = json.loads(marker.read_text())
    if certificate["uuid"] != info["uuid"]:
        raise RuntimeError("GPU identity changed since doctor")
    if certificate["build_mtime_ns"] != (container.WORK / "stage-bootstrap.ok").stat().st_mtime_ns:
        raise RuntimeError("Environment build changed since GPU doctor")
    gpu_doctor.check_admission(info)
    arguments = argparse.Namespace(artifacts=container.ARTIFACTS, entry="baseline",
                                   gpu=args.gpu, image="cx-user-rootfs")
    original = frozen_run.command(arguments, [
        "--task", "stack_bowls_two", "--task-config", "demo_randomized",
        "--seed", "0", "--episodes", str(args.shared_profile_scenes),
    ])
    inner = original[original.index(arguments.image) + 1:]
    cmd = container.command(inner, mount_project=True)
    split = cmd.index("--")
    if args.fast_load:
        proof = json.loads((container.HERE / "reports/target_fast_load_cpu_20261022.json").read_text())
        if not proof.get("passed"):
            raise RuntimeError("CPU model-state equivalence proof is required")
        cmd[split:split] = ["--setenv", "CX_PYTHON",
                           str(container.CANONICAL / "VLA-Adapter/cx_new_direction_guide_TTA/deployment/startup_python.sh")]
    cmd[split:split] = gpu_doctor.driver_options(info) + [
        "--setenv", "TORCHINDUCTOR_COMPILE_THREADS", str(args.compile_threads),
    ]
    if args.gemm_backends == "ATEN":
        split = cmd.index("--")
        cmd[split:split] = ["--setenv", "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN",
                           "--setenv", "CX_COMPILER_VARIANT", "gemm_aten"]
    split = cmd.index("--")
    driver = gpu_doctor.query(["-i", str(args.gpu), "--query-gpu=driver_version"]).strip()
    cache_options = ["--setenv", "CX_COMPILER_CACHE_MODE", args.cache_mode,
                     "--setenv", "CX_TARGET_GPU_UUID", info["uuid"],
                     "--setenv", "CX_TARGET_DRIVER_VERSION", driver]
    if args.shared_model_pair:
        cache_options += ["--setenv", "CX_SHARED_MODEL_ROLE", "owner",
                          "--setenv", "CX_SHARED_CONSUMER_SEED", "1" if args.shared_profile_scenes == 3 else "0"]
    if cleanup_token:
        cache_options += ["--setenv", "CX_DEFER_SCRATCH_CLEANUP", "1",
                          "--setenv", "CX_CLEANUP_TOKEN", cleanup_token]
    if args.profile_rollout:
        cache_options += ["--setenv", "CX_ROLLOUT_TIMING", "1"]
    if bundle:
        cache_options += ["--ro-bind", str(bundle), "/run/cxcache_bundle"]
    cmd[split:split] = cache_options
    report = container.HERE / "reports" / ("target_baseline_" + str(time.time_ns()) + ".jsonl")
    print("BASELINE_ACCEPTANCE_GPU", info, "TELEMETRY", str(report), flush=True)
    print("BASELINE_COMPILE_THREADS", args.compile_threads,
          "CPU_AFFINITY", sorted(os.sched_getaffinity(0)), flush=True)
    def cleanup(reason, code):
        if cleanup_token:
            cleaned = finish_deferred_cleanup(cleanup_token)
            if args.shared_model_pair and reason is None and code == 0 and len(cleaned) != 2:
                raise RuntimeError("Shared diagnostic did not finalize exactly two private process caches")
    guarded_command(cmd, info, report, gpu=args.gpu,
        deadline=(480 if args.shared_profile_scenes == 3 else 270) if args.shared_model_pair else 1800,
        cleanup=cleanup)


if __name__ == "__main__":
    main()
