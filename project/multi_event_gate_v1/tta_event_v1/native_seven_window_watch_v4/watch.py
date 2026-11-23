"""Bounded foreground monitor; its exit is never an evaluation result."""
import argparse, collections, datetime, json, pathlib, re, subprocess, time

T = pathlib.Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--invocation', required=True)
    args = parser.parse_args()
    assert all(re.fullmatch(r'native_seven_[a-z0-9_]+', x) for x in (args.backend, args.campaign))
    assert re.fullmatch(r'cx-tta-[a-z0-9-]+\.service', args.unit)
    base = T / 'outputs' / args.backend
    campaign = T / 'outputs' / args.campaign
    started = time.monotonic()
    while time.monotonic() - started < 7200:
        result = subprocess.run(['systemctl', '--user', 'show', args.unit, '-p', 'ActiveState', '-p', 'MainPID', '-p', 'InvocationID'], check=True, capture_output=True, text=True)
        unit = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if unit['ActiveState'] in ('active', 'activating'):
            assert unit['InvocationID'] == args.invocation, 'Inspect changed service identity; no automatic restart'
        else:
            assert unit['InvocationID'] in ('', args.invocation), 'Unexpected replacement service identity'
        value = dict(time=datetime.datetime.now().isoformat(timespec='seconds'), unit=unit, read_only=True)
        stages = sorted(p for p in base.glob('*_*') if p.is_dir() and (p / 'live_status.json').exists() and not (p / 'actual_host_wait_exit.json').exists())
        if stages:
            folder = stages[-1]
            live = read(folder / 'live_status.json')
            resource = live.get('resources', {})
            value.update(stage=folder.name, phase=live.get('phase'), GPU=live.get('gpu'), status_age_seconds=round(time.time() - (folder / 'live_status.json').stat().st_mtime, 2), resource={k: resource.get(k) for k in ('seconds', 'gpu_pids', 'account_gpu_count', 'memory_mib', 'process_rolling180')})
            if (folder / 'execution_state.json').exists():
                root = pathlib.Path(read(folder / 'execution_state.json')['root'].replace('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA', str(T.parents[1])))
                value.update(root=str(root), accepted_this_batch=dict(collections.Counter(read(p)['task'] for p in (root / 'accepted').glob('*.json'))), early_dispatches=len(list(root.glob('cases/*/early_pairing_dispatch.json'))), finished={c: len(list(root.glob('cases/*/' + c + '/result.json'))) for c in ('M_event_full', 'TTA', 'zero_update', 'shadow_update', 'handoff_reference')}, GPU_complete=(root / 'GPU_episodes_complete.json').exists(), native_audit=(root / 'CPU_episode_audit.json').exists())
                if not (root / 'attempt/barrier/permit.json').exists():
                    value['phase'] = 'CPU_preparation_before_GPU_permit'
        # Read-only account context explains why a free physical card may not
        # be admissible. Never reserve a GPU, restart a job or synthesize data.
        try:
            from native_governor180_v5.observer import latest, OWN
            from native_governor180_v5.policy import daytime
            fleet = latest()
            value['account_context'] = dict(own_GPU_count=fleet['own_gpu_count'], free_physical_GPUs=fleet['available_indices'], daytime=daytime(fleet['wall_time']), owned_cards=[dict(GPU=c['index'], util180=c['own_rolling180']['util'], memory_pct180=c['own_rolling180']['memory_pct']) for c in fleet['cards'] if c['own_pids']])
            if args.backend == 'native_seven_official_v8' and 'wait' in value.get('phase', ''):
                history = read(OWN / 'window.json')
                now = fleet['monotonic']
                rows = [r for r in history if now - 180 <= r['monotonic'] <= now]
                occupied = [r['monotonic'] for r in rows if any(c['own_pids'] for c in r['cards'])]
                quiet = 0 if fleet['own_gpu_count'] else max(0, now - (occupied[-1] if occupied else rows[0]['monotonic'])) if rows else 0
                value['CPU_wait_explanation'] = dict(project_daytime_serialization=True, no_GPU_owned_by_this_queue=True, account_quiet_seconds=round(quiet, 1), required_actual_window_seconds=180, reason='wait_for_other_account_GPU_jobs_and_full_quiet_window' if daytime(fleet['wall_time']) else 'original_empty_card_and_max3_admission')
        except (OSError, RuntimeError, AssertionError, KeyError) as error:
            value['account_observer_error'] = repr(error)
        value['actual_host_exits'] = {p.parent.name: read(p) for p in base.glob('*/actual_host_wait_exit.json')}
        value['qualified'] = {g: sorted(p.name for p in (base / ('supervision_' + g)).glob('qualified_*.json'))[-1:] for g in ('plain', 'gated')}
        value['campaign_failure'] = read(campaign / 'failure.json') if (campaign / 'failure.json').exists() else None
        print(json.dumps(value, ensure_ascii=False), flush=True)
        if unit['ActiveState'] not in ('active', 'activating') or value['campaign_failure']:
            break
        time.sleep(45)
    print('READ_ONLY_WATCH_ENDED_NOT_GPU_OUTCOME', flush=True)

if __name__ == '__main__':
    main()
