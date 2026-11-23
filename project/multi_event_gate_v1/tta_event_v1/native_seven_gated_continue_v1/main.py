"""Continue untouched gated official stages; no GPU computation in this parent."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

T = Path(__file__).resolve().parents[1]
P = T.parents[1]
O = T / 'outputs/native_seven_gated_continue_v1'
BASE = T / 'outputs/native_seven_official_v9'


def write(path, value):
    assert path.resolve().is_relative_to(O.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def eligible(files, remaining_observer_seconds, required_gpu_seconds):
    assert required_gpu_seconds > 0
    assert not set(files).intersection({
        'actual_host_wait_exit.json', 'execution_state.json',
        'supervision_actual_exit.json', 'campaign_failure.json',
        'supervision_host.log', 'actual_launch_wait.json',
    }), 'Inspect existing attempt; no automatic GPU retry'
    return remaining_observer_seconds > required_gpu_seconds + 1800


def wait_for_observation():
    from native_governor180_v5.observer import latest
    start = time.monotonic()
    while time.monotonic() - start < 6 * 3600:
        try:
            fleet = latest()
            if fleet['available_indices'] and fleet['own_gpu_count'] < 3:
                return fleet
            reason = 'No empty card or account already uses three GPUs'
        except RuntimeError as error:
            reason = str(error)
        value = dict(phase='CPU_only_wait_for_fresh_observer_and_empty_GPU', reason=reason, elapsed_seconds=time.monotonic() - start, no_GPU_work_in_parent=True)
        temp = O / ('live_status.' + str(os.getpid()) + '.tmp')
        temp.write_text(json.dumps(value) + '\n')
        os.replace(temp, O / 'live_status.json')
        time.sleep(15)
    raise TimeoutError('Six hours CPU-only wait; no GPU launched by this parent during wait')


def main():
    from native_seven_official_v9.common import read, verify, sha
    from native_seven_official_v9.collection_plan import read_release
    from native_seven_official_v9.qualification import qualified_stage
    from native_seven_official_v9.supervision import aggregate
    from native_governor180_v5.observer import OWN

    O.mkdir(parents=True, exist_ok=True)
    supervision = BASE / 'supervision_gated'
    supervision.mkdir(exist_ok=True)
    with (supervision / 'supervision.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(O / 'launch.json', dict(
            pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
            tasks=['beat_block_hammer', 'turn_switch', 'adjust_bottle', 'open_laptop'],
            backend='native_seven_official_v9', official_seed=0,
            no_GPU_work_in_parent=True, no_failed_GPU_retry=True,
            source_sha256={str(p): sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()},
        ))
        for number in range(17):
            stage = 'gated_' + str(number).zfill(3)
            folder = BASE / stage
            if (folder / 'actual_host_wait_exit.json').exists():
                qualified_stage(folder)
                continue
            fleet = wait_for_observation()
            observer = read(OWN / 'launch.json')
            remaining = observer['lifetime_seconds'] - (float(Path('/proc/uptime').read_text().split()[0]) - observer['start_ticks'] / os.sysconf('SC_CLK_TCK'))
            inherited = 0
            if number % 2:
                root, prior = qualified_stage(BASE / ('gated_' + str(number - 1).zfill(3)))
                inherited = prior['inherited_stage_GPU_seconds'] + read_release(root / 'attempt/resources.jsonl')['cleanup_seconds_after_start']
            required = 14400 - inherited
            if not eligible([p.name for p in folder.iterdir()] if folder.exists() else [], remaining, required):
                write(O / 'CPU_boundary_stop.json', dict(stage=stage, reason='observer_lifetime_insufficient_for_full_next_stage', observer_seconds_remaining=remaining, required_GPU_seconds=required, no_live_GPU_in_this_parent=True))
                return
            if (folder / 'study.json').exists():
                spec = read(folder / 'study.json')
                verify(spec)
                assert spec['group'] == 'gated' and spec['official_seed'] == 0
            env = dict(os.environ, CX_ONPOLICY_STAGE=stage, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(P / 'deployment') + ':' + str(T))
            command = [sys.executable, '-B', '-u', '-m', 'native_seven_official_v9.launch']
            folder.mkdir(exist_ok=True)
            write(O / (stage + '_launch.json'), dict(command=command, stage=stage, observer_seconds_remaining=remaining, required_GPU_seconds=required, fresh_own_GPU_count=fleet['own_gpu_count'], fresh_free_GPUs=fleet['available_indices']))
            with (folder / 'supervision_host.log').open('x') as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = process.wait()
            actual = dict(exit_code=code, actual_Popen_wait_return=True, pid=process.pid, proc_absent_after_wait=not Path('/proc', str(process.pid)).exists())
            write(O / (stage + '_actual_launch_wait.json'), actual)
            # This is the original supervision wrapper's exact wait record.
            with (folder / 'supervision_actual_exit.json').open('x') as handle:
                json.dump(actual, handle)
            assert code == 0, ('Stopped; no failed-GPU retry', stage, code)
            qualified_stage(folder)
            result = aggregate('gated')
            print('GATED_QUALIFIED', json.dumps(result['groups']), flush=True)
        result = aggregate('gated')
        assert result['complete'] and result['scenes_per_task'] == 100
        with (supervision / 'official100_complete.json').open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        write(O / 'complete.json', result)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        if O.exists():
            write(O / 'failure.json', dict(traceback=traceback.format_exc(), no_automatic_GPU_retry=True))
        raise
