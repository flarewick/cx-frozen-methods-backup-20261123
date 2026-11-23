"""Verify full physical and RGB source replay before external source collection."""
import argparse
import importlib
import json
from pathlib import Path
import signal
import traceback

import numpy as np

from .audit import digest
from .paths import new_run, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--gpu', default='0')
    args = p.parse_args()
    if not args.gpu.isdigit():
        p.error('Nonnegative GPU index required')
    source = args.source.resolve()
    from .audit_recovery_sources import audit
    from .collect_recovery import signature
    from .eval_diagnostics import state_snapshot
    from .evaluate_b2 import configure_runtime, resolve_tokenizer, seed_all
    from .probe_action_replay import (load_source, source_runtime, source_scene_index,
                                     prime_warp_cache, compare_physical)
    saved = audit(source)
    m = json.loads((source / 'manifest.json').read_text())
    episodes = json.loads((source / 'episodes.json').read_text())
    if m['role'] != 'gate' or len(episodes) != 1 or not episodes[0]['eligible']:
        raise ValueError('Exactly one eligible preregistered gate scene required')
    seed = episodes[0]['seed']
    manifest, episode, rows = load_source(source, seed)
    runtime = source_runtime(source)
    run = new_run('source_replay', group=('intervention_gates', m['task']))
    print(run, flush=True)
    write_json(run / 'manifest.json', dict(source=str(source), protocol_sha256=m['protocol_sha256'],
                                         scope='Full action-only source replay, no policy queries'))
    task = None
    def timeout(*_):
        raise TimeoutError('Replay exceeded 900 seconds')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(900)
    try:
        prime_warp_cache(source, run)
        configure_runtime(run, runtime, resolve_tokenizer())
        config = json.loads((source / 'resolved_config.json').read_text())
        config.update(save_path=str(run / 'runtime_data'), cx_runtime=str(runtime),
                      save_data=False, eval_video_log=False, eval_video_save_dir=None)
        task = getattr(importlib.import_module(f"envs.{m['task']}"), m['task'])()
        seed_all(seed)
        task.setup_demo(now_ep_num=source_scene_index(manifest, seed), seed=seed,
                        **dict(config, need_plan=True, is_test=True))
        task.set_instruction(instruction=episode['instruction'])
        obs = task.get_obs(); checks = []
        for i, row in enumerate(rows):
            before = compare_physical(row['before'], state_snapshot(task))
            rgb = signature(obs) == row['observation_components']
            if not before['matched'] or not rgb:
                raise RuntimeError(f'Physical/RGB mismatch before chunk {i}')
            with np.load(source / 'observations' / f'seed_{seed}' / f'chunk_{i:03d}.base.npz') as z:
                actions = z['base_action']
            count = task.take_action_cnt
            for action in actions:
                task.take_action(action)
                obs = task.get_obs()
            after = compare_physical(row['after'], state_snapshot(task))
            if task.take_action_cnt-count != row['executed'] or not after['matched']:
                raise RuntimeError(f'Native command/physical mismatch after chunk {i}')
            checks.append(dict(chunk=i, rgb_exact=rgb, before=before, after=after))
        passed = bool(task.eval_success) == episode['success'] and task.take_action_cnt == episode['executed_actions']
        result = dict(passed=bool(passed), seed=seed, checks=checks,
                      native_actions=int(task.take_action_cnt), success=bool(task.eval_success))
        write_json(run / 'result.json', result)
        if not passed:
            raise RuntimeError('Source terminal mismatch')
        evidence = dict(saved['evidence_sha256'])
        evidence.update({str(run / n): digest(run / n) for n in ('manifest.json', 'result.json')})
        write_json(source / 'gate.json', dict(passed=True, protocol_sha256=m['protocol_sha256'],
            replay_run=str(run), evidence_sha256=evidence,
            scope='One-scene recording and exact RGB/physical replay gate. Not teacher or adapter validation.'))
        print(json.dumps(dict(passed=True, chunks=len(checks), native_actions=task.take_action_cnt)), flush=True)
    except BaseException:
        write_json(run / 'failure.json', dict(traceback=traceback.format_exc()))
        raise
    finally:
        signal.alarm(0)
        if task is not None and hasattr(task, 'scene'):
            task.close_env()


if __name__ == '__main__':
    main()
