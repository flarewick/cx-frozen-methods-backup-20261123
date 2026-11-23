"""Isolated schema3 expert data for bowls/blocks adapter initialization; no VLA evaluation."""
import argparse
import atexit
import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import traceback

from .audit import digest
from .paths import WORKSPACE_ROOT, new_run, output_path, write_json, replace_json
from .stacking_training_design import TASKS, load_design
from .expert_screening import run_expert_plan, ExpertPlanningFailure
from .gpu_job_status import publish_job


def options(argv=None):
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--task', choices=TASKS, required=True)
    ap.add_argument('--split', choices=('smoke', 'adapter_fit', 'adapter_validation'), default='smoke')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--min-free-gb', type=int, default=10)
    ap.add_argument('--tokenizer-path', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    if not a.gpu.isdigit() or a.min_free_gb < 1:
        ap.error('Numeric GPU and positive free disk bound required')
    # The simulator changes cwd to its private runtime before reading the protocol again.
    a.protocol = a.protocol.resolve(strict=True)
    if a.tokenizer_path is not None:
        a.tokenizer_path = a.tokenizer_path.resolve(strict=True)
    return a


def preflight(a):
    from .collect_smoke import make_config
    from .stacking_support import task_contract
    p = load_design(a.protocol)
    # Use the same frozen official runtime as evaluation, including provenance.
    from .evaluate_stacking import options as eval_options, preflight as eval_preflight
    _, _, tokenizer, evaluation = eval_preflight(eval_options([
        '--task', a.task, '--condition', 'Z_zero', '--dry-run'] +
        ([] if a.tokenizer_path is None else ['--tokenizer-path', str(a.tokenizer_path)])))
    runtime = Path(evaluation['runtime_source'])
    extra, assets, budget = task_contract(runtime, a.task)
    config_args = argparse.Namespace(task=a.task, task_config=p['task_config'])
    config = make_config(runtime, config_args, WORKSPACE_ROOT/'outputs/dry_run_not_created')
    if config['save_freq'] != 15 or config['data_type']['rgb'] is not True:
        raise ValueError('Unexpected expert observation contract')
    spec = p['splits'][a.split]
    return p, runtime, tokenizer, dict(task=a.task, split=a.split, episodes=spec['episodes'],
        seed_window=[spec['start'], spec['stop']], task_config=p['task_config'],
        role=spec['role'], native_evaluation_budget=budget, expert_save_freq=config['save_freq'],
        schema_version=3, video=False, mode='expert_collection_not_evaluation',
        training_ready=False, gpu_started=False, runtime_source=str(runtime),
        sources={str(f): digest(f) for f in extra+assets},
        output_parent=str(WORKSPACE_ROOT/'outputs/stacking_training/expert'/a.task/a.split))


def cleanup_frames(task, attempt_path):
    """Delete only this collector's current temporary frames, not retained data."""
    raw = getattr(task, 'folder_path', {}).get('cache')
    if raw is None:
        return
    path, attempt = Path(raw), output_path(attempt_path)
    if path.is_symlink() or not path.resolve().is_relative_to(attempt) or '.cache' not in path.parts:
        raise ValueError('Refusing unsafe frame cache cleanup')
    if path.exists():
        shutil.rmtree(path)


def collect(task, config, args, p, run, seed_all, unstable_type, export, qualify):
    """Plan/replay on a predetermined scene window; failure is never a gate label."""
    spec = p['splits'][args.split]
    rows, accepted = [], 0
    protocol_sha = digest(args.protocol)
    for seed in range(spec['start'], spec['stop']):
        if shutil.disk_usage(run).free < args.min_free_gb * 2**30:
            raise OSError('Insufficient free disk; stopped before next scene')
        attempt = run/'expert_data'/f'seed_{seed}'
        row = dict(task=args.task, seed=seed, split=args.split, episode_index=accepted,
                   accepted=False, plan_success=False, replay_success=False)
        opened = False
        try:
            for replay in (False, True):
                seed_all(seed)
                opts = copy.deepcopy(config)
                opts.update(need_plan=not replay, save_data=replay, save_path=str(attempt))
                opened = True
                task.setup_demo(now_ep_num=accepted, seed=seed, **opts)
                if replay:
                    traj = task.load_tran_data(accepted)
                    opts.update(left_joint_path=traj['left_joint_path'], right_joint_path=traj['right_joint_path'])
                    task.set_path_lst(opts)
                # The known missing-target exception is skippable only in expert planning.
                info = task.play_once() if replay else run_expert_plan(task)
                success = bool(task.plan_success and task.check_success())
                row['replay_success' if replay else 'plan_success'] = success
                if not replay and success:
                    task.save_traj_data(accepted)
                if replay and success:
                    h5 = export(task, config['cx_runtime'], args.task, info, seed, accepted)
                    import h5py
                    with h5py.File(h5, 'r+') as h:
                        h.attrs.update(stacking_split=args.split, stacking_protocol_sha256=protocol_sha,
                                       instruction_split='seen', data_role=spec['role'])
                    row.update(hdf5=str(h5), capture=qualify(h5, args.task, seed, args.split, protocol_sha),
                               scene_info=info, accepted=True)
                if replay:
                    cleanup_frames(task, attempt)
                task.close_env()
                opened = False
                if not success:
                    row['excluded_reason'] = 'expert_replay_failed' if replay else 'expert_plan_failed'
                    break
            if row['accepted']:
                accepted += 1
        except (ExpertPlanningFailure, unstable_type) as exc:
            row.update(excluded_reason='expert_missing_grasp_target' if isinstance(exc, ExpertPlanningFailure)
                       else 'unstable_scene', error_type=type(exc).__name__, error=str(exc))
        except BaseException:
            row['error'] = traceback.format_exc()
            rows.append(row)
            replace_json(run/'attempts.json', rows)
            raise
        finally:
            if opened:
                try:
                    task.close_env()
                except Exception:
                    pass
        rows.append(row)
        replace_json(run/'attempts.json', rows)
        replace_json(run/'progress.json', dict(accepted=accepted, requested=spec['episodes'], attempts=len(rows)))
        publish_job(run, 'collecting', ready=accepted > 0, completed=accepted, requested=spec['episodes'])
        print(f'\033[32mEXPERT DATA {args.task}/{args.split}: {accepted}/{spec["episodes"]} accepted, '
              f'{len(rows)} attempts (not policy success rate)\033[0m', flush=True)
        if accepted == spec['episodes']:
            break
    return dict(status='collection_complete' if accepted == spec['episodes'] else 'insufficient_expert_successes',
                task=args.task, split=args.split, accepted_episodes=accepted, attempts=len(rows),
                residual_training_ready=False, gate_training_ready=False, policy_evaluation=False)


def execute(a, p, source, tokenizer, run):
    from .native_entry_storage import NativeEntryStorage
    from .collect_smoke import make_config
    from .recording import instrument_runtime, install_recording
    from .capture import export_capture
    from .stacking_training_data import qualify_capture
    storage = NativeEntryStorage(runtime_source=source)
    atexit.register(storage.cleanup)  # Before importing CUDA libraries.
    runtime = storage.prepare_runtime(run)
    original = digest(runtime/'envs/_base_task.py')
    count = instrument_runtime(runtime)
    write_json(run/'instrumentation.json', dict(original_sha256=original,
        instrumented_sha256=digest(runtime/'envs/_base_task.py'), counted_step_calls=count))
    storage.configure_runtime(run, runtime, tokenizer)
    import torch
    import torch.multiprocessing as mp
    from .evaluate_b2 import seed_all
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; collection not started')
    mp.set_start_method('spawn', force=True)
    a.task_config = p['task_config']
    config = make_config(runtime, a, run)
    config.update(eval_mode=False, eval_video_log=False, eval_video_save_dir=None, render_freq=0)
    write_json(run/'resolved_config.json', config)
    task = getattr(importlib.import_module('envs.'+a.task), a.task)()
    from envs.utils.create_actor import UnStableError
    install_recording(task, effect_schema=True)
    result = collect(task, config, a, p, run, seed_all, UnStableError, export_capture, qualify_capture)
    write_json(run/'source_runtime_audit.json', storage.verify_source())
    load_design(a.protocol)
    return result


def main(argv=None):
    a = options(argv)
    p, source, tokenizer, manifest = preflight(a)
    if a.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    run = new_run('collect', group=('stacking_training', 'expert', a.task, a.split))
    print('RUN', run, flush=True)
    manifest.update(protocol=str(a.protocol.resolve()), protocol_sha256=digest(a.protocol))
    write_json(run/'manifest.json', manifest)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    publish_job(run, 'initializing', ready=False)
    from .evaluate_stacking import Tee
    import sys
    old_out, old_err = sys.stdout, sys.stderr
    try:
        with (run/'console.log').open('x') as log:
            sys.stdout, sys.stderr = Tee(old_out, log), Tee(old_err, log)
            try:
                result = execute(a, p, source, tokenizer, run)
                write_json(run/'result.json', result)
                if result['status'] == 'collection_complete':
                    from .stacking_training_data import audit_collection
                    write_json(run/'data_audit.json', audit_collection(run, a.protocol))
                    publish_job(run, 'finished', ready=False, result=str(run/'result.json'))
                    print('COLLECTION COMPLETE', run, flush=True)
                    return 0
                publish_job(run, 'failed', ready=False, reason=result['status'])
                return 2
            except BaseException:
                error = traceback.format_exc()
                write_json(run/'failure.json', dict(traceback=error))
                publish_job(run, 'failed', ready=False, traceback=error)
                raise
            finally:
                sys.stdout.close(); sys.stderr.close()
    finally:
        sys.stdout, sys.stderr = old_out, old_err


if __name__ == '__main__':
    raise SystemExit(main())
