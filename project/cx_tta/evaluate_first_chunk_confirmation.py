"""Held-out whole-scene confirmation of frozen first-full ARM correction then exit."""
import argparse
import atexit
import importlib
import os
from pathlib import Path
import signal
import time
import traceback
import uuid
import numpy as np

from .paths import ROOT, new_run, output_path, write_json, replace_json
from .audit import digest
from .native_history_cache import read, verify, require
from .arm_prescription_policy import ArmPrescriptionPolicy
from .first_chunk_contract import CONDITIONS, validate_protocol, registry_records
from .causal_arm_analysis import order
from .collect_phase_targets import start, TargetTrace, episode_record
from .evaluate_structured_attribution import PhysicalPolicy, steady_probe
from .geometry_labels import episode_definition
from .train_output_support import BASE


def inputs(args):
    p = read(args.protocol)
    validate_protocol(p)
    verify(p['source_sha256'])
    require(args.task in p['tasks'] and args.block in p['process_blocks'], 'Registered task/block')
    rows = registry_records(args.protocol, args.registry)
    seeds = [s for s in p['seeds'] if next(r for r in rows if r['task'] == args.task and r['seed'] == s)['eligible']]
    require(bool(seeds), 'Eligible registered scenes required')
    if args.mode == 'smoke':
        seeds = seeds[:1]
    return p, rows, seeds


def rollout(task, config, reg, run, wrapper, definitions, names, args):
    from .evaluate_b2 import step_chunk_official
    wrapper.reset_history()
    wrapper.episode_seed = reg['seed']
    trace = None
    try:
        obs = start(task, config, reg, run)
        physical = PhysicalPolicy(wrapper, task)
        stem = f"seed{reg['seed']}_b{args.block}_{wrapper.condition}"
        trace = TargetTrace(run, stem, physical, args.task, definitions, names)
        while task.take_action_cnt < task.step_lim and not task.eval_success:
            if args.mode == 'smoke' and wrapper.query_count >= 5:
                break
            _, _, obs = step_chunk_official(task, physical, obs, reg['instruction'], True, trace=trace)
            replace_json(run/'progress.json', dict(phase='rollout', seed=reg['seed'], condition=wrapper.condition,
                query=wrapper.query_count, native_count=int(task.take_action_cnt)))
        complete = args.mode == 'full' and (bool(task.eval_success) or task.take_action_cnt == task.step_lim)
        trace.cache.finish(complete=complete, success=bool(task.eval_success) if complete else None,
            role='diagnostic_confirmation')
        ep = episode_record(reg['seed'], wrapper.condition, wrapper, task, trace, complete)
        ep.update(repeat=args.block, instruction=reg['instruction'],
            observed_native_success=bool(task.eval_success), initial_native_sha256=reg['initial_physical_sha256'])
        if not complete:
            ep['reason'] = 'bounded_interface_only_no_success_claim'
        return ep
    finally:
        task.close_env()


def execute(args, storage):
    p, rows, seeds = inputs(args)
    schedule = order(seeds, CONDITIONS, args.block)
    manifest = dict(stage=p['stage'], protocol=str(args.protocol), protocol_sha256=digest(args.protocol),
        task=args.task, mode=args.mode, block_id=args.block, registry=str(args.registry),
        registry_records_sha256=digest(args.registry/'records.json'), seeds=seeds, attempted_seeds=p['seeds'],
        conditions=list(CONDITIONS), max_chunks=5 if args.mode == 'smoke' else None,
        order=[dict(seed=s, repeat=b, condition=c) for s,b,c in schedule],
        invalid_scenes=[r for r in rows if r['task'] == args.task and not r['eligible']],
        role='Whole-scene confirmation only; excluded from fitting and threshold selection',
        process_instance_id=str(uuid.uuid4()), pid=os.getpid(), no_RGB_or_video=True,
        no_new_training=True, training_approved=False)
    if args.dry_run:
        print('DRY_RUN', manifest, flush=True)
        return
    import torch
    from .cache_expert import create_base_policy
    from .evaluate_b2 import configure_inference, resolve_tokenizer
    from .evaluate_structured_action import check_training_policy_identity
    from .output_support_policy import OutputSupportPolicy
    from .collect_smoke import make_config
    from .first_chunk_confirmation_analysis import summarize
    require(torch.cuda.is_available(), 'Visible CUDA required')
    run = new_run('smoke' if args.mode == 'smoke' else 'block', group=('diagnostics','first_chunk_confirmation',args.task))
    print('RUN', run, flush=True)
    write_json(run/'manifest.json', manifest)
    (run/'legal_cache').mkdir()
    episodes = []
    began = time.monotonic()
    try:
        tokenizer = resolve_tokenizer()
        identity = check_training_policy_identity(BASE, tokenizer)
        storage.runtime_source = Path(p['reference'][args.task]['run'])/'robotwin'
        runtime = storage.prepare_runtime(run)
        storage.configure_runtime(run, runtime, tokenizer)
        config = make_config(runtime, argparse.Namespace(task=args.task,task_config='demo_randomized'), run)
        config.update(eval_mode=True, save_data=False, eval_video_log=False, eval_video_save_dir=None, render_freq=0)
        write_json(run/'resolved_config.json', config)
        old = read(Path(p['reference'][args.task]['run'])/'provenance.json')
        runtime_hash = {str(f.relative_to(runtime)):digest(f) for f in runtime.rglob('*.py') if '__pycache__' not in f.parts}
        require(runtime_hash == old['runtime_sha256'], 'Original official runtime')
        verify(old['code_sha256'])
        write_json(run/'provenance.json', dict(runtime_sha256=runtime_hash, identity=identity['live_identity'],
            policy_source_sha256=identity['policy_source_sha256'], source_sha256=p['source_sha256']))
        configure_inference('compiled', 0)
        torch.set_num_threads(2)
        replace_json(run/'progress.json', dict(phase='loading_shared_base', completed=0, requested=len(schedule)))
        base = create_base_policy()
        base._model.eval().requires_grad_(False)
        wrappers = {c:ArmPrescriptionPolicy(base, Path(p['training_run']), c) for c in CONDITIONS}
        for w in wrappers.values():
            w.check_robot_config(config)
        task = getattr(importlib.import_module('envs.'+args.task), args.task)()
        write_json(run/'hardware.json', dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            cuda=torch.version.cuda, compiled=hasattr(base._sample_actions,'_torchdynamo_orig_callable'),
            global_dynamo_disabled=bool(torch._dynamo.config.disable), sample_kwargs=base._sample_kwargs,
            shared_sampler=all(w.sampler is base._sample_actions for w in wrappers.values()),
            process_instance_id=manifest['process_instance_id'], pid=os.getpid(),
            compiler_scratch=read(run/'compiler_temp.json')['path']))
        first = next(r for r in rows if r['task'] == args.task and r['seed'] == seeds[0])
        l0 = OutputSupportPolicy(base, Path(p['training_run']), 'L0')
        replace_json(run/'progress.json', dict(phase='common_warmup', completed=0, requested=len(schedule)))
        probe = steady_probe(task, config, seeds[0], first['instruction'], {'L0':l0,'S0_zero':wrappers['Z_zero']})
        np.savez_compressed(run/'steady_probe.npz', base=probe.pop('base_values'))
        write_json(run/'steady_probe.json', probe)
        require(probe['passed'], 'Zero condition equals true L0')
        for seed, block, condition in schedule:
            reg = next(r for r in rows if r['task'] == args.task and r['seed'] == seed)
            definitions, _ = episode_definition(runtime, args.task, reg['grounded_objects'])
            names = [reg['grounded_objects']['{A}'].split('/')[0],
                     reg['grounded_objects']['{B}'].split('/')[0] if args.task == 'place_empty_cup' else 'box']
            ep = rollout(task, config, reg, run, wrappers[condition], definitions, names, args)
            episodes.append(ep)
            replace_json(run/'episodes.json', episodes)
            print('EPISODE', args.task, seed, block, condition, ep['success'], ep['executed_actions'], flush=True)
        result = summarize(run)
        result['elapsed_s'] = time.monotonic()-began
        write_json(run/'result.json', result)
        write_json(run/'source_runtime_audit.json', storage.verify_source())
        replace_json(run/'progress.json', dict(phase='finished', interface_passed=result['interface_passed']))
        for condition, v in result['outcomes'].items():
            rate = 'N/A' if v['ratio'] is None else f"{v['ratio']:.1%}"
            print(args.task, condition, f"success {v['successes']}/{v['completed']} rate {rate} censored {v['censored']}", flush=True)
        require(result['interface_passed'], 'Local prescription audit failed')
    except BaseException:
        write_json(run/'failure.json', dict(traceback=traceback.format_exc(), completed_records=len(episodes)))
        raise


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--protocol', required=True, type=Path)
    ap.add_argument('--registry', required=True, type=Path)
    ap.add_argument('--task', required=True, choices=('place_empty_cup','beat_block_hammer'))
    ap.add_argument('--mode', choices=('smoke','full'), default='full')
    ap.add_argument('--block', type=int, choices=(0,1), default=0)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    args.protocol = output_path(args.protocol)
    args.registry = output_path(args.registry)
    require(args.gpu.isdigit(), 'Numeric GPU')
    if args.dry_run:
        execute(args, None)
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['JAX_PLATFORMS'] = 'cpu'
    from .native_entry_storage import NativeEntryStorage
    from .evaluate_structured_low_storage import _terminate
    storage = NativeEntryStorage()
    atexit.register(storage.cleanup)
    signal.signal(signal.SIGTERM, _terminate)
    def timeout(*_):
        raise TimeoutError('Declared 3 hour local prescription process bound')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(10800)
    try:
        execute(args, storage)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
