"""New stacking adapter evaluation, official execution and whole-episode continuation."""
import gzip
import importlib
import importlib.util
import json
from pathlib import Path
import time
from .audit import digest
from .paths import write_json,replace_json
from .evaluate_stacking import read,config_output,OFFICIAL_CONSOLE
from .multi_event import EventConfig
from .expert_screening import run_expert_plan,ExpertPlanningFailure,planning_failure_record
from .gpu_job_status import publish_job
from .official_eval_console import episode_output
from .stacking_eval_resume import options,preflight

def execute(args, p, ref, tokenizer, manifest, storage, run):
    import numpy as np
    import torch
    from .collect_smoke import make_config
    from .evaluate_b2 import seed_all, configure_inference, step_chunk_official
    from .cache_expert import create_base_policy
    from .stacking_training import verify_base
    from .stacking_policy import StackingPolicy
    from .evaluate_structured_attribution import steady_probe
    from .collect_recovery import signature
    from .attribution_physics import physical_snapshot
    from .eval_diagnostics import json_hash
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this process; no evaluation started')
    runtime = storage.prepare_runtime(run)
    storage.configure_runtime(run, runtime, tokenizer)
    config = make_config(runtime, args, run)
    config.update(eval_mode=True, save_data=False, eval_video_log=False, eval_video_save_dir=None, render_freq=0)
    write_json(run/'resolved_config.json', config)
    actual = {str(f.relative_to(runtime)):digest(f) for f in runtime.rglob('*.py')}
    if actual != read(ref/'provenance.json')['runtime_sha256']:
        raise ValueError('Runtime differs from audited source')
    identity = verify_base(read(Path(p['training_run'])/'training_plan.json'))
    write_json(run/'provenance.json', dict(runtime_sha256=actual, policy_identity=identity,
               source_sha256=p['source_sha256'], console_source=str(OFFICIAL_CONSOLE)))
    spec = importlib.util.spec_from_file_location('cx_official_instructions', runtime/'description/utils/generate_episode_instructions.py')
    language = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(language)
    configure_inference('compiled', 0)
    torch.set_num_threads(2)
    config_output(config, not args.no_color)
    policy_name = 'cx_new_direction_guide_TTA/' + args.condition
    print(f'\033[34mTask Name: {args.task}\033[0m' if not args.no_color else f'Task Name: {args.task}', flush=True)
    print(f'\033[34mPolicy Name: {policy_name}\033[0m' if not args.no_color else f'Policy Name: {policy_name}', flush=True)
    task = getattr(importlib.import_module('envs.'+args.task), args.task)()
    from envs.utils.create_actor import UnStableError
    task.suc = task.test_num = 0
    wrappers = {}
    episodes, attempts = [], []
    opts = dict(config, need_plan=True, save_data=False, is_test=True)

    def setup(seed, index):
        seed_all(seed)
        task.setup_demo(now_ep_num=index, seed=seed, **opts)
        if task.step_lim != manifest['official_task_action_budget']:
            raise RuntimeError('Native task budget differs from audited configuration')

    def screen(seed, index):
        replace_json(run/'progress.json', dict(phase='expert_screen', seed=seed, completed=len(episodes), requested=args.episodes))
        row = dict(seed=seed, episode_index=index, eligible=False)
        try:
            setup(seed, index)
            native = physical_snapshot(task)
            if not native['usable']: raise RuntimeError('Initial physical signature unavailable')
            row['initial_physical_sha256'] = json_hash(native)
            info = run_expert_plan(task)
            row.update(expert_plan_success=bool(task.plan_success), expert_success=bool(task.check_success()))
            row['eligible'] = row['expert_plan_success'] and row['expert_success']
            if row['eligible']: row['scene_info'] = info['info']
            else: row['reason'] = 'expert_plan_or_success_failed'
        except UnStableError as exc:
            row.update(reason='unstable_scene', error=str(exc))
        except ExpertPlanningFailure as exc:
            row.update(planning_failure_record(exc))
        finally:
            task.close_env()
        if not row['eligible']: print(f"SKIP seed={seed}: {row['reason']} (not counted as policy failure)", flush=True)
        return row

    def instruction(info):
        values = language.generate_episode_descriptions(args.task, [info], args.episodes)
        return str(np.random.choice(values[0][args.instruction_type]))

    def rollout(seed, index, row):
        if not wrappers:
            replace_json(run/'progress.json', dict(phase='loading_shared_base', seed=seed, completed=0, requested=args.episodes))
            base = create_base_policy()
            base._model.eval().requires_grad_(False)
            w = StackingPolicy(base, Path(p['training_run']), args.condition, EventConfig(**manifest['event_config']))
            zero = StackingPolicy(base, Path(p['training_run']), 'Z_zero')
            l0 = StackingPolicy(base, Path(p['training_run']), 'L0')
            for v in (w, zero, l0): v.check_robot_config(config)
            # Discarded warmup prompt is generated with the same official function.
            seed_all(seed)
            warmup_instruction = instruction(row['scene_info'])
            probe = steady_probe(task, config, seed, warmup_instruction, {'L0':l0, 'S0_zero':zero})
            probe.pop('base_values')
            write_json(run/'steady_probe.json', probe)
            if not probe['passed']: raise RuntimeError('Z_zero differs from true L0 in steady probe')
            wrappers['active'] = w
            write_json(run/'hardware.json', dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                       cuda=torch.version.cuda, compiled=hasattr(base._sample_actions, '_torchdynamo_orig_callable')))
        w = wrappers['active']
        w.reset_history(); w.episode_seed = seed
        active_chunks = 0
        onset_saved = False
        event_packets = 0
        began = time.monotonic()
        try:
            setup(seed, index)
            if json_hash(physical_snapshot(task)) != row['initial_physical_sha256']:
                raise RuntimeError('Screen/policy initial physical signatures differ')
            prompt = instruction(row['scene_info'])
            task.set_instruction(instruction=prompt)
            obs = task.get_obs()
            initial = signature(obs)
            stem = f'episode{index:03d}_seed{seed}'
            with gzip.open(run/'queries'/(stem+'.jsonl.gz'), 'wt', encoding='utf-8') as trace:
                while task.take_action_cnt < task.step_lim and not task.eval_success:
                    if args.smoke_queries is not None and w.query_count >= args.smoke_queries: break
                    executed, _, obs = step_chunk_official(task, w, obs, prompt, True)
                    if executed == 0: raise RuntimeError('Policy made no progress')
                    d = w.last_diagnostics
                    active_chunks += int(d['prescription']['active'])
                    limit = w.event_config.max_interventions if args.condition.startswith('M_') else 1
                    if active_chunks > limit: raise RuntimeError('Registered intervention budget exceeded')
                    trace.write(json.dumps(dict(d, executed_actions=executed, native_count=int(task.take_action_cnt)), allow_nan=False)+'\n')
                    trace.flush()
                    if w.last_event_packet is not None and event_packets < args.event_cache_limit:
                        np.savez_compressed(run/'onsets'/(stem+f'_query{w.query_count-1:03d}.npz'),
                            **w.last_event_packet, applied_delta=w.last_numeric['applied_delta'], actions=w.last_numeric['actions'])
                        event_packets += 1
                        onset_saved = True
                    replace_json(run/'progress.json', dict(phase='rollout', seed=seed, completed=len(episodes),
                        requested=args.episodes, query=w.query_count, native_count=int(task.take_action_cnt)))
            complete = args.smoke_queries is None and (bool(task.eval_success) or task.take_action_cnt == task.step_lim)
            return dict(seed=seed, episode_index=index, condition=args.condition, eligible=True, complete=complete,
                success=bool(task.eval_success) if complete else None, observed_success=bool(task.eval_success),
                instruction=prompt, initial_observation_signature=initial, initial_physical_sha256=row['initial_physical_sha256'],
                executed_actions=int(task.take_action_cnt), queries=w.query_count, native_limit=int(task.step_lim),
                active_chunks=active_chunks, onset_saved=onset_saved, event_packets=event_packets, elapsed_s=time.monotonic()-began,
                reason=None if complete else 'bounded_interface_only_no_success_claim',
                process_session=run.name, evidence_sha256={str(f):digest(f) for f in
                    [run/'queries'/(stem+'.jsonl.gz'), *sorted((run/'onsets').glob(stem+'*.npz'))]})
        finally:
            # Same accepted-scene cache-clear schedule as the official loop.
            task.close_env(clear_cache=((index+2) % config['clear_cache_freq'] == 0))

    def update(a, e):
        attempts[:], episodes[:] = a, e
        replace_json(args.dataset_run/'screening.json', attempts)
        replace_json(args.dataset_run/'episodes.json', episodes)
        done = [ep for ep in episodes if ep['complete']]
        publish_job(args.dataset_run, 'evaluating' if len(done)>args.initial_completed else 'warming', ready=len(done)>args.initial_completed,
            task=args.task, condition=args.condition, requested=args.episodes,
            initial_completed=args.initial_completed,
            completed=len(done), successes=sum(ep['success'] for ep in done),
            last_episode=done[-1] if done else None)

    def announce(ep, successes, completed):
        task.suc, task.test_num = successes, completed
        print(episode_output(args.task, policy_name, args.task_config, 'stacking_adapter_seed0',
                            ep['success'], successes, completed, ep['seed'], not args.no_color), end='', flush=True)

    from .evaluation_resume import stream
    result = stream(args.dataset_run, manifest, screen, rollout, update, announce)
    complete = [e for e in episodes if e['complete']]
    result.update(task=args.task, condition=args.condition, successes=sum(e['success'] for e in complete),
        success_rate=sum(e['success'] for e in complete)/len(complete) if complete else None,
        valid_seeds=[e['seed'] for e in episodes], invalid_seeds=[r['seed'] for r in attempts if not r['eligible']],
        policy_errors=0, compiled=True, original_model_unchanged=True, no_video=True,
        censored=sum(not e['complete'] for e in episodes))
    for path, expected in manifest['source_sha256'].items():
        if digest(Path(path)) != expected: raise RuntimeError('Source changed during evaluation: '+path)
    write_json(run/'source_runtime_audit.json', storage.verify_source())
    write_json(run/'result.json', result)
    replace_json(run/'progress.json', dict(phase='finished', **result))
    if not result['complete']: print('SMOKE ONLY: truncated interface run; no official success rate.', flush=True)
    print('RESULT', run/'result.json', flush=True)


def main(argv=None):
    from .stacking_eval_resume import main as entry
    return entry(execute,argv)

if __name__=='__main__':main()
