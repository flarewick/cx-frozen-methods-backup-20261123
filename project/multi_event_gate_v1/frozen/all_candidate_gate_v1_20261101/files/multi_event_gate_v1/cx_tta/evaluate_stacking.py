"""Stacking-ready frozen evaluator with per-worker readiness and completion records.

Use --dry-run for dependency checks; no GPU/model imports in that path.
"""
import argparse
import atexit
import gzip
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid

from .audit import digest
from .official_eval_console import episode_output, evaluate_stream
from .paths import ROOT, WORKSPACE_ROOT, new_run, write_json, replace_json
from .multi_event import EventConfig
from .stacking_support import task_contract
from .gpu_job_status import publish_job
from .expert_screening import run_expert_plan, ExpertPlanningFailure, planning_failure_record

CONDITIONS = ('Z_zero', 'F_first_full', 'G_candidate_cost2', 'G_observation_cost2', 'M_shadow', 'M_first_full', 'M_event_full', 'M_legacy_candidate', 'M_legacy_observation')
TASKS = ('beat_block_hammer', 'place_empty_cup', 'place_shoe', 'turn_switch', 'stack_bowls_two', 'lift_pot', 'stack_blocks_two')
REFERENCE = ROOT / 'outputs/reports/protocols/shared_onset_expansion_20261007_145526_085093/protocol.json'
OFFICIAL_ROOT = ROOT.parent.parent / 'qfw/robotwin/RoboTwin'
OFFICIAL_CONSOLE = OFFICIAL_ROOT / 'script/eval_policy_client.py'


def read(path):
    return json.loads(Path(path).read_text())


def options(argv=None):
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--task', required=True, choices=TASKS)
    ap.add_argument('--condition', '--cond', choices=CONDITIONS, default='Z_zero')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--episodes', type=int, default=100, help='Number of valid completed scenes, not attempted seeds')
    starts = ap.add_mutually_exclusive_group()
    starts.add_argument('--seed', type=int, default=0, help='Official seed group; first scene = 100000*(1+seed)')
    starts.add_argument('--seed-start', type=int, help='Explicit first scene seed, overriding the official group formula')
    ap.add_argument('--max-attempts', type=int, default=1000)
    ap.add_argument('--task-config', choices=('demo_randomized',), default='demo_randomized')
    ap.add_argument('--instruction-type', choices=('seen', 'unseen'), default='unseen')
    ap.add_argument('--tokenizer-path', type=Path)
    ap.add_argument('--no-color', action='store_true')
    ap.add_argument('--arm-domain', choices=('official', 'strict'), default='official',
                    help='Official native execution with domain diagnostics, or the old diagnostic abort rule')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--preview-output', action='store_true', help='Print labeled formatting examples; no evaluation')
    ap.add_argument('--smoke-queries', type=int, help='Interface-only truncation, requires --episodes 1; excluded from success rates')
    ap.add_argument('--event-config', type=Path, default=WORKSPACE_ROOT/'configs/events.json')
    ap.add_argument('--event-cache-limit', type=int, default=6)
    args = ap.parse_args(argv)
    if not 0 <= args.event_cache_limit <= 100: ap.error('Event cache limit must be 0..100')
    if args.arm_domain != 'official': ap.error('New multi-event entry uses the official native action boundary')
    if args.seed_start is None:
        args.seed_start = 100000 * (1 + args.seed)
    if not args.gpu.isdigit() or args.seed < 0 or args.seed_start < 0:
        ap.error('Nonnegative seed and numeric GPU required')
    if not 1 <= args.episodes <= args.max_attempts <= 10000:
        ap.error('Require 1 <= episodes <= max-attempts <= 10000')
    if args.smoke_queries is not None and (args.episodes != 1 or not 1 <= args.smoke_queries <= 10):
        ap.error('--smoke-queries requires --episodes 1 and a 1..10 query limit')
    return args


def preflight(args):
    p = read(REFERENCE)
    # Keep audited actor/model dependencies frozen without editing the old protocol.
    for name, expected in p['source_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError('Frozen dependency changed: ' + name)
    ref = Path(p['reference'].get(args.task, p['reference']['beat_block_hammer'])['run'])
    runtime = ref / 'robotwin'
    tokenizer = args.tokenizer_path or ROOT / 'outputs/base_cache_20260926_024808_872087/cache/openpi/big_vision/paligemma_tokenizer.model'
    tokenizer = tokenizer.expanduser().resolve()
    required = [OFFICIAL_CONSOLE, tokenizer, runtime/'envs'/f'{args.task}.py',
                runtime/'description/utils/generate_episode_instructions.py',
                Path(p['training_run'])/'A_bounded_clipped_seed0.pt',
                Path(p['gate_run'])/'observation_base_four_class/model.npz',
                Path(p['gate_run'])/'candidate_conditioned_four_class/model.npz']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    event_config = EventConfig(**read(args.event_config))
    manifest = dict(
        schema_version=2, task=args.task, condition=args.condition, gpu=args.gpu,
        arm_domain=args.arm_domain,
        arm_domain_behavior='Observe finite out-of-URDF goals without clipping; native executor handles them' if args.arm_domain == 'official' else 'Abort on finite out-of-URDF arm targets (legacy diagnostic rule)',
        episodes=args.episodes, count_semantics='100 valid completed scenes, success OR failure',
        seed_start=args.seed_start, max_attempts=args.max_attempts,
        task_config=args.task_config, instruction_type=args.instruction_type,
        screening='Official expert plan_success AND check_success; invalid seeds advance; accepted index unchanged',
        instruction='Official generate_episode_descriptions + numpy.random.choice after scene reset; Python RNG additionally seeded per scene',
        rollout='compiled/internal model noise; 32 actions per query; get_obs after each action; original native budget',
        condition_design='One condition per process; independent processes are NOT numerically paired trajectories',
        warmup='Existing zero-vs-L0 steady probe before first policy episode; no warmup actions or results counted',
        error_policy='UnStableError or verified missing-grasp-target expert failure skipped; infrastructure and rollout errors abort',
        training_run=p['training_run'], gate_run=p['gate_run'], runtime_source=str(runtime),
        frozen_reference=str(REFERENCE), frozen_reference_sha256=digest(REFERENCE),
        official_output_source=str(OFFICIAL_CONSOLE), official_output_sha256=digest(OFFICIAL_CONSOLE),
        source_sha256={str(f):digest(f) for f in required + [Path(__file__), ROOT/'cx_tta/official_eval_console.py', ROOT/'cx_tta/official_onset_domain.py']},
        storage='No videos/RGB/full feature sequences; gzipped query metadata, onset packets, outcomes/provenance; private /tmp compiler scratch',
        smoke_queries=args.smoke_queries, training_approved=False)
    extra,assets,budget = task_contract(runtime,args.task)
    manifest['source_sha256'].update({str(f):digest(f) for f in extra})
    manifest.update(asset_preflight=[str(f) for f in assets],official_task_action_budget=budget)
    manifest['source_sha256'].update({str(f):digest(f) for f in (WORKSPACE_ROOT/'cx_tta').glob('*.py')})
    manifest['source_sha256'].update({str(f):digest(f) for f in [args.event_config,WORKSPACE_ROOT/'scripts/eval_stacking.sh']})
    manifest.update(entry='isolated_stacking_v1', event_config=event_config.to_dict(),
        event_cache_limit=args.event_cache_limit, data_role='EVAL_diagnostic_not_training_labels',
        old_gate_scope='First-onset training only; multi-event legacy modes are diagnostic extrapolation',
        new_multievent_router_trained=False, workspace=str(WORKSPACE_ROOT))
    return p,ref,tokenizer,manifest


class Tee:
    """Non-owning stream proxy; retained logging handlers can outlive the log file.

    close() only detaches the mirrored file. The with-block owns that file and
    Python owns stdout/stderr. Late atexit logging still goes to the terminal.
    """
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file
        self._detached = False
    def write(self, text):
        self.terminal.write(text)
        if not self._detached and not self.file.closed:
            self.file.write(text)
        return len(text)
    def flush(self):
        if not self.terminal.closed:
            self.terminal.flush()
        if not self._detached and not self.file.closed:
            self.file.flush()
    def close(self):
        if not self._detached:
            try:
                self.flush()
            finally:
                self._detached = True
    def isatty(self): return self.terminal.isatty()
    def fileno(self): return self.terminal.fileno()
    def writable(self): return self.terminal.writable()
    @property
    def closed(self): return self.terminal.closed
    @property
    def encoding(self): return self.terminal.encoding
    @property
    def errors(self): return self.terminal.errors


def config_output(config, color):
    import re
    lines = ['============= Config =============\n']
    dr = config['domain_randomization']
    for title, key in [('Messy Table','cluttered_table'), ('Random Background','random_background'),
                       ('Random Light','random_light'), ('Random Table Height','random_table_height'),
                       ('Random Head Camera Distance','random_head_camera_dis')]:
        lines.append(f'\033[95m{title}:\033[0m {dr[key]}')
        if key == 'random_background' and dr[key]: lines.append(f" - Clean Background Rate: {dr['clean_background_rate']}")
        if key == 'random_light' and dr[key]: lines.append(f" - Crazy Random Light Rate: {dr['crazy_random_light_rate']}")
    for title, key in [('Head Camera','head_camera'), ('Wrist Camera','wrist_camera')]:
        lines.append(f"\033[94m{title} Config:\033[0m {config['camera'][key+'_type']}, {config['camera']['collect_'+key]}")
    lines.extend([f"\033[94mEmbodiment Config:\033[0m {config['embodiment_name']}", '\n=================================='])
    text = '\n'.join(lines)
    print(text if color else re.sub(r'\x1b\[[0-9;]*m', '', text), flush=True)


def execute(args, p, ref, tokenizer, manifest, storage, run):
    import numpy as np
    import torch
    from .collect_smoke import make_config
    from .evaluate_b2 import seed_all, configure_inference, step_chunk_official
    from .cache_expert import create_base_policy
    from .evaluate_structured_action import check_training_policy_identity
    from .train_output_support import BASE
    from .multi_event_policy import MultiEventPolicy
    from .official_onset_domain import policy_classes
    ArmPrescriptionPolicy, SharedOnsetPolicy, OutputSupportPolicy = policy_classes(args.arm_domain)
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
    identity = check_training_policy_identity(BASE, tokenizer)
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
            if args.condition.startswith('M_'):
                w = MultiEventPolicy(base, Path(p['training_run']), args.condition, Path(p['gate_run']), EventConfig(**manifest['event_config']))
            else:
                w = (SharedOnsetPolicy(base, Path(p['training_run']), args.condition, Path(p['gate_run']))
                     if args.condition.startswith('G_') else ArmPrescriptionPolicy(base, Path(p['training_run']), args.condition))
            zero = ArmPrescriptionPolicy(base, Path(p['training_run']), 'Z_zero')
            l0 = OutputSupportPolicy(base, Path(p['training_run']), 'L0')
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
                    if args.condition.startswith('M_'):
                        if w.last_event_packet is not None and event_packets < args.event_cache_limit:
                            np.savez_compressed(run/'onsets'/(stem+f'_query{w.query_count-1:03d}.npz'),
                                **w.last_event_packet, applied_delta=w.last_numeric['applied_delta'], actions=w.last_numeric['actions'])
                            event_packets += 1
                            onset_saved = True
                    else:
                        window = d['gate_decision']['window']
                        if not onset_saved and window is not None and w.query_count-1 == window[0]:
                            delta = w.last_numeric['raw_delta'].copy(); delta[:, [6,13]] = 0
                            np.savez_compressed(run/'onsets'/(stem+'.npz'), **w.last_legal_inputs,
                                                candidate_delta=delta[None], applied_delta=w.last_numeric['applied_delta'],
                                                actions=w.last_numeric['actions'])
                            onset_saved = True
                    replace_json(run/'progress.json', dict(phase='rollout', seed=seed, completed=len(episodes),
                        requested=args.episodes, query=w.query_count, native_count=int(task.take_action_cnt)))
            complete = args.smoke_queries is None and (bool(task.eval_success) or task.take_action_cnt == task.step_lim)
            return dict(seed=seed, episode_index=index, condition=args.condition, eligible=True, complete=complete,
                success=bool(task.eval_success) if complete else None, observed_success=bool(task.eval_success),
                instruction=prompt, initial_observation_signature=initial, initial_physical_sha256=row['initial_physical_sha256'],
                executed_actions=int(task.take_action_cnt), queries=w.query_count, native_limit=int(task.step_lim),
                active_chunks=active_chunks, onset_saved=onset_saved, event_packets=event_packets, elapsed_s=time.monotonic()-began,
                reason=None if complete else 'bounded_interface_only_no_success_claim')
        finally:
            # Same accepted-scene cache-clear schedule as the official loop.
            task.close_env(clear_cache=((index+2) % config['clear_cache_freq'] == 0))

    def update(a, e):
        attempts[:], episodes[:] = a, e
        replace_json(run/'screening.json', attempts)
        replace_json(run/'episodes.json', episodes)
        done = [ep for ep in episodes if ep['complete']]
        publish_job(run, 'evaluating' if done else 'warming', ready=bool(done),
            task=args.task, condition=args.condition, requested=args.episodes,
            completed=len(done), successes=sum(ep['success'] for ep in done),
            last_episode=done[-1] if done else None)

    def announce(ep, successes, completed):
        task.suc, task.test_num = successes, completed
        print(episode_output(args.task, policy_name, args.task_config, 'A_bounded_clipped_seed0',
                            ep['success'], successes, completed, ep['seed'], not args.no_color), end='', flush=True)

    result = evaluate_stream(args.seed_start, args.episodes, args.max_attempts, screen, rollout, update, announce)
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
    args = options(argv)
    p, ref, tokenizer, manifest = preflight(args)
    if args.dry_run or args.preview_output:
        print(json.dumps(dict(manifest, validation='CPU dependencies only; GPU rollout not executed'), ensure_ascii=False, indent=2))
        if args.preview_output:
            print('FORMAT PREVIEW ONLY — these are not evaluation outcomes')
            for ok, suc, total in [(True,1,1), (False,1,2)]:
                print(episode_output(args.task, 'cx_new_direction_guide_TTA/'+args.condition, args.task_config,
                    'A_bounded_clipped_seed0', ok, suc, total, args.seed_start+total-1, not args.no_color), end='')
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, JAX_PLATFORMS='cpu')
    from .native_entry_storage import NativeEntryStorage
    from .evaluate_structured_low_storage import _terminate
    storage = NativeEntryStorage(runtime_source=ref/'robotwin')
    atexit.register(storage.cleanup)
    signal.signal(signal.SIGTERM, _terminate)
    run = new_run('eval', group=('stacking_eval', args.task, args.condition))
    for name in ('queries', 'onsets'): (run/name).mkdir()
    manifest.update(process_instance_id=str(uuid.uuid4()), pid=os.getpid(), run=str(run))
    manifest['parallel_worker'] = dict(job_id=os.environ.get('CX_GPU_JOB_ID'),
        managed=bool(os.environ.get('CX_GPU_JOB_STATUS')),
        torch_compile_threads=os.environ.get('TORCHINDUCTOR_COMPILE_THREADS'),
        comparison_scope='Independent processes; no exact paired-trajectory claim')
    write_json(run/'manifest.json', manifest)
    publish_job(run, 'initializing', ready=False, task=args.task, condition=args.condition, requested=args.episodes, completed=0)
    original_out, original_err = sys.stdout, sys.stderr
    with (run/'terminal.log').open('w', buffering=1) as log:
        tee_out, tee_err = Tee(original_out,log), Tee(original_err,log)
        sys.stdout, sys.stderr = tee_out, tee_err
        try:
            print('RUN', run, flush=True)
            execute(args, p, ref, tokenizer, manifest, storage, run)
            publish_job(run, 'finished', ready=True, result=str(run/'result.json'),
                last_episode=read(run/'episodes.json')[-1], **read(run/'result.json'))
        except BaseException:
            failure = traceback.format_exc()
            # Persist the FIRST exception in terminal.log, before atexit handlers.
            print(failure, file=sys.stderr, flush=True)
            try:
                publish_job(run, 'failed', ready=False, error=failure)
            except (OSError,ValueError) as telemetry_error:
                print('Worker status write failed: '+str(telemetry_error),file=sys.stderr,flush=True)
            write_json(run/'failure.json', dict(complete=False, traceback=failure,
                note='Interrupted/errors are not skipped or counted as completed trials; no resume/splicing'))
            raise
        finally:
            sys.stdout, sys.stderr = original_out, original_err
            tee_out.close()
            tee_err.close()


if __name__ == '__main__':
    main()
