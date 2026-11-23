"""Shared compiled VLA / internal-noise zero-residual attribution control.

This entry does not train. A strict sham gate precedes component experiments.
All simulation measurements are passive offline diagnostics, never policy input.
"""
from __future__ import annotations

import argparse
import atexit
import gzip
import importlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

from .audit import digest
from .paths import ROOT, new_run, output_path, replace_json, write_json

TRAINING = ROOT / 'outputs/effect_learning/structured_action_20261003_095511_385074'
DESIGN = ROOT / 'outputs/reports/analyses/structured_closed_loop_20261004_014129_714285/next_experiments.json'
REFERENCES = {
    'beat_block_hammer': 'triplet_20261003_164610_951290',
    'place_shoe': 'triplet_20261003_164743_303998',
    'turn_switch': 'triplet_20261003_164652_020439',
    'place_empty_cup': 'triplet_20261003_165335_817210',
}


def read_json(path):
    return json.loads(Path(path).read_text())


def pilot_reference(task, training_run):
    """Only reuse sealed L0 eligibility/language, never its success as new control."""
    design = read_json(DESIGN)
    pointer = ROOT / 'outputs/evaluations/structured_action' / task / 'comparisons' / REFERENCES[task] / 'L0.json'
    index = read_json(pointer)
    run = output_path(Path(index['run']))
    for name in ('manifest', 'result'):
        if digest(run / f'{name}.json') != index[f'{name}_sha256']:
            raise ValueError('Sealed L0 pointer changed: ' + name)
    manifest, result = read_json(run / 'manifest.json'), read_json(run / 'result.json')
    if not (manifest['task'] == task and manifest['condition'] == 'L0'
            and manifest['scope'] == 'closed_loop_development' and result['complete']
            and result['errors'] == 0 and manifest['compiled']
            and manifest['rollout_protocol'] == 'official'
            and manifest['noise_source'] == 'model_internal_torch'
            and Path(manifest['training_run']).resolve() == Path(training_run).resolve()):
        raise ValueError('Completed official L0 reference required')
    rows = read_json(run / 'episodes.json')
    records = {r['seed']: r for r in rows}
    if len(records) != len(rows) or set(records) != set(manifest['seeds']):
        raise ValueError('Duplicate/missing reference records')
    selected = sorted(min(r['seed'] for r in rows if r.get('eligible') is True and r.get('success') is outcome)
                      for outcome in (True, False))
    if selected != design['pilot_seeds'][task]:
        raise ValueError('Preregistered baseline-only scene selection differs')
    for seed in selected:
        r = records[seed]
        if not r['complete'] or not isinstance(r.get('instruction'), str) or not r['instruction']:
            raise ValueError('Reference eligibility/instruction incomplete')
    evidence = dict(run=str(run), pointer=str(pointer), pointer_sha256=digest(pointer),
        source_sha256={name: digest(run / name) for name in
                       ('manifest.json', 'result.json', 'episodes.json', 'provenance.json')},
        design=str(DESIGN), design_sha256=digest(DESIGN),
        selection=design['selection'], original_outcomes_reused=False)
    return tuple(selected), records, manifest, evidence


def episode_order(seeds):
    """AB/BA per scene: no condition is always the first cold scene of a pair."""
    return [(seed, repeat, condition) for seed in seeds for repeat in range(2)
            for condition in (('L0', 'S0_zero') if repeat == 0 else ('S0_zero', 'L0'))]


class PhysicalPolicy:
    """Read native state before the sole query; truncate only between full chunks."""
    def __init__(self, policy, task, maximum=None):
        self.policy, self.task, self.maximum = policy, task, maximum
        self.before = None

    def predict(self, observation, instruction, noise, *, enabled):
        from .attribution_physics import physical_snapshot
        from .evaluate_structured_action import SmokeBudgetReached
        if self.maximum is not None and self.policy.query_count >= self.maximum:
            raise SmokeBudgetReached(dict(seed=self.policy.episode_seed, eligible=True,
                complete=False, success=None, observed_native_success=bool(self.task.eval_success),
                reason='interface_smoke_query_budget', instruction=instruction,
                executed_actions=int(self.task.take_action_cnt), inference_chunks=self.policy.query_count))
        self.before = physical_snapshot(self.task)
        if not self.before['usable']:
            raise RuntimeError('Native physical diagnostic coverage incomplete: ' + str(self.before['coverage']['errors']))
        return self.policy.predict(observation, instruction, noise, enabled=enabled)


class CompactTrace:
    """Small float32 arrays + compressed passive metadata; no images/videos."""
    def __init__(self, run, stem, physical_policy):
        import numpy as np
        self.np = np
        self.stem = stem
        self.policy = physical_policy
        self.path = output_path(run / 'traces' / f'{stem}.jsonl.gz')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        self.numeric = {key: [] for key in ('base', 'raw_delta', 'applied_delta', 'state')}
        self.first = None

    def record(self, *, observation, before, after, noise, base, actions, delta, executed,
               inference_s, observation_updates=0):
        from .attribution_physics import physical_snapshot
        from .eval_diagnostics import array_hash, json_hash
        policy = self.policy.policy
        values = policy.last_numeric
        for key in self.numeric:
            source = 'command_state' if key == 'state' else key
            self.numeric[key].append(self.np.asarray(values[source], dtype=self.np.float32).copy())
        native_after = physical_snapshot(self.policy.task)
        if not native_after['usable']:
            raise RuntimeError('Post-chunk native physical coverage incomplete: ' + str(native_after['coverage']['errors']))
        row = dict(policy.last_diagnostics, native_before=self.policy.before,
            native_after=native_after, native_before_sha256=json_hash(self.policy.before),
            native_after_sha256=json_hash(native_after),
            executed=executed, obs_updates=observation_updates, inference_s=inference_s,
            action_chunk_sha256=array_hash(actions), raw_max_abs_correction=float(self.np.abs(values['raw_delta']).max()),
            applied_max_abs_correction=float(self.np.abs(delta).max()))
        if noise is not None or observation_updates != 32:
            raise RuntimeError('Official internal-noise / per-action observation contract violated')
        with gzip.open(self.path, 'at', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        # Checkpoint tiny numeric arrays each query, preserving evidence on interruption.
        target = self.path.with_name(f'{self.stem}.npz')
        temporary = target.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            self.np.savez_compressed(stream, **{key: self.np.stack(value) for key, value in self.numeric.items()})
        temporary.replace(target)
        if self.first is None:
            self.first = row


def steady_probe(task, config, seed, instruction, wrappers, maximum=4):
    """Discard warmup; replay one fixed observation with restored INTERNAL RNG.

    No sampled noise is supplied and no warmup/probe action is executed. The
    simulated scene is closed afterwards; actual rollouts recreate their seeds.
    """
    import numpy as np
    from .eval_diagnostics import array_hash, json_hash
    from .attribution_physics import physical_snapshot
    from .evaluate_b2 import seed_all
    from .recovery_contract import rng_capture, rng_restore, rng_fingerprint
    opts = dict(config, need_plan=True, save_data=False, is_test=True)
    seed_all(seed)
    setup_started = False
    result = None
    try:
        setup_started = True
        task.setup_demo(now_ep_num=0, seed=seed, **opts)
        task.set_instruction(instruction=instruction)
        observation = task.get_obs()
        native = physical_snapshot(task)
        if not native['usable']:
            raise RuntimeError('Warmup physical snapshot incomplete')
        for _ in range(2):
            policy = wrappers['S0_zero']
            policy.reset_history()
            policy.episode_seed = seed
            policy.predict(observation, instruction, None, enabled=True)
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        common = rng_capture()
        results, comparisons = [], []
        for i in range(maximum):
            rng_restore(common)
            condition = ('L0', 'S0_zero')[i % 2]
            policy = wrappers[condition]
            policy.reset_history()
            policy.episode_seed = seed
            actions, delta, base = policy.predict(observation, instruction, None, enabled=condition != 'L0')
            values = dict(policy.last_diagnostics, condition=condition, applied_zero=not bool(np.any(delta)),
                identity=np.array_equal(actions.view(np.uint8), base.view(np.uint8)))
            results.append((base.copy(), values))
            if i:
                reference = results[0][0]
                diff = base.astype(np.float64) - reference.astype(np.float64)
                comparisons.append(dict(index=i, condition=condition,
                    base_exact=array_hash(base) == array_hash(reference), base_max_abs=float(np.abs(diff).max()),
                    base_rms=float(np.sqrt(np.mean(diff ** 2))),
                    rng_before_equal=values['rng_before_query'] == results[0][1]['rng_before_query'],
                    rng_after_equal=values['rng_after_sampler'] == results[0][1]['rng_after_sampler']))
        after = physical_snapshot(task)
        result = dict(passed=all(r['base_exact'] and r['rng_before_equal'] and r['rng_after_equal']
                                    for r in comparisons)
                          and all(v['identity'] and v['applied_zero'] for _, v in results)
                          and json_hash(native) == json_hash(after),
            warmup_queries=2, action_attempts=0, common_rng_sha256=rng_fingerprint(common),
            initial_native_sha256=json_hash(native), final_native_sha256=json_hash(after),
            comparisons=comparisons, queries=[v for _, v in results],
            base_values=np.stack([b for b, _ in results]),
            scope='Within one sampler, fixed observation, restored internal RNG; no task success claim')
        rng_restore(common)
        return result
    finally:
        if setup_started:
            task.close_env(clear_cache=False)
        for policy in wrappers.values():
            policy.reset_history()


def arguments():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--task', choices=tuple(REFERENCES), required=True)
    p.add_argument('--training-run', type=Path, default=TRAINING)
    p.add_argument('--gpu', default='0')
    p.add_argument('--tokenizer-path', type=Path)
    p.add_argument('--max-chunks', type=int, help='1..4: interface smoke ONLY; cannot pass the sham gate')
    p.add_argument('--scene-timeout', type=int, default=1800)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not a.gpu.isdigit() or a.scene_timeout < 60 or (a.max_chunks is not None and not 1 <= a.max_chunks <= 4):
        p.error('Require nonnegative GPU, timeout>=60 and optional 1..4 chunk smoke budget')
    a.training_run = a.training_run.resolve()
    return a


def execute(a, storage):
    import numpy as np
    from .cache_expert import create_base_policy
    from .collect_smoke import make_config
    from .evaluate_b2 import configure_inference, resolve_tokenizer, run_episode
    from .evaluate_structured_action import check_training_policy_identity, SmokeBudgetReached
    from .structured_action_deploy import read_checkpoint
    from .structured_action_attribution_policy import StructuredAttributionPolicy
    from .structured_attribution_analysis import analyze_run
    tokenizer = resolve_tokenizer(a.tokenizer_path)
    checkpoint = a.training_run / 'S0_seed0.pt'
    payload, _ = read_checkpoint(checkpoint, 'S0')
    seeds, reference, ref_manifest, ref_evidence = pilot_reference(a.task, a.training_run)
    identity = check_training_policy_identity(a.training_run, tokenizer)
    manifest = dict(schema_version=1, task=a.task, stage='sham', seeds=list(seeds),
        conditions=['L0', 'S0_zero'], repeats=2, max_chunks=a.max_chunks,
        training_run=str(a.training_run), checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        compiled=True, policy_init_seed=0, rollout_protocol='official', noise_source='model_internal_torch',
        task_config=ref_manifest['task_config'], instruction_type=ref_manifest['instruction_type'],
        no_video=True, original_native_budget=True, scene_timeout=a.scene_timeout,
        shared_base_models=1, shared_compiled_samplers=1,
        warmup='2 discarded no-action queries; 4 fixed-input restored-internal-RNG probes; recreate scenes',
        order=[dict(seed=s, repeat=r, condition=c) for s, r, c in episode_order(seeds)],
        reference=ref_evidence, scope='Reused development attribution; not independent success-rate confirmation',
        stage2='Not started: requires all four completed sham controls accepted. New process requires new same-process baseline.')
    if a.dry_run:
        print(json.dumps(dict(manifest=manifest, training_identity=identity, gpu_started=False,
                             head_parameters=payload['metadata'].get('parameters')), ensure_ascii=False, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU visible in this container. No model, scene or evaluation has started.')
    run = new_run('sham_smoke' if a.max_chunks else 'sham', group=('diagnostics', 'structured_attribution', a.task))
    print('RUN', run, flush=True)
    write_json(run / 'manifest.json', manifest)
    write_json(run / 'training_identity_audit.json', identity)
    rows, started = [], time.monotonic()
    try:
        runtime = storage.prepare_runtime(run)
        storage.configure_runtime(run, runtime, tokenizer)
        a.task_config, a.instruction_type = manifest['task_config'], manifest['instruction_type']
        config = make_config(runtime, a, run)
        config.update(eval_mode=True, save_data=False, eval_video_log=False, eval_video_save_dir=None, render_freq=0)
        write_json(run / 'resolved_config.json', config)
        source = read_json(Path(ref_evidence['run']) / 'provenance.json')
        runtime_sha = {str(f.relative_to(runtime)): digest(f) for f in runtime.rglob('*.py') if '__pycache__' not in f.parts}
        if runtime_sha != source['runtime_sha256']:
            raise ValueError('Native runtime differs from reference L0')
        for path, sha in source['code_sha256'].items():
            if digest(Path(path)) != sha:
                raise ValueError('Frozen original source changed: ' + path)
        code = {str(f): digest(f) for f in (ROOT / 'cx_tta').glob('*.py')}
        write_json(run / 'provenance.json', dict(runtime_sha256=runtime_sha,
            source_sha256=code, identity=identity['live_identity'], policy_source_sha256=identity['policy_source_sha256']))
        task = getattr(importlib.import_module('envs.' + a.task), a.task)()
        configure_inference('compiled', 0)
        torch.set_num_threads(2)
        replace_json(run / 'progress.json', dict(phase='loading_one_shared_base', completed=0, requested=8))
        base = create_base_policy()
        base._model.eval().requires_grad_(False)
        wrappers = {name: StructuredAttributionPolicy(base, checkpoint, name) for name in manifest['conditions']}
        for policy in wrappers.values():
            policy.check_robot_config(config)
            if policy.sampler is not base._sample_actions:
                raise RuntimeError('Compiled sampler identity mismatch')
        write_json(run / 'hardware.json', dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            cuda=torch.version.cuda, compiled=hasattr(base._sample_actions, '_torchdynamo_orig_callable'),
            global_dynamo_disabled=bool(torch._dynamo.config.disable), matmul_precision=torch.get_float32_matmul_precision(),
            shared_sampler=True, sampler_process=os.getpid(), sample_kwargs=base._sample_kwargs))
        def timeout(*_):
            raise TimeoutError('Declared attribution scene/warmup timeout exceeded')
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(a.scene_timeout)
        replace_json(run / 'progress.json', dict(phase='common_warmup_and_fixed_input_probe', completed=0, requested=8))
        probe = steady_probe(task, config, seeds[0], reference[seeds[0]]['instruction'], wrappers)
        np.savez_compressed(run / 'steady_probe.npz', base=probe.pop('base_values'))
        write_json(run / 'steady_probe.json', probe)
        signal.alarm(0)
        if not probe['passed']:
            raise RuntimeError('Fixed-input shared-sampler probe failed; stopped before rollouts')
        for index, (seed, repeat, condition) in enumerate(episode_order(seeds)):
            signal.alarm(a.scene_timeout)
            replace_json(run / 'progress.json', dict(phase='sham_control', completed=len(rows), requested=8,
                seed=seed, repeat=repeat, condition=condition))
            policy = wrappers[condition]
            policy.reset_history()
            policy.episode_seed = seed
            physical = PhysicalPolicy(policy, task, a.max_chunks)
            stem = f'seed{seed}_r{repeat}_{condition}'
            trace = CompactTrace(run, stem, physical)
            try:
                # Same episode index on every recreation of a scene, independent of condition/order.
                row = run_episode(task, config, runtime, seed, list(seeds).index(seed), physical, condition != 'L0',
                    instruction_type=a.instruction_type, instruction_override=reference[seed]['instruction'],
                    trace=trace, rollout_protocol='official')
                row['complete'] = True
                if a.max_chunks:
                    row['observed_native_success'] = row.pop('success')
                    row.update(success=None, complete=False, reason='interface_smoke_no_rate_claim')
            except SmokeBudgetReached as exc:
                row = exc.record
            except Exception as exc:
                row = dict(seed=seed, eligible=True, complete=False, success=None, reason='error',
                           error_type=type(exc).__name__, traceback=traceback.format_exc())
            row.update(repeat=repeat, condition=condition, queries=policy.query_count, trace_stem=stem)
            if trace.first:
                row.update(initial_native_sha256=trace.first['native_before_sha256'],
                           initial_observation_sha256=trace.first['observation_sha256'])
            rows.append(row)
            replace_json(run / 'episodes.json', rows)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            signal.alarm(0)
            if row.get('reason') == 'error':
                break
        summary = analyze_run(run)
        summary.update(elapsed_seconds=time.monotonic() - started,
            numeric_bytes=sum(p.stat().st_size for p in (run / 'traces').glob('*.npz')),
            trace_bytes=sum(p.stat().st_size for p in (run / 'traces').glob('*.jsonl.gz')))
        write_json(run / 'result.json', summary)
        replace_json(run / 'progress.json', dict(phase='finished', gate_passed=summary['gate_passed']))
        for condition in manifest['conditions']:
            completed = [r for r in rows if r['condition'] == condition and r['complete'] and isinstance(r['success'], bool)]
            count = sum(r['success'] for r in completed)
            rate = f'{count / len(completed):.1%}' if completed else 'N/A'
            print(f'{a.task}/{condition}: success {count}/{len(completed)} ({rate}); '
                  'stratified repeated sham diagnostic, not a representative success-rate estimate', flush=True)
        print('SHAM_GATE', summary['gate_passed'], 'RUN', run, flush=True)
        if any(r.get('reason') == 'error' for r in rows):
            raise RuntimeError('Attribution contains error; inspect episodes.json')
    except BaseException:
        if not (run / 'failure.json').exists():
            write_json(run / 'failure.json', dict(status='failed_or_interrupted', completed_records=len(rows),
                traceback=traceback.format_exc()))
        raise
    finally:
        signal.alarm(0)


def main():
    a = arguments()
    # Register scratch cleanup before Torch imports so compiler worker teardown is first.
    from .evaluate_structured_low_storage import PrivateEvalStorage, _terminate
    storage = PrivateEvalStorage()
    if not a.dry_run:
        atexit.register(storage.cleanup)
        signal.signal(signal.SIGTERM, _terminate)
        os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
        os.environ['JAX_PLATFORMS'] = 'cpu'
    execute(a, storage)


if __name__ == '__main__':
    main()
