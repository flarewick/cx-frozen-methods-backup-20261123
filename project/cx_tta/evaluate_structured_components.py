"""Frozen S0 component attribution with a new same-process compiled L0.

All four full zero-residual controls must be accepted before this entry starts.
This is a diagnostic development experiment; it neither trains nor estimates
independent task-level generalization. Native measurements remain passive.
"""
from __future__ import annotations

import argparse
import atexit
import importlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

from .audit import digest
from .paths import ROOT, new_run, output_path, replace_json, write_json
from .evaluate_structured_attribution import (
    CompactTrace, PhysicalPolicy, REFERENCES, TRAINING, pilot_reference,
    read_json, steady_probe,
)


CONDITIONS = ('L0', 'S0', 'S0_quarter', 'S0_no_gripper', 'S0_no_history')
# The acceptance path and its digest are frozen after the four source runs are
# independently audited. An arbitrary passing-looking JSON cannot authorize a
# component run. Both constants are supplied by the acceptance report.
PROTOCOL = ROOT / 'outputs/reports/attribution/sham_accepted_20261004_032834_921398/stage2_protocol.json'
PROTOCOL_SHA256 = '7148310e4c79d7d785ec90f8afe7fcbca655d60642ddf707b227721d6aea779c'


def component_order(seeds):
    """One repeat; reverse all conditions on the second preregistered scene."""
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError('Exactly two distinct preregistered component scenes required')
    return [(seed, 0, condition) for index, seed in enumerate(seeds)
            for condition in (CONDITIONS if index == 0 else tuple(reversed(CONDITIONS)))]


def validate_acceptance(protocol_path, expected_sha256, task, training_run, stage1_run=None):
    """Recheck the sealed source evidence and all four CPU numerical gates."""
    from .structured_attribution_analysis import analyze_run
    protocol_path = output_path(Path(protocol_path))
    if not expected_sha256 or digest(protocol_path) != expected_sha256:
        raise ValueError('Stage-2 acceptance protocol is missing or its sealed SHA256 differs')
    protocol = read_json(protocol_path)
    if (protocol.get('schema_version') != 1 or protocol.get('stage') != 'components'
            or protocol.get('conditions') != list(CONDITIONS) or protocol.get('repeats') != 1
            or set(protocol.get('tasks', {})) != set(REFERENCES)
            or Path(protocol.get('training_run', '')).resolve() != Path(training_run).resolve()
            or protocol.get('all_stage1_gates_passed') is not True):
        raise ValueError('Invalid or incomplete four-task stage-2 preregistration')
    if digest(Path(training_run) / 'S0_seed0.pt') != protocol.get('checkpoint_sha256'):
        raise ValueError('Frozen S0 checkpoint differs from the accepted protocol')
    evidence, checked_sources = {}, {}
    def check_source(source, expected, label):
        path = Path(source).resolve()
        if not path.is_file():
            raise ValueError(label + ' is missing: ' + str(path))
        actual = checked_sources.get(path)
        if actual is None:
            actual = checked_sources[path] = digest(path)
        if actual != expected:
            raise ValueError(label + ' changed: ' + str(path))
    for name, accepted in protocol['tasks'].items():
        run = output_path(Path(accepted['stage1_run']))
        sources = accepted.get('source_sha256')
        if not isinstance(sources, dict) or not sources:
            raise ValueError('Missing sealed sham source inputs: ' + name)
        for source, expected in sources.items():
            check_source(source, expected, 'Accepted sham source')
        provenance = read_json(run / 'provenance.json')
        live_sources, policy_sources = provenance.get('source_sha256'), provenance.get('policy_source_sha256')
        required_live = {str(ROOT / 'cx_tta' / filename) for filename in (
            'structured_action_attribution_policy.py', 'structured_action_deploy.py',
            'structured_action_adapter.py', 'structured_features.py',
            'evaluate_structured_attribution.py', 'attribution_physics.py',
            'recovery_contract.py', 'evaluate_structured_low_storage.py')}
        if (not isinstance(live_sources, dict) or not required_live.issubset(live_sources)
                or not isinstance(policy_sources, dict) or not policy_sources):
            raise ValueError('Complete live wrapper/policy provenance required: ' + name)
        # Validate the files the accepted control actually consumed, not just
        # the immutable JSON that recorded their old digests. New stage-2 files
        # may be added; every previously recorded source must remain identical.
        for source, expected in live_sources.items():
            check_source(source, expected, 'Accepted live project source')
        for source, expected in policy_sources.items():
            check_source(source, expected, 'Accepted live policy source')
        report = analyze_run(run)
        manifest = read_json(run / 'manifest.json')
        accepted_seeds = accepted.get('seeds', protocol.get('seeds', {}).get(name))
        if (report.get('gate_passed') is not True or report.get('stage') != 'sham'
                or report.get('task') != name or manifest['seeds'] != accepted_seeds
                or manifest['training_run'] != str(Path(training_run).resolve())
                or manifest['checkpoint_sha256'] != protocol['checkpoint_sha256']
                or manifest.get('compiled') is not True
                or manifest.get('rollout_protocol') != 'official'
                or manifest.get('noise_source') != 'model_internal_torch'
                or type(manifest.get('policy_init_seed')) is not int or manifest['policy_init_seed'] != 0
                or manifest.get('shared_base_models') != 1 or manifest.get('shared_compiled_samplers') != 1
                or manifest.get('task_config') != 'demo_randomized' or manifest.get('instruction_type') != 'unseen'
                or manifest.get('original_native_budget') is not True or manifest.get('no_video') is not True):
            raise ValueError('Recomputed complete sham gate/protocol failed: ' + name)
        evidence[name] = dict(run=str(run), recomputed_gate_passed=True,
                              sources_checked=len(sources), live_project_sources_checked=len(live_sources),
                              live_policy_sources_checked=len(policy_sources), seeds=manifest['seeds'])
    selected = output_path(Path(protocol['tasks'][task]['stage1_run']))
    if stage1_run is not None and Path(stage1_run).resolve() != selected:
        raise ValueError('--stage1-run must equal the certified run for the selected task')
    return protocol, selected, dict(protocol=str(protocol_path), protocol_sha256=expected_sha256,
                                     all_four_recomputed=True, tasks=evidence)


class InitialPairingGuard:
    """Stop on first-query context or base mismatch; corrected paths may diverge."""
    def __init__(self):
        self.initial = {}
        self.comparisons = {}

    def observe(self, seed, condition, row, base):
        import numpy as np
        from .eval_diagnostics import array_hash
        key = (seed, condition)
        if key in self.initial:
            raise RuntimeError('Duplicate component initial query')
        self.initial[key] = (dict(row), np.asarray(base, np.float32).copy())
        baseline = self.initial.get((seed, 'L0'))
        if baseline is None:
            return
        left, left_base = baseline
        for candidate in CONDITIONS[1:]:
            current = self.initial.get((seed, candidate))
            if current is None or (seed, candidate) in self.comparisons:
                continue
            right, right_base = current
            diff = left_base.astype(np.float64) - right_base.astype(np.float64)
            checks = {field: bool(left.get(field) and left[field] == right.get(field))
                      for field in ('observation_sha256', 'native_before_sha256',
                                    'rng_before_query', 'rng_after_sampler')}
            checks['structured_inputs'] = bool(left.get('structured_input_sha256')
                and left['structured_input_sha256'] == right.get('structured_input_sha256'))
            checks['reset_history'] = bool(left.get('has_history') is False
                and right.get('has_history') is False and left.get('history_query') is None
                and right.get('history_query') is None)
            checks['base_exact'] = array_hash(left_base) == array_hash(right_base)
            result = dict(seed=seed, condition=candidate, checks=checks,
                          max_abs=float(np.abs(diff).max()),
                          rms=float(np.sqrt(np.mean(diff ** 2))), passed=all(checks.values()))
            self.comparisons[(seed, candidate)] = result
            if not result['passed']:
                raise RuntimeError('Component initial query is not paired: ' + json.dumps(result))

    def summary(self, seeds):
        expected = {(seed, condition) for seed in seeds for condition in CONDITIONS[1:]}
        rows = [self.comparisons[key] for key in sorted(self.comparisons)]
        return dict(scope='Initial query only; post-correction divergence is expected',
                    expected_comparisons=len(expected), comparisons=rows,
                    passed=set(self.comparisons) == expected and all(row['passed'] for row in rows))


class PairedCompactTrace(CompactTrace):
    def __init__(self, run, stem, physical_policy, guard, seed, condition):
        super().__init__(run, stem, physical_policy)
        self.guard, self.seed, self.condition = guard, seed, condition

    def record(self, **kwargs):
        was_first = self.first is None
        super().record(**kwargs)
        if was_first:
            self.guard.observe(self.seed, self.condition, self.first, self.numeric['base'][0])


def arguments():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--task', choices=tuple(REFERENCES), required=True)
    parser.add_argument('--training-run', type=Path, default=TRAINING)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--tokenizer-path', type=Path)
    parser.add_argument('--stage1-run', type=Path)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--protocol-sha256', default=PROTOCOL_SHA256)
    parser.add_argument('--max-chunks', type=int, help='1..4: interface smoke only; no efficacy/gate claim')
    parser.add_argument('--scene-timeout', type=int, default=1800)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if (not args.gpu.isdigit() or args.scene_timeout < 60
            or (args.max_chunks is not None and not 1 <= args.max_chunks <= 4)):
        parser.error('Require nonnegative GPU, timeout>=60 and optional 1..4 chunk smoke budget')
    args.training_run = args.training_run.resolve()
    return args


def execute(args, storage):
    import numpy as np
    from .cache_expert import create_base_policy
    from .collect_smoke import make_config
    from .evaluate_b2 import configure_inference, resolve_tokenizer, run_episode
    from .evaluate_structured_action import check_training_policy_identity, SmokeBudgetReached
    from .structured_action_deploy import read_checkpoint
    from .structured_action_attribution_policy import StructuredAttributionPolicy
    from .structured_attribution_analysis import analyze_run

    protocol, stage1_run, acceptance = validate_acceptance(args.protocol, args.protocol_sha256,
        args.task, args.training_run, args.stage1_run)
    tokenizer = resolve_tokenizer(args.tokenizer_path)
    checkpoint = args.training_run / 'S0_seed0.pt'
    payload, _ = read_checkpoint(checkpoint, 'S0')
    seeds, reference, reference_manifest, reference_evidence = pilot_reference(args.task, args.training_run)
    accepted_seeds = protocol['tasks'][args.task].get('seeds', protocol.get('seeds', {}).get(args.task))
    if list(seeds) != accepted_seeds:
        raise ValueError('Accepted stage-1 and baseline-only scene selection differ')
    identity = check_training_policy_identity(args.training_run, tokenizer)
    stage1_manifest = read_json(stage1_run / 'manifest.json')
    manifest = dict(schema_version=1, task=args.task, stage='components', stage1_run=str(stage1_run),
        seeds=list(seeds), conditions=list(CONDITIONS), repeats=1, max_chunks=args.max_chunks,
        training_run=str(args.training_run), checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        compiled=True, policy_init_seed=0, rollout_protocol='official', noise_source='model_internal_torch',
        task_config=reference_manifest['task_config'], instruction_type=reference_manifest['instruction_type'],
        no_video=True, original_native_budget=True, scene_timeout=args.scene_timeout,
        shared_base_models=1, shared_compiled_samplers=1,
        warmup='2 discarded no-action S0_zero queries; 4 fixed-input restored-internal-RNG probes; recreate scenes',
        order=[dict(seed=seed, repeat=repeat, condition=condition)
               for seed, repeat, condition in component_order(seeds)],
        reference=reference_evidence, acceptance=acceptance,
        scope='Frozen single-factor diagnostics on selected development scenes; no training or independent success-rate claim',
        next_step='Inspect same-runtime outcomes and component evidence before changing training')
    for key in ('task', 'seeds', 'training_run', 'checkpoint_sha256', 'task_config', 'instruction_type'):
        if manifest[key] != stage1_manifest[key]:
            raise ValueError('Certified stage-1/component manifest differs: ' + key)
    if args.dry_run:
        print(json.dumps(dict(manifest=manifest, training_identity=identity, gpu_started=False,
            head_parameters=payload['metadata'].get('parameters'), requested_full_rollouts=10),
            ensure_ascii=False, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU visible. No model, scene or component evaluation has started.')
    run = new_run('components_smoke' if args.max_chunks else 'components',
                  group=('diagnostics', 'structured_attribution', args.task))
    print('RUN', run, flush=True)
    write_json(run / 'manifest.json', manifest)
    write_json(run / 'training_identity_audit.json', identity)
    write_json(run / 'attribution_storage_note.json', dict(
        helper='cx_tta.evaluate_structured_low_storage.PrivateEvalStorage',
        queries='ALL executed queries; four compact float32 arrays plus compressed passive metadata',
        images=False, video=False, weight_copies=False,
        caches='private /tmp/cxsa_* per process; removed at exit after compiler workers',
        generic_storage_policy_note='The shared helper query-0 description belongs to the original evaluator only'))
    rows, started, guard = [], time.monotonic(), InitialPairingGuard()
    try:
        runtime = storage.prepare_runtime(run)
        storage.configure_runtime(run, runtime, tokenizer)
        args.task_config, args.instruction_type = manifest['task_config'], manifest['instruction_type']
        config = make_config(runtime, args, run)
        config.update(eval_mode=True, save_data=False, eval_video_log=False,
                      eval_video_save_dir=None, render_freq=0)
        write_json(run / 'resolved_config.json', config)
        source = read_json(Path(reference_evidence['run']) / 'provenance.json')
        runtime_sha = {str(path.relative_to(runtime)): digest(path)
                       for path in runtime.rglob('*.py') if '__pycache__' not in path.parts}
        if runtime_sha != source['runtime_sha256']:
            raise ValueError('Native runtime differs from the sealed reference L0')
        for path, expected in source['code_sha256'].items():
            if digest(Path(path)) != expected:
                raise ValueError('Frozen original source changed: ' + path)
        code = {str(path): digest(path) for path in (ROOT / 'cx_tta').glob('*.py')}
        write_json(run / 'provenance.json', dict(runtime_sha256=runtime_sha, source_sha256=code,
            identity=identity['live_identity'], policy_source_sha256=identity['policy_source_sha256']))
        task = getattr(importlib.import_module('envs.' + args.task), args.task)()
        configure_inference('compiled', 0)
        torch.set_num_threads(2)
        replace_json(run / 'progress.json', dict(phase='loading_one_shared_base', completed=0, requested=10))
        base = create_base_policy()
        base._model.eval().requires_grad_(False)
        # Construct all heads before warmup/episode seeding. S0_zero is probe-only.
        wrappers = {condition: StructuredAttributionPolicy(base, checkpoint, condition)
                    for condition in (*CONDITIONS, 'S0_zero')}
        for wrapper in wrappers.values():
            wrapper.check_robot_config(config)
            if wrapper.sampler is not base._sample_actions:
                raise RuntimeError('Compiled sampler identity mismatch')
        write_json(run / 'hardware.json', dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            cuda=torch.version.cuda, compiled=hasattr(base._sample_actions, '_torchdynamo_orig_callable'),
            global_dynamo_disabled=bool(torch._dynamo.config.disable),
            matmul_precision=torch.get_float32_matmul_precision(), shared_sampler=True,
            sampler_process=os.getpid(), sample_kwargs=base._sample_kwargs))
        def timeout(*_):
            raise TimeoutError('Declared component scene/warmup timeout exceeded')
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.scene_timeout)
        replace_json(run / 'progress.json', dict(phase='common_warmup_and_fixed_input_probe', completed=0, requested=10))
        probe = steady_probe(task, config, seeds[0], reference[seeds[0]]['instruction'],
                             {condition: wrappers[condition] for condition in ('L0', 'S0_zero')})
        np.savez_compressed(run / 'steady_probe.npz', base=probe.pop('base_values'))
        write_json(run / 'steady_probe.json', probe)
        signal.alarm(0)
        if not probe['passed']:
            raise RuntimeError('New-process fixed-input shared-sampler probe failed before rollouts')
        for seed, repeat, condition in component_order(seeds):
            signal.alarm(args.scene_timeout)
            replace_json(run / 'progress.json', dict(phase='frozen_component_diagnostic', completed=len(rows),
                requested=10, seed=seed, repeat=repeat, condition=condition))
            wrapper = wrappers[condition]
            wrapper.reset_history()
            wrapper.episode_seed = seed
            physical = PhysicalPolicy(wrapper, task, args.max_chunks)
            stem = f'seed{seed}_r{repeat}_{condition}'
            trace = PairedCompactTrace(run, stem, physical, guard, seed, condition)
            try:
                row = run_episode(task, config, runtime, seed, list(seeds).index(seed), physical,
                    condition != 'L0', instruction_type=args.instruction_type,
                    instruction_override=reference[seed]['instruction'], trace=trace, rollout_protocol='official')
                row['complete'] = True
                if args.max_chunks:
                    row['observed_native_success'] = row.pop('success')
                    row.update(success=None, complete=False, reason='interface_smoke_no_rate_claim')
            except SmokeBudgetReached as exc:
                row = exc.record
            except Exception as exc:
                row = dict(seed=seed, eligible=True, complete=False, success=None, reason='error',
                           error_type=type(exc).__name__, traceback=traceback.format_exc())
            row.update(repeat=repeat, condition=condition, queries=wrapper.query_count, trace_stem=stem)
            if trace.first:
                row.update(initial_native_sha256=trace.first['native_before_sha256'],
                           initial_observation_sha256=trace.first['observation_sha256'])
            rows.append(row)
            replace_json(run / 'episodes.json', rows)
            replace_json(run / 'initial_pairing.json', guard.summary(seeds))
            print(json.dumps(row, ensure_ascii=False), flush=True)
            signal.alarm(0)
            if row.get('reason') == 'error':
                break
        from .structured_attribution_analysis import _probe
        from .structured_component_metrics import summarize_components
        summary = analyze_run(run)
        # The frozen stage-2 analyzer checks stage 1 and initial pairing. This
        # entry additionally recomputes the NEW process's saved probe numbers.
        summary['steady_probe'] = _probe(run)
        summary.update(elapsed_seconds=time.monotonic() - started, initial_pairing=guard.summary(seeds),
            numeric_bytes=sum(path.stat().st_size for path in (run / 'traces').glob('*.npz')),
            trace_bytes=sum(path.stat().st_size for path in (run / 'traces').glob('*.jsonl.gz')))
        if not summary['initial_pairing']['passed']:
            summary['gate_passed'] = False
            summary['errors'].append('Online first-query pairing incomplete or failed')
        if not summary['steady_probe']['passed']:
            summary['gate_passed'] = False
            summary['errors'].append('New-process fixed-input probe failed numerical reanalysis')
        write_json(run / 'component_metrics.json', summarize_components(run))
        write_json(run / 'result.json', summary)
        replace_json(run / 'progress.json', dict(phase='finished', gate_passed=summary['gate_passed']))
        for condition in CONDITIONS:
            completed = [row for row in rows if row['condition'] == condition
                         and row['complete'] and isinstance(row['success'], bool)]
            successes = sum(row['success'] for row in completed)
            rate = f'{successes / len(completed):.1%}' if completed else 'N/A'
            print(f'{args.task}/{condition}: success {successes}/{len(completed)} ({rate}); '
                  'selected development component diagnostic, not representative task success rate', flush=True)
        print('COMPONENT_GATE', summary['gate_passed'], 'RUN', run, flush=True)
        if any(row.get('reason') == 'error' for row in rows):
            raise RuntimeError('Component experiment contains an error; inspect episodes.json')
    except BaseException:
        if not (run / 'failure.json').exists():
            write_json(run / 'failure.json', dict(status='failed_or_interrupted', completed_records=len(rows),
                                                traceback=traceback.format_exc()))
        raise
    finally:
        signal.alarm(0)


def main():
    args = arguments()
    # Register before Torch imports: compiler workers are shut down before scratch cleanup.
    from .evaluate_structured_low_storage import PrivateEvalStorage, _terminate
    storage = PrivateEvalStorage()
    if not args.dry_run:
        atexit.register(storage.cleanup)
        signal.signal(signal.SIGTERM, _terminate)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        os.environ['JAX_PLATFORMS'] = 'cpu'
    execute(args, storage)


if __name__ == '__main__':
    main()
