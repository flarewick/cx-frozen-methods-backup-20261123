"""Official compiled L0/S0/SP closed-loop development screening, without video."""
import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

from .audit import digest
from .cache_expert import CHECKPOINT, RT, create_base_policy
from .collect_smoke import ROBOTWIN, make_config
from .evaluate_b2 import (TASKS, configure_inference, configure_runtime, prepare_runtime,
    read_legacy_l0, resolve_tokenizer, run_episode)
from .paths import ROOT, new_run, output_path, replace_json, write_json
from .structured_action_deploy import StructuredActionPolicy, read_checkpoint

BASE_EVIDENCE = ROOT / 'outputs/feature_caches/fixed_recovery_20261003_031535_304147'


def json_read(path):
    return json.loads(Path(path).read_text())


def base_identity(tokenizer):
    """Reuse the existing full-hash certificate only while its file stat agrees."""
    file = CHECKPOINT / 'model.safetensors'
    stat = file.stat()
    current_stat = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    cached = BASE_EVIDENCE / 'protocol.json'
    audit = BASE_EVIDENCE / 'audit.json'
    if cached.is_file() and audit.is_file():
        p = json_read(cached)
        reusable = (json_read(audit).get('passed') and p.get('checkpoint') == str(CHECKPOINT)
                    and p.get('checkpoint_stat') == current_stat)
    else:
        reusable = False
    if reusable:
        sha = p['identity']['base_checkpoint_sha256']
        evidence = dict(method='Existing audited full SHA256, unchanged size/mtime; not rehashed',
            path=str(cached), sha256=digest(cached), audit_path=str(audit), audit_sha256=digest(audit))
    else:
        sha = digest(file)
        evidence = dict(method='Full SHA256 computed for this process')
    norm = CHECKPOINT / 'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    return dict(checkpoint=str(CHECKPOINT), base_checkpoint_sha256=sha,
        checkpoint_stat=current_stat, normalization_sha256=digest(norm),
        tokenizer_sha256=digest(tokenizer), weight_identity_evidence=evidence)


def check_training_policy_identity(training_run, tokenizer):
    """Check available training cache certificates, preserving historical gaps."""
    prepared_file = Path(training_run) / 'protocol.json'
    prepared = json_read(prepared_file)
    sources = prepared['source_sha256']
    live = base_identity(tokenizer)
    identity_keys = ('base_checkpoint_sha256', 'normalization_sha256', 'tokenizer_sha256')
    evidence = []
    def frozen_json(path):
        path = Path(path).resolve()
        if sources.get(str(path)) != digest(path):
            raise ValueError('Training certificate missing/changed: ' + str(path))
        return json_read(path)
    # Both historical/new recovery bases carry explicit policy identity records.
    recovery = []
    for path in sources:
        if Path(path).name == 'protocol.json':
            value = json_read(path)
            if all(key in value.get('identity', {}) for key in identity_keys):
                value = frozen_json(path)
                if Path(value['checkpoint']).resolve() != CHECKPOINT.resolve():
                    raise ValueError('Recovery policy checkpoint path differs from live policy')
                for key in identity_keys:
                    if value['identity'][key] != live[key]:
                        raise ValueError('Recovery/live identity mismatch: ' + key)
                recovery.append((Path(path), value))
                evidence.append(dict(role='recovery_base', path=path, sha256=sources[path],
                    identity=value['identity'], historical_noise=value.get('noise')))
    if not recovery:
        raise ValueError('No frozen recovery policy identity certificate')
    # The expert manifest is bound through the frozen prefix checkpoint audit,
    # rather than trusting an unsealed sibling manifest merely by its filename.
    checkpoint_audits = [Path(path) for path in sources if Path(path).name == 'checkpoint_audit.json']
    expert_manifests = []
    for audit_file in checkpoint_audits:
        certificate = frozen_json(audit_file)
        reference = Path(certificate['reference'])
        if not reference.is_absolute():
            reference = ROOT / reference
        if (reference.name != 'manifest.json' or not reference.parent.name.startswith('base_cache_')
                or digest(reference) != certificate['reference_sha256']):
            continue
        manifest = json_read(reference)
        used_paths = {str(Path(path).resolve()) for path in sources if Path(path).parent == reference.parent and path.endswith('.npz')}
        if not used_paths:
            continue
        if Path(manifest['checkpoint']).resolve() != CHECKPOINT.resolve():
            raise ValueError('Expert base checkpoint path differs from live policy')
        expert_identity = dict(base_checkpoint_sha256=manifest['checkpoint_sha256'],
            normalization_sha256=manifest['normalization_sha256'], tokenizer_sha256=manifest['tokenizer_sha256'])
        for key in identity_keys:
            if expert_identity[key] != live[key]:
                raise ValueError('Expert/live identity mismatch: ' + key)
        members = {str(Path(row['path']).resolve()): row for row in manifest['files']}
        if not used_paths.issubset(members):
            raise ValueError('Prepared expert cache files not covered by the bound manifest')
        expert_manifests.append(reference)
        evidence.append(dict(role='expert_base', path=str(reference), sha256=digest(reference),
            bound_by=str(audit_file), binding_sha256=sources[str(audit_file)],
            used_cache_files=len(used_paths), identity=expert_identity,
            historical_noise='Explicit NumPy Gaussian; differs from official internal Torch noise'))
    if not expert_manifests:
        raise ValueError('No reliably bound manifest for the cached expert actions')
    # Common available source subset. Missing original action-cache code capture
    # is reported; it cannot be repaired by hashing today's files.
    prefix = str(RT / 'RoboTwin/policy/pi05/src/openpi') + '/'
    register = str(RT / 'pi05_integration/register_motus_config.py')
    policy_sources = {path: sha for path, sha in sources.items()
        if path.endswith('.py') and (path.startswith(prefix) or path == register)}
    required = {str(RT / 'RoboTwin/policy/pi05/src/openpi' / relative) for relative in
        ('models_pytorch/pi0_pytorch.py', 'models_pytorch/gemma_pytorch.py',
         'policies/policy.py', 'policies/policy_config.py', 'transforms.py')}
    required.add(register)
    if not required.issubset(policy_sources):
        raise ValueError('Frozen policy-source common subset is incomplete')
    for path, sha in policy_sources.items():
        if digest(Path(path)) != sha:
            raise ValueError('Live/training policy source changed: ' + path)
    for path, value in recovery:
        common = set(policy_sources).intersection(value.get('source_sha256', {}))
        for source in common:
            if policy_sources[source] != value['source_sha256'][source]:
                raise ValueError('Recovery/prepared policy-source certificate conflict: ' + source)
        evidence.append(dict(role='recovery_policy_source_subset', path=str(path), common_files=len(common)))
    for path in sources:
        if Path(path).name == 'feature_audit.json':
            certificate = frozen_json(path)
            if 'model_code' not in certificate:
                continue
            for row in certificate['model_code']:
                if policy_sources.get(row['path']) != row['sha256']:
                    raise ValueError('Prefix/prepared policy-source certificate conflict')
            for key in ('normalization_sha256', 'tokenizer_sha256'):
                if certificate[key] != live[key]:
                    raise ValueError('Prefix/live identity mismatch: ' + key)
            evidence.append(dict(role='prefix_policy_source_subset', path=path, sha256=sources[path],
                common_files=len(certificate['model_code'])))
    return dict(passed=True, scope='Available common policy identity/source certificates; not historical rollout replay',
        prepared_protocol_sha256=digest(prepared_file), live_identity=live,
        certificates=evidence, policy_source_sha256=policy_sources,
        historical_noise=prepared['historical_noise'],
        limitations=['Expert action-cache manifest seals weights/norm/tokenizer but lacks its original model-code capture.',
            'Historical raw L0 provenance is missing for some anchor/recovery sources; no claim of full replay equivalence.',
            'Expert explicit-noise bases and official internal-noise evaluation differ; the cached training distribution is retained.'])


class SmokeBudgetReached(Exception):
    def __init__(self, record):
        super().__init__('Interface smoke ended at its declared query budget')
        self.record = record


class BoundedSmokePolicy:
    """Stop only before the next query; executed blocks use the official callback."""
    def __init__(self, wrapper, task, seed, maximum):
        self.wrapper, self.task, self.seed, self.maximum = wrapper, task, seed, maximum

    def predict(self, observation, instruction, noise, *, enabled):
        if self.wrapper.query_count >= self.maximum:
            raise SmokeBudgetReached(dict(seed=self.seed, eligible=True, complete=False,
                success=None, observed_native_success=bool(self.task.eval_success),
                reason='interface_smoke_query_budget', instruction=instruction,
                executed_actions=int(self.task.take_action_cnt),
                inference_chunks=self.wrapper.query_count))
        return self.wrapper.predict(observation, instruction, noise, enabled=enabled)


def reference_records(path, task, seeds):
    """Lock eligibility/instruction from the newly executed same-entry L0."""
    path = Path(path).resolve()
    if path.is_file():
        path = Path(json_read(path)['run']).resolve()
    manifest = json_read(path / 'manifest.json')
    if manifest['condition'] != 'L0' or manifest['task'] != task or manifest['scope'] != 'closed_loop_development':
        raise ValueError('Reference must be a completed same-entry L0 development run')
    result = json_read(path / 'result.json')
    if not result['complete']:
        raise ValueError('Incomplete/error L0 cannot supply formal reference eligibility')
    records = {row['seed']: row for row in json_read(path / 'episodes.json')}
    if set(records) != set(seeds):
        raise ValueError('L0 reference seeds differ from requested seeds')
    return path, manifest, records


def arguments():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--condition', choices=('L0', 'S0', 'SP'), required=True)
    p.add_argument('--training-run', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--gpu', default='0')
    p.add_argument('--seed-start', type=int, default=100380)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--policy-init-seed', type=int, default=0)
    p.add_argument('--task-config', choices=('demo_randomized',), default='demo_randomized')
    p.add_argument('--instruction-type', choices=('seen', 'unseen'), default='unseen')
    p.add_argument('--tokenizer-path', type=Path)
    p.add_argument('--legacy-l0', type=Path, help='Lock historical seeds/instructions only; rerun L0 under this entry')
    p.add_argument('--reference-l0', type=Path, help='New-entry L0 folder or run-index JSON to lock eligibility/language')
    p.add_argument('--max-chunks', type=int, help='Interface smoke budget only; produces no success rate claim')
    p.add_argument('--scene-timeout', type=int, default=1800)
    p.add_argument('--run-index', type=Path, help='Write this completed run pointer for triplet aggregation')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not a.gpu.isdigit() or not 1 <= a.episodes <= 20:
        p.error('Nonnegative GPU and 1..20 development episodes required')
    if not 100380 <= a.seed_start <= a.seed_start + a.episodes - 1 <= 100399:
        p.error('This screening entry is restricted to development seeds100380..100399')
    if not 0 <= a.policy_init_seed < 2**32 or a.scene_timeout < 60:
        p.error('Invalid initialization seed or scene timeout')
    if a.max_chunks is not None and (not 1 <= a.max_chunks <= 4 or a.episodes > 2):
        p.error('Bounded interface smoke requires 1..2 scenes and 1..4 chunks')
    if a.condition == 'L0' and a.reference_l0 is not None:
        p.error('L0 cannot reference itself')
    a.training_run = a.training_run.resolve()
    a.checkpoint = (a.checkpoint or a.training_run / f'{"S0" if a.condition == "L0" else a.condition}_seed0.pt').resolve()
    seeds = tuple(range(a.seed_start, a.seed_start + a.episodes))
    try:
        payload, _ = read_checkpoint(a.checkpoint, None if a.condition == 'L0' else a.condition)
        if Path(payload['metadata']['training_run']).resolve() != a.training_run:
            raise ValueError('Checkpoint belongs to another training run')
        a.tokenizer_path = resolve_tokenizer(a.tokenizer_path)
        legacy = read_legacy_l0(a.legacy_l0, a.task) if a.legacy_l0 else None
        if legacy is not None and not set(seeds).issubset(legacy):
            raise ValueError('Historical instruction rows do not cover all requested seeds')
        if not (ROBOTWIN / 'envs' / f'{a.task}.py').is_file():
            raise ValueError('Official task source missing')
        if a.run_index is not None:
            a.run_index = output_path(a.run_index)
            if a.run_index.exists():
                raise ValueError('Run-index output already exists')
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        p.error(str(exc))
    return a, seeds, payload['metadata'], legacy


def main():
    args, seeds, metadata, legacy = arguments()
    training_identity = check_training_policy_identity(args.training_run, args.tokenizer_path)
    if args.dry_run:
        print(json.dumps(dict(task=args.task, condition=args.condition, seeds=seeds,
            checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
            training_run=str(args.training_run), metadata=metadata,
            compiled=True, noise_source='model_internal_torch', observation_update='after_each_action',
            policy_init_seed=args.policy_init_seed, gpu=args.gpu, gpu_started=False,
            max_chunks=args.max_chunks, training_identity=training_identity,
            scope='interface_smoke' if args.max_chunks else 'closed_loop_development'),
            ensure_ascii=False, indent=2))
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['JAX_PLATFORMS'] = 'cpu'
    scope = 'interface_smoke' if args.max_chunks else 'closed_loop_development'
    group = ('diagnostics' if args.max_chunks else 'evaluations', 'structured_action', args.task, args.condition)
    run = new_run('smoke' if args.max_chunks else 'eval', group=group)
    print('RUN', run, flush=True)
    reference = None
    manifest = dict(task=args.task, condition=args.condition, scope=scope, seeds=list(seeds),
        training_run=str(args.training_run), checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
        projection_sha256=metadata['projection_sha256'], robot_config_sha256=metadata['robot_config_sha256'],
        policy_init_seed=args.policy_init_seed, compiled=True, rollout_protocol='official',
        instruction_type=args.instruction_type, task_config=args.task_config, noise_source='model_internal_torch',
        max_chunks=args.max_chunks, scene_timeout=args.scene_timeout, residual_strength=1., no_video=True,
        legacy_l0=str(args.legacy_l0.resolve()) if args.legacy_l0 else None,
        legacy_l0_sha256=digest(args.legacy_l0) if args.legacy_l0 else None,
        legacy_outcomes_reused=False, gpu=args.gpu)
    records = []
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Structured action scene exceeded its declared timeout')
    signal.signal(signal.SIGALRM, timeout)
    try:
        if args.reference_l0 is not None:
            ref_path, ref_manifest, reference = reference_records(args.reference_l0, args.task, seeds)
            for key in ('training_run', 'projection_sha256', 'robot_config_sha256', 'policy_init_seed',
                        'compiled', 'rollout_protocol', 'instruction_type', 'task_config', 'noise_source',
                        'legacy_l0_sha256', 'residual_strength'):
                if manifest[key] != ref_manifest[key]:
                    raise ValueError('L0 reference protocol mismatch: ' + key)
            manifest['reference_l0'] = str(ref_path)
            manifest['reference_l0_result_sha256'] = digest(ref_path / 'result.json')
            manifest['reference_l0_episodes_sha256'] = digest(ref_path / 'episodes.json')
        write_json(run / 'manifest.json', manifest)
        write_json(run / 'training_identity_audit.json', training_identity)
        runtime = prepare_runtime(run)
        configure_runtime(run, runtime, args.tokenizer_path)
        config = make_config(runtime, args, run)
        config.update(eval_mode=True, save_data=False, eval_video_log=False,
            eval_video_save_dir=None, render_freq=0)
        write_json(run / 'resolved_config.json', config)
        code = ['structured_action_adapter.py', 'structured_action_deploy.py', 'evaluate_structured_action.py',
            'cache_joint_prefix.py', 'structured_features.py', 'robot_kinematics.py', 'audit_pose_capacity.py',
            'evaluate_b2.py', 'deploy.py', 'recovery_contract.py', 'geometry_readout.py',
            'structured_readout_probe.py', 'cache_expert.py', 'collect_smoke.py', 'capture.py']
        provenance = dict(**training_identity['live_identity'], code_sha256={str(ROOT / 'cx_tta' / name):
            digest(ROOT / 'cx_tta' / name) for name in code},
            runtime_sha256={str(path.relative_to(runtime)): digest(path)
                for path in runtime.rglob('*.py') if '__pycache__' not in path.parts},
            policy_source_sha256=training_identity['policy_source_sha256'],
            training_identity_audit_sha256=digest(run / 'training_identity_audit.json'),
            task_config_sha256=digest(runtime / 'task_config/demo_randomized.yml'),
            observation_contract='RGB head/right/left; current command vector; arms/pi, raw grippers; legal command FK; '
                'expert-only shared PCA32; real previous32-action query, first query past all0; no GT/task/seed inputs')
        write_json(run / 'provenance.json', provenance)
        if reference is not None:
            ref_provenance = json_read(ref_path / 'provenance.json')
            for key in ('base_checkpoint_sha256', 'checkpoint_stat', 'normalization_sha256', 'tokenizer_sha256',
                        'code_sha256', 'runtime_sha256', 'policy_source_sha256', 'task_config_sha256'):
                if provenance[key] != ref_provenance[key]:
                    raise ValueError('L0 runtime/reference mismatch: ' + key)
        task = getattr(importlib.import_module('envs.' + args.task), args.task)()
        configure_inference('compiled', args.policy_init_seed)
        import torch
        torch.set_num_threads(2)
        base = create_base_policy()
        base._model.eval().requires_grad_(False)
        query_path = run / 'queries.jsonl'
        query_path.touch(exist_ok=False)
        def record_query(value):
            with query_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        wrapper = StructuredActionPolicy(base, args.checkpoint, args.condition, record_query)
        wrapper.check_robot_config(config)
        write_json(run / 'hardware.json', dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            cuda=torch.version.cuda, compiled=hasattr(base._sample_actions, '_torchdynamo_orig_callable'),
            global_dynamo_disabled=bool(torch._dynamo.config.disable), sample_kwargs=base._sample_kwargs,
            matmul_precision=torch.get_float32_matmul_precision(), trainable_backbone_parameters=0,
            trainable_adapter_parameters=0))
        for index, seed in enumerate(seeds):
            signal.alarm(args.scene_timeout)
            replace_json(run / 'progress.json', dict(phase='evaluating', completed=len(records), total=len(seeds), seed=seed))
            wrapper.reset_history()
            wrapper.episode_seed = seed
            instruction = legacy[seed]['instruction'] if legacy else None
            if reference is not None:
                if not reference[seed]['eligible']:
                    row = dict(seed=seed, eligible=False, complete=True, reason='reference_l0_ineligible',
                        reference_reason=reference[seed].get('reason'))
                    records.append(row)
                    replace_json(run / 'episodes.json', records)
                    continue
                instruction = reference[seed]['instruction']
            policy = BoundedSmokePolicy(wrapper, task, seed, args.max_chunks) if args.max_chunks else wrapper
            try:
                row = run_episode(task, config, runtime, seed, index, policy, args.condition != 'L0',
                    instruction_type=args.instruction_type, instruction_override=instruction, rollout_protocol='official')
                row['complete'] = True
                if args.max_chunks:
                    row['observed_native_success'] = row.pop('success', None)
                    row['success'] = None
                    row['scope'] = 'interface_smoke_no_rate_claim'
            except SmokeBudgetReached as exc:
                row = exc.record
            except Exception as exc:
                if type(exc).__name__ == 'UnStableError':
                    row = dict(seed=seed, eligible=False, complete=True, reason='unstable_scene')
                else:
                    row = dict(seed=seed, eligible=None, complete=False, success=None,
                        reason='error', error_type=type(exc).__name__, traceback=traceback.format_exc())
            row.update(condition=args.condition, queries=wrapper.query_count)
            records.append(row)
            replace_json(run / 'episodes.json', records)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if row.get('reason') == 'error':
                break
        signal.alarm(0)
        eligible = [row for row in records if row.get('eligible') is True and row.get('complete') and isinstance(row.get('success'), bool)]
        errors = sum(row.get('reason') == 'error' for row in records)
        invalid = sum(row.get('eligible') is False for row in records)
        success = sum(row['success'] for row in eligible)
        summary = dict(scope=scope, task=args.task, condition=args.condition, attempted=len(records),
            requested=len(seeds), complete=len(records) == len(seeds) and errors == 0 and all(row.get('complete') for row in records),
            completed_eligible=len(eligible), success=success if not args.max_chunks else None,
            success_rate=success / len(eligible) if eligible and not args.max_chunks else None,
            invalid=invalid, errors=errors, censored=sum(not row.get('complete') for row in records),
            elapsed_seconds=time.monotonic() - started)
        write_json(run / 'result.json', summary)
        replace_json(run / 'progress.json', dict(phase='complete' if summary['complete'] else 'incomplete', **summary))
        if args.run_index is not None:
            args.run_index.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.run_index, dict(run=str(run), task=args.task, condition=args.condition,
                result_sha256=digest(run / 'result.json'), manifest_sha256=digest(run / 'manifest.json')))
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if args.max_chunks:
            print(f'{args.task}/{args.condition} interface smoke: no success-rate claim; '
                f'queries={sum(row["queries"] for row in records if "queries" in row)}, invalid={invalid}, errors={errors}', flush=True)
        else:
            rate = f'{100 * success / len(eligible):.1f}%' if eligible else 'N/A'
            print(f'{args.task}/{args.condition}: success {success}/{len(eligible)} ({rate}); '
                f'invalid={invalid}, errors={errors}, attempted={len(records)}/{len(seeds)}', flush=True)
        if errors:
            raise RuntimeError('Structured evaluation contains an error; inspect episodes.json')
    except BaseException:
        if not (run / 'failure.json').exists():
            write_json(run / 'failure.json', dict(status='failed_or_interrupted', traceback=traceback.format_exc(),
                completed_records=len(records)))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
