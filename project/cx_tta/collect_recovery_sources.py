"""Self-contained official L0 source rollouts, not yet teacher recovery labels."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import numpy as np

from .audit import digest
from .paths import ROOT, new_run, write_json, replace_json
from .recovery_source_protocol import TASKS, validate


def storage_check(run, maximum_gib):
    folders = [run]
    if (run / 'compiler_temp.json').is_file():
        folders.append(Path(json.loads((run / 'compiler_temp.json').read_text())['path']))
    size = sum(p.stat().st_size for folder in folders for p in folder.rglob('*')
               if p.is_file() and not p.is_symlink())
    # rglob does not descend the runtime assets symlink.
    if size > maximum_gib * 2**30:
        raise OSError(f'Run and private scratch reached storage bound: {size / 2**30:.3f} GiB')
    return size


class RecordingPolicy:
    """Delegate the original prediction once; record without drawing random numbers."""
    def __init__(self, policy):
        from .deploy import PolicyWithAdapter
        self.wrapper = PolicyWithAdapter(policy)
        self.pending = None

    def predict(self, observation, instruction, noise, *, enabled):
        from .recovery_contract import rng_capture
        if enabled or noise is not None or self.pending is not None:
            raise ValueError('One pending official L0 query required')
        before = rng_capture()
        result = self.wrapper.predict(observation, instruction, None, enabled=False)
        after = rng_capture()
        self.pending = (before, after)
        return result


def recorder(run, seed, wrapper, limit_gib):
    from .eval_diagnostics import EpisodeTrace, array_hash
    from .collect_recovery import rng_json, save_observation
    from .recovery_contract import rng_fingerprint, rng_capture

    class SavedTrace(EpisodeTrace):
        def __init__(self):
            super().__init__(run / 'traces' / f'r0_seed{seed}.jsonl')
            self.folder = run / 'observations' / f'seed_{seed}'
            self.folder.mkdir(parents=True)
            self.entries = []
            self.native_count = 0

        def record(self, **kwargs):
            if wrapper.pending is None:
                raise ValueError('Missing matching policy RNG')
            before, after = wrapper.pending
            prefix = self.folder / f'chunk_{self.count:03d}'
            snapshot = prefix.with_suffix('.npz')
            base_file = prefix.with_suffix('.base.npz')
            rng_file = prefix.with_suffix('.rng.json')
            rng_before_save = rng_fingerprint(rng_capture())
            save_observation(snapshot, kwargs['observation'])
            np.savez_compressed(base_file, base_action=np.asarray(kwargs['base'], np.float32))
            write_json(rng_file, dict(before=rng_json(before), after=rng_json(after)))
            entry = dict(chunk=self.count, native_count=self.native_count,
                         previous_chunk=self.count - 1 if self.count else None,
                         executed=int(kwargs['executed']),
                         base_sha256=array_hash(kwargs['base']),
                         files={str(p.relative_to(run)): digest(p) for p in (snapshot, base_file, rng_file)})
            super().record(**kwargs)
            self.entries.append(entry)
            self.native_count += int(kwargs['executed'])
            replace_json(self.folder / 'index.json', self.entries)
            if rng_before_save != rng_fingerprint(rng_capture()):
                raise RuntimeError('Recording changed global RNG')
            wrapper.pending = None
            storage_check(run, limit_gib)

    return SavedTrace()


def arguments():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--stage', choices=('gate', 'train'), required=True)
    p.add_argument('--gate-run', type=Path)
    p.add_argument('--gpu', default='0')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not a.gpu.isdigit():
        p.error('Nonnegative GPU index required')
    a.protocol = a.protocol.resolve()
    protocol = validate(json.loads(a.protocol.read_text()))
    if a.stage == 'train':
        if a.gate_run is None:
            p.error('Train sources require a verified --gate-run')
        a.gate_run = a.gate_run.resolve()
        gate = json.loads((a.gate_run / 'gate.json').read_text())
        if not gate['passed'] or gate['protocol_sha256'] != digest(a.protocol):
            p.error('Gate did not pass for this exact protocol')
        for path, sha in gate['evidence_sha256'].items():
            if digest(Path(path)) != sha:
                p.error('Changed gate evidence: ' + path)
    return a, protocol


def main():
    args, protocol = arguments()
    seeds = protocol['scenes'][args.task][args.stage]
    job = dict(task=args.task, role=args.stage, seeds=seeds,
               protocol=str(args.protocol), protocol_sha256=digest(args.protocol),
               no_video=True, recovery_training_labels=False)
    if args.dry_run:
        print(json.dumps(job, indent=2)); return
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    from .evaluate_b2 import (prepare_runtime, configure_runtime, resolve_tokenizer,
        write_provenance, configure_inference, run_episode)
    from .collect_smoke import make_config
    from .cache_expert import create_base_policy
    run = new_run('source', group=('diagnostics', 'recovery_sources', args.task))
    print(run, flush=True)
    if shutil.disk_usage(run).free < protocol['storage']['minimum_reported_free_gib'] * 2**30:
        raise OSError('Insufficient reported filesystem free space')
    compat = dict(task=args.task, condition='L0', rollout_protocol='official',
                  inference_mode='compiled', trace=True, repeats=1, legacy_l0=None,
                  task_config='demo_randomized', instruction_type=protocol['instruction_type'],
                  policy_init_seed=protocol['policy_init_seed'], gpu=args.gpu)
    write_json(run / 'manifest.json', dict(**job, args=compat, scope='Recovery source discovery; '
               'development L0 outcomes, not adapter evaluation or verified recovery labels'))
    write_json(run / 'protocol.json', protocol)
    snapshot = run / 'source_snapshot'; snapshot.mkdir()
    for path in protocol['source_sha256']:
        src = Path(path); shutil.copyfile(src, snapshot / src.name)
    records = []
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Per-scene source collection exceeded 1800 seconds')
    signal.signal(signal.SIGALRM, timeout)
    try:
        runtime = prepare_runtime(run)
        tokenizer = resolve_tokenizer()
        configure_runtime(run, runtime, tokenizer)
        signal.alarm(1800)
        write_provenance(run, runtime, tokenizer)
        from types import SimpleNamespace
        config = make_config(runtime, SimpleNamespace(**compat), run)
        config.update(eval_mode=True, save_data=False, eval_video_log=False,
                      eval_video_save_dir=None, render_freq=0)
        write_json(run / 'resolved_config.json', config)
        task = getattr(importlib.import_module(f'envs.{args.task}'), args.task)()
        configure_inference('compiled', protocol['policy_init_seed'])
        base = create_base_policy()
        import torch
        if (not torch.cuda.is_available() or torch._dynamo.config.disable or
                not hasattr(base._sample_actions, '_torchdynamo_orig_callable')):
            raise RuntimeError('Compiled CUDA policy required')
        write_json(run / 'hardware.json', dict(gpu=torch.cuda.get_device_name(0),
            torch=torch.__version__, cuda=torch.version.cuda, compiled=True,
            sample_kwargs=base._sample_kwargs))
        wrapper = RecordingPolicy(base)
        for index, seed in enumerate(seeds):
            signal.alarm(1800)
            replace_json(run / 'progress.json', dict(phase='collecting', seed=seed,
                                                    completed=len(records), total=len(seeds)))
            trace = recorder(run, seed, wrapper, protocol['storage']['maximum_run_gib'])
            try:
                result = run_episode(task, config, runtime, seed, index, wrapper, False,
                    instruction_type=protocol['instruction_type'], trace=trace, rollout_protocol='official')
            except Exception as exc:
                if type(exc).__name__ != 'UnStableError':
                    raise
                # No replacement and no interpretation as a policy failure.
                result = dict(seed=seed, eligible=False, reason='unstable_scene')
            if wrapper.pending is not None:
                raise RuntimeError('Unfinished query; source is incomplete')
            result.update(role=args.stage, saved_query_observations=len(trace.entries))
            records.append(result)
            replace_json(run / 'episodes.json', records)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        eligible = [x for x in records if x['eligible']]
        report = dict(status='complete', attempted_seeds=len(records), eligible_episodes=len(eligible),
            successes=sum(x['success'] for x in eligible),
            failures=sum(not x['success'] for x in eligible),
            success_rate=sum(x['success'] for x in eligible)/len(eligible) if eligible else None,
            saved_observations=sum(x['saved_query_observations'] for x in records),
            seconds=time.monotonic()-started, bytes_run_and_scratch=storage_check(
                run, protocol['storage']['maximum_run_gib']), recovery_labels=0)
        write_json(run / 'result.json', report)
        replace_json(run / 'progress.json', dict(phase='complete', result=report))
        print(json.dumps(report), flush=True)
        from .audit_recovery_sources import audit
        audit_result = audit(run)
        write_json(run / 'source_audit.json', audit_result)
        if not audit_result['passed']:
            raise RuntimeError('Saved source interface audit failed')
        rate = f"{report['success_rate']:.1%}" if eligible else 'N/A'
        print(f"采集完成 | {args.task} | L0成功 {report['successes']}/{len(eligible)} ({rate}) | "
              f"失败 {report['failures']} | 总尝试 {len(records)} | 恢复标签 0", flush=True)
    except BaseException:
        write_json(run / 'failure.json', dict(status='incomplete', traceback=traceback.format_exc()))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
