"""Expert collector v2: stable physical-actor labels; old collector stays frozen."""
import atexit
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback

from .audit import digest
from .paths import new_run, write_json, output_path
from .collect_stacking_experts import options, preflight as old_preflight, collect
from .stacking_training_design import load_design
from .gpu_job_status import publish_job

REVISION = 'stable_actor_registry_v2'


def preflight(a):
    p, source, tokenizer, manifest = old_preflight(a)
    if p.get('collector_revision') != REVISION:
        raise ValueError('Use the v2 collection protocol; old protocol remains immutable')
    manifest['collector_revision'] = REVISION
    return p, source, tokenizer, manifest


def export_recording(task, runtime, task_name, info, seed, index):
    from .capture import export_capture
    import numpy as np
    folder = output_path(Path(task.save_dir))
    write_json(folder/'actor_registry.json', task._cx_actor_registry.metadata())
    try:
        return export_capture(task, runtime, task_name, info, seed, index)
    except BaseException:
        # Dense commands otherwise only exist in RAM. Preserve them for CPU repair,
        # without fabricating commands from sparse recorded observation frames.
        try:
            dense = task._cx_dense
            np.savez_compressed(folder/'dense_trace_on_export_failure.npz',
                tick_before=np.asarray([v[0] for v in dense], np.int64),
                dt_s=np.asarray([v[1] for v in dense], np.float64),
                command=np.stack([v[2] for v in dense]))
        except Exception as backup_error:
            print('Dense backup failed; retaining original export error:', backup_error, file=sys.stderr)
        raise


def execute(a, p, source, tokenizer, run):
    from .native_entry_storage import NativeEntryStorage
    from .collect_smoke import make_config
    from .recording import instrument_runtime
    from .stacking_recording import install_recording
    from .stacking_training_data import qualify_capture
    storage = NativeEntryStorage(runtime_source=source)
    atexit.register(storage.cleanup)
    runtime = storage.prepare_runtime(run)
    original = digest(runtime/'envs/_base_task.py')
    count = instrument_runtime(runtime)
    write_json(run/'instrumentation.json', dict(original_sha256=original,
        instrumented_sha256=digest(runtime/'envs/_base_task.py'), counted_step_calls=count,
        recorder_revision=REVISION))
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
    result = collect(task, config, a, p, run, seed_all, UnStableError, export_recording, qualify_capture)
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
    manifest.update(protocol=str(a.protocol), protocol_sha256=digest(a.protocol))
    write_json(run/'manifest.json', manifest)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    publish_job(run, 'initializing', ready=False)
    from .evaluate_stacking import Tee
    old_out, old_err = sys.stdout, sys.stderr
    try:
        with (run/'console.log').open('x') as log:
            sys.stdout, sys.stderr = Tee(old_out, log), Tee(old_err, log)
            try:
                result = execute(a, p, source, tokenizer, run)
                write_json(run/'result.json', result)
                if result['status'] != 'collection_complete':
                    publish_job(run, 'failed', ready=False, reason=result['status'])
                    return 2
                from .stacking_training_data import audit_collection
                write_json(run/'data_audit.json', audit_collection(run, a.protocol))
                publish_job(run, 'finished', ready=False, result=str(run/'result.json'))
                print('COLLECTION COMPLETE', run, flush=True)
                return 0
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
