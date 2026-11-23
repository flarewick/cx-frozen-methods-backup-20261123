"""Delete only an explicitly deferred private scratch after GPU release."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import gpu_doctor
import user_container as container


def scratch_target(scratch, run_tmp):
    scratch = Path(scratch)
    if scratch.parent != Path('/tmp') or not re.fullmatch(r'cxsa_[A-Za-z0-9_]+', scratch.name):
        raise ValueError('Not an evaluator private temporary directory')
    root = Path(run_tmp).resolve()
    target = root/scratch.name
    if target.is_symlink() or target.resolve().parent != root:
        raise ValueError('Unsafe scratch target')
    if target.exists() and (not target.is_dir() or target.stat().st_uid != os.getuid()):
        raise ValueError('Scratch is not an owned directory')
    return target


def finish(queue, *, output_root=None, run_tmp=None):
    queue=Path(queue)
    output_root=Path(output_root or container.PROJECT/'multi_event_gate_v1/outputs').resolve()
    run_tmp=Path(run_tmp or container.WORK/'run-tmp')
    if queue.is_symlink() or not queue.resolve().is_relative_to(output_root):
        raise ValueError('Cleanup record is outside the evaluation output root')
    if queue.name!='storage_cleanup.json' or not re.fullmatch(r'process_[0-9a-f]{12}',queue.parent.name) or queue.parent.parent.name!='sessions':
        raise ValueError('Cleanup record is not a process-session record')
    if queue.stat().st_uid != os.getuid():
        raise ValueError('Cleanup record belongs to another account')
    record=json.loads(queue.read_text())
    if record.get('phase')=='after_gpu_release' and record.get('removed'):
        return record
    if record.get('phase')!='deferred_to_cpu_launcher' or record.get('removed'):
        raise ValueError('Scratch cleanup was not explicitly deferred')
    for line in gpu_doctor.query(['--query-compute-apps=gpu_uuid,pid']).splitlines():
        parts=line.split(',')
        if len(parts)!=2 or parts[0].strip()!=record['gpu_uuid']:
            continue
        try:
            if Path('/proc/'+parts[1].strip()).stat().st_uid==os.getuid():
                raise RuntimeError('This account still has a compute process on the selected GPU')
        except FileNotFoundError:
            pass
    target=scratch_target(record['scratch'],run_tmp)
    if target.exists():
        shutil.rmtree(target)
    record.update(phase='after_gpu_release',removed=True,completed_unix_ns=time.time_ns())
    descriptor,name=tempfile.mkstemp(prefix='.cleanup_',dir=queue.parent)
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump(record,stream,indent=2);stream.write('\n')
        os.replace(name,queue)
    finally:
        Path(name).unlink(missing_ok=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('queue',type=Path)
    args=parser.parse_args()
    print('CPU_SCRATCH_CLEANUP',json.dumps(finish(args.queue)))
