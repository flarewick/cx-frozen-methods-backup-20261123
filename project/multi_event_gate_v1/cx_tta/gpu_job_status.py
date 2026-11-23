"""Optional small CPU-side heartbeat; no CUDA calls, threads or sampler RNG."""
import os
import time
from pathlib import Path
from .paths import output_path, replace_json


def publish_job(run, phase, **fields):
    target = os.environ.get('CX_GPU_JOB_STATUS')
    if not target:
        return
    path = output_path(Path(target))
    if path.name != 'worker_status.json':
        raise ValueError('Unexpected scheduler status filename')
    replace_json(path, dict(schema=1, job_id=os.environ.get('CX_GPU_JOB_ID'),
        run=str(output_path(Path(run))), pid=os.getpid(), phase=phase,
        updated_at_epoch=time.time(), **fields))
