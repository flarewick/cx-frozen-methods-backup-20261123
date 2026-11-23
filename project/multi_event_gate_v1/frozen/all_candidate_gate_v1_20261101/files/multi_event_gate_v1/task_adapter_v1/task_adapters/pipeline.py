"""One command: smoke -> demonstrations -> frozen-VLA cache -> adapter -> audit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from cx_tta.paths import output_path,replace_json
from .protocol import SIX,prepare,load,tasks_arg


def options(argv=None):
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--tasks','--task',help='Comma-separated official names; default six new tasks')
    ap.add_argument('--run',type=Path,help='Explicit immutable dataset directory for resume')
    ap.add_argument('--phase',choices=('pipeline','prepare','status','smoke','collect','training','cache','train','audit'),default='pipeline')
    ap.add_argument('--task-config',choices=('demo_randomized','demo_clean'),default=None)
    ap.add_argument('--train-scenes',type=int,default=None);ap.add_argument('--val-scenes',type=int,default=None)
    ap.add_argument('--updates',type=int,default=None);ap.add_argument('--gpu',default='0')
    ap.add_argument('--min-free-gb',type=int,default=10);ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args(argv)
    if not a.gpu.isdigit() or a.min_free_gb<1:ap.error('Numeric GPU and positive disk bound required')
    return a


def resolve(a):
    if a.run:
        run=output_path(a.run.resolve());p=load(run)
        if a.tasks is not None and tasks_arg(a.tasks)!=p['tasks']:raise ValueError('--run task set differs')
        if a.task_config is not None and a.task_config!=p['task_config']:raise ValueError('--run config differs')
        for actual,expected in ((a.train_scenes,p['splits']['adapter_fit']['episodes']),
                                (a.val_scenes,p['splits']['adapter_validation']['episodes']),(a.updates,p['updates'])):
            if actual is not None and actual!=expected:raise ValueError('--run budget differs')
        return run,p
    return prepare(tasks_arg(a.tasks or ','.join(SIX)),a.task_config or 'demo_randomized',
                   a.train_scenes if a.train_scenes is not None else 40,
                   a.val_scenes if a.val_scenes is not None else 10,
                   a.updates if a.updates is not None else 1400)


def status(run,p):
    from .collection import resume_state
    rows=[]
    for task in p['tasks']:
        for split,spec in p['splits'].items():
            folder=run/'collections'/task/split
            state,_=resume_state(folder,p,task,split,decode=False)
            rows.append(dict(task=task,split=split,accepted=state['accepted'],requested=spec['episodes']))
    from cx_tta.collection_jobs import read
    return dict(run=str(run),collections=rows,training=read(run/'training/training_progress.json'),
                audit=read(run/'training/training_audit.json'),gpu_started=False)


def main(argv=None):
    a=options(argv);run,p=resolve(a)
    print('DATASET_RUN',run,flush=True)
    if a.dry_run or a.phase in ('prepare','status'):
        print(json.dumps(status(run,p),ensure_ascii=False,indent=2),flush=True);return
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    with (run/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def call(module,*args):
            replace_json(run/'pipeline_status.json',dict(state='running',module=module,args=list(map(str,args))))
            subprocess.run([sys.executable,'-B','-m',module,*map(str,args)],check=True)
        try:
            if a.phase in ('pipeline','smoke','collect'):
                selected=('smoke',) if a.phase=='smoke' else ('smoke','adapter_fit','adapter_validation')
                # Every task must pass a real independent smoke before formal collection.
                for split in selected:
                    for task in p['tasks']:
                        call('task_adapters.collection','--run',run,'--task',task,'--split',split,
                             '--gpu',a.gpu,'--min-free-gb',a.min_free_gb)
            if a.phase in ('pipeline','training','cache','train','audit'):
                call('task_adapters.training','prepare','--run',run,'--gpu',a.gpu)
                phases=('cache','train','audit') if a.phase in ('pipeline','training') else (a.phase,)
                for phase in phases:
                    call('task_adapters.training',phase,'--run',run/'training','--gpu',a.gpu)
            replace_json(run/'pipeline_status.json',dict(state='phase_complete',phase=a.phase,
                checkpoint=str(run/'training/adapter_seed0.pt'),training_audited=(run/'training/training_audit.json').exists()))
            print('PHASE COMPLETE',a.phase,run,flush=True)
        except BaseException:
            replace_json(run/'pipeline_status.json',dict(state='interrupted_or_failed',phase=a.phase,traceback=traceback.format_exc()))
            raise


if __name__=='__main__':main()
