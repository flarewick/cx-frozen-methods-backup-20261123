"""Whole-demonstration resume, private runtime, no video and no task-specific policy."""
import argparse
import atexit
import fcntl
import importlib
import os
from pathlib import Path
import signal
import sys
import traceback
import uuid

from cx_tta.audit import digest
from cx_tta.paths import output_path,write_json,replace_json
from cx_tta.collection_jobs import read
from cx_tta.collection_resume import EXCLUDED
from .protocol import load


def resume_state(folder, p, task, split, decode=True):
    spec=p['splits'][split]; sha=digest(folder.parents[2]/'protocol.json')
    rows=read(folder/'attempts.json') if (folder/'attempts.json').exists() else []
    accepted=[];terminal=set()
    for r in rows:
        seed=r.get('seed')
        if type(seed) is not int or not spec['start']<=seed<spec['stop'] or r.get('task')!=task or r.get('split')!=split:
            raise ValueError('Collection identity or seed window changed')
        if r.get('accepted'):
            if seed in terminal or r.get('episode_index')!=len(accepted) or not r.get('plan_success') or not r.get('replay_success'):
                raise ValueError('Duplicate or unqualified accepted scene')
            f=output_path(Path(r['hdf5']))
            if not f.is_relative_to(folder) or digest(f)!=r['capture']['source_sha256']:
                raise ValueError('Accepted demonstration changed; refusing silent resampling')
            if decode:
                from cx_tta.stacking_training_data import qualify_capture
                qualify_capture(f,task,seed,split,sha)
            accepted.append(r);terminal.add(seed)
        elif r.get('excluded_reason') in EXCLUDED:
            if seed in terminal: raise ValueError('Repeated terminal seed')
            terminal.add(seed)
    if len(accepted)>spec['episodes']: raise ValueError('Collection count overflow')
    return dict(accepted=len(accepted),remaining=spec['episodes']-len(accepted),
                terminal_seeds=sorted(terminal)),accepted


def audit(run,p,task,split):
    folder=run/'collections'/task/split
    state,accepted=resume_state(folder,p,task,split)
    if state['remaining']: raise ValueError('Incomplete collection: '+str(folder))
    result=dict(passed=True,task=task,split=split,scenes=state['accepted'],
                protocol_sha256=digest(run/'protocol.json'),
                captures={r['hdf5']:r['capture']['source_sha256'] for r in accepted},
                gate_labels=False,policy_success_evidence=False)
    replace_json(folder/'data_audit.json',result)
    return result


def execute(run,p,task_name,split,gpu,min_free_gb):
    from cx_tta.native_entry_storage import NativeEntryStorage
    from cx_tta.collect_smoke import make_config
    from cx_tta.recording import instrument_runtime
    from cx_tta.stacking_recording import install_recording
    from cx_tta.collect_stacking_experts_v2 import export_recording
    from cx_tta.collect_stacking_resumable import collect
    from cx_tta.stacking_training_data import qualify_capture
    from cx_tta.evaluate_stacking import Tee
    folder=output_path(run/'collections'/task_name/split);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(task=task_name,split=split,protocol_sha256=digest(run/'protocol.json'),
                      episodes=p['splits'][split]['episodes'],video=False)
        if (folder/'manifest.json').exists():
            if read(folder/'manifest.json')!=manifest: raise ValueError('Collection manifest changed')
        else: write_json(folder/'manifest.json',manifest)
        state,_=resume_state(folder,p,task_name,split)
        if not state['remaining']:
            audit(run,p,task_name,split);print('REUSED',folder,flush=True);return
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no collection started')
        session=folder/'sessions'/('session_'+uuid.uuid4().hex[:12]);session.mkdir(parents=True)
        write_json(session/'manifest.json',dict(manifest,initial=state))
        original=sys.stdout,sys.stderr
        with (session/'console.log').open('x') as log:
            stdout,stderr=Tee(original[0],log),Tee(original[1],log);sys.stdout,sys.stderr=stdout,stderr
            storage=NativeEntryStorage(runtime_source=Path(p['runtime_source']));atexit.register(storage.cleanup)
            try:
                runtime=storage.prepare_runtime(session)
                count=instrument_runtime(runtime)
                write_json(session/'instrumentation.json',dict(counted_step_calls=count))
                storage.configure_runtime(session,runtime,Path(p['tokenizer']))
                import torch.multiprocessing as mp
                from cx_tta.evaluate_b2 import seed_all
                mp.set_start_method('spawn',force=True)
                args=argparse.Namespace(task=task_name,task_config=p['task_config'],split=split,
                    protocol=run/'protocol.json',min_free_gb=min_free_gb,gpu=gpu)
                config=make_config(runtime,args,session)
                config.update(eval_mode=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
                if config['save_freq']!=15 or config['data_type']['rgb'] is not True:
                    raise ValueError('Unexpected expert recording frequency/RGB contract')
                write_json(session/'resolved_config.json',config)
                task=getattr(importlib.import_module('envs.'+task_name),task_name)()
                from envs.utils.create_actor import UnStableError
                install_recording(task,effect_schema=True)
                result=collect(task,config,args,p,folder,session,state,seed_all,UnStableError,export_recording,qualify_capture)
                replace_json(folder/'result.json',result)
                if result['status']!='collection_complete': raise RuntimeError('Expert seed window exhausted')
                audit(run,p,task_name,split)
                write_json(session/'source_runtime_audit.json',storage.verify_source())
                load(run)
                print('COLLECTION COMPLETE',folder,flush=True)
            except BaseException:
                replace_json(folder/'last_failure.json',dict(session=str(session),traceback=traceback.format_exc()));raise
            finally:
                sys.stdout,sys.stderr=original;stdout.close();stderr.close()


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--task',required=True);ap.add_argument('--split',required=True)
    ap.add_argument('--gpu',default='0');ap.add_argument('--min-free-gb',type=int,default=10)
    a=ap.parse_args();run=output_path(a.run.resolve());p=load(run)
    if a.task not in p['tasks'] or a.split not in p['splits'] or a.min_free_gb<1: ap.error('Invalid task/split/disk bound')
    if a.split!='smoke':audit(run,p,a.task,'smoke')
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    execute(run,p,a.task,a.split,a.gpu,a.min_free_gb)


if __name__=='__main__':main()
