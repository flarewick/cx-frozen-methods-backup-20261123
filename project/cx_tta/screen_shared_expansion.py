"""Thin official expert-valid-scene/instruction registry before policy outcomes."""
import argparse
import atexit
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
import signal
import traceback
from .paths import new_run,write_json,replace_json,output_path
from .audit import digest
from .shared_expansion_contract import validate_protocol
from .native_history_cache import read,verify,require


def run_screen(protocol,gpu):
    p=read(protocol);validate_protocol(p);verify(p['source_sha256']);require(p['stage']=='shared_onset_expansion','Only new formal scenes are screened')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu;os.environ['JAX_PLATFORMS']='cpu'
    import torch
    require(torch.cuda.is_available(),'Visible GPU required for scene renderer/planner')
    from .native_entry_storage import NativeEntryStorage
    from .evaluate_b2 import seed_all,resolve_tokenizer
    from .collect_smoke import make_config
    from .capture import instruction_for
    from .attribution_physics import physical_snapshot
    from .collect_recovery import signature
    from .eval_diagnostics import json_hash
    run=new_run('registry',group=('reports','shared_expansion_registries'));print('REGISTRY',run,flush=True)
    storage=NativeEntryStorage(runtime_source=Path(p['reference']['place_empty_cup']['run'])/'robotwin');atexit.register(storage.cleanup)
    write_json(run/'manifest.json',dict(stage='shared_expansion_eligibility_only',protocol=str(protocol),protocol_sha256=digest(protocol),
        tasks=p['tasks'],seeds=p['seeds'],model_loaded=False,policy_sampler_calls=0,no_video=True,scene_replacement=False))
    records=[];current=None
    def timeout(*_):raise TimeoutError('Expert scene screening timeout900s')
    signal.signal(signal.SIGALRM,timeout)
    try:
        runtime=storage.prepare_runtime(run);storage.configure_runtime(run,runtime,resolve_tokenizer())
        configs={}
        for task_name in p['tasks']:
            args=SimpleNamespace(task=task_name,task_config='demo_randomized')
            config=make_config(runtime,args,run);config.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
            configs[task_name]=config;write_json(run/(task_name+'_config.json'),config)
            cls=getattr(importlib.import_module('envs.'+task_name),task_name)
            for index,seed in enumerate(p['seeds']):
                signal.alarm(900);current=cls();seed_all(seed)
                replace_json(run/'progress.json',dict(task=task_name,seed=seed,completed=len(records),requested=len(p['tasks'])*len(p['seeds'])))
                row=dict(task=task_name,seed=seed,scene_index=index,eligible=False)
                try:
                    current.setup_demo(now_ep_num=index,seed=seed,**dict(config,need_plan=True,is_test=True))
                    obs=current.get_obs();initial=physical_snapshot(current);require(initial['usable'],'Initial physical coverage')
                    row.update(initial_physical_sha256=json_hash(initial),initial_observation_signature=signature(obs),native_limit=int(current.step_lim))
                    scene_info=current.play_once();plan=bool(current.plan_success);success=bool(current.check_success())
                    row.update(expert_plan_success=plan,expert_success=success,eligible=bool(plan and success))
                    if row['eligible']:
                        row.update(instruction=instruction_for(runtime,task_name,scene_info,seed,split='unseen'),
                            grounded_objects={k:str(v) for k,v in scene_info.get('info',{}).items()})
                    else:row['reason']='expert_plan_or_success_failed'
                except Exception as exc:
                    if type(exc).__name__!='UnStableError':raise
                    row.update(eligible=False,reason='unstable_scene')
                finally:
                    try:current.close_env()
                    finally:current=None;signal.alarm(0)
                records.append(row);replace_json(run/'records.json',records);print('SCREEN',task_name,seed,row['eligible'],row.get('reason'),flush=True)
        verify(p['source_sha256'])
        write_json(run/'source_runtime_audit.json',storage.verify_source())
        write_json(run/'result.json',dict(status='complete',attempted=len(records),tasks={t:dict(attempted=len(p['seeds']),eligible=sum(r['eligible'] for r in records if r['task']==t),invalid=sum(not r['eligible'] for r in records if r['task']==t)) for t in p['tasks']},
            policy_model_loaded=False,policy_sampler_calls=0,policy_success_outcomes_collected=0,all_fixed_seeds_attempted=True,no_replacement=True,
            source_sha256={str(run/n):digest(run/n) for n in ('records.json','manifest.json','source_runtime_audit.json')} | {str(protocol):digest(protocol)}))
        replace_json(run/'progress.json',dict(phase='finished',completed=len(records)))
        print('REGISTRY_COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed=len(records),policy_outcomes=0));raise
    finally:
        if current is not None:current.close_env()
        signal.alarm(0)


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--protocol',required=True,type=Path);ap.add_argument('--gpu',default='0');ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args();a.protocol=output_path(a.protocol);require(a.gpu.isdigit(),'Numeric GPU')
    p=read(a.protocol);validate_protocol(p);verify(p['source_sha256'])
    if a.dry_run:print('SCREEN_DRY_RUN',p['tasks'],p['seeds'],'VLA_calls=0');return
    run_screen(a.protocol,a.gpu)

if __name__=='__main__':main()
