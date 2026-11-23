"""Lifecycle and identity for new-adapter evaluations; old transfer runs remain frozen."""
import argparse
import atexit
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import traceback
import uuid
from collections import Counter

from .audit import digest
from .paths import ROOT,WORKSPACE_ROOT,new_run,output_path,write_json,replace_json
from .evaluate_stacking import read,Tee,REFERENCE,OFFICIAL_CONSOLE
from .evaluation_resume import identity,load_journal
from .multi_event import EventConfig
from .stacking_support import task_contract
from .gpu_job_status import publish_job


def options(argv=None):
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--task',choices=('stack_bowls_two','stack_blocks_two'),required=True)
    ap.add_argument('--condition',choices=('Z_zero','M_first_full','M_event_full'),default='M_first_full')
    ap.add_argument('--training-run',type=Path,required=True)
    ap.add_argument('--resume-run',type=Path,help='Optional explicit dataset root; otherwise resume matching identity automatically')
    ap.add_argument('--gpu',default='0');ap.add_argument('--episodes',type=int,default=100)
    ap.add_argument('--seed-start',type=int,default=7105000);ap.add_argument('--max-attempts',type=int,default=1000)
    ap.add_argument('--task-config',choices=('demo_randomized',),default='demo_randomized')
    ap.add_argument('--instruction-type',choices=('unseen',),default='unseen')
    ap.add_argument('--tokenizer-path',type=Path)
    ap.add_argument('--event-config',type=Path,default=WORKSPACE_ROOT/'configs/events.json')
    ap.add_argument('--event-cache-limit',type=int,default=0)
    ap.add_argument('--no-color',action='store_true');ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args(argv)
    if not a.gpu.isdigit() or not 1<=a.episodes<=a.max_attempts<=1000 or not 0<=a.event_cache_limit<=100:ap.error('Invalid counts/GPU')
    if a.seed_start<7105000 or a.seed_start+a.max_attempts>7107000:ap.error('Use registered final/confirmation EVAL windows; no TRAIN/validation scenes')
    if a.seed_start<7106000<a.seed_start+a.max_attempts:ap.error('Do not mix final/confirmation windows')
    a.training_run=output_path(a.training_run.resolve());a.event_config=a.event_config.resolve();a.smoke_queries=None
    return a


def preflight(args):
    from .stacking_adapter import read_checkpoint
    from .cache_expert import RT
    checkpoint=args.training_run/'adapter_seed0.pt';payload=read_checkpoint(checkpoint);meta=payload['metadata']
    training_plan=read(args.training_run/'training_plan.json')
    deployed_dependencies=[]
    for name,sha in training_plan['source_sha256'].items():
        file=Path(name)
        if file.is_relative_to(RT) or file.suffix=='.urdf':
            if digest(file)!=sha:raise ValueError('Training/deployment VLA code or embodiment changed: '+name)
            deployed_dependencies.append(file)
    reference=read(REFERENCE);ref=Path(reference['reference'].get(args.task,reference['reference']['beat_block_hammer'])['run'])
    runtime=ref/'robotwin'
    tokenizer=(args.tokenizer_path or ROOT/'outputs/base_cache_20260926_024808_872087/cache/openpi/big_vision/paligemma_tokenizer.model').resolve()
    extra,assets,budget=task_contract(runtime,args.task)
    config=EventConfig(**read(args.event_config))
    files=[REFERENCE,OFFICIAL_CONSOLE,tokenizer,checkpoint,args.training_run/'training_plan.json',
        args.training_run/'training_audit.json',Path(meta['projection_path']),Path(meta['robot_config_path']),args.event_config,
        runtime/'description/utils/generate_episode_instructions.py',*extra,*assets,*deployed_dependencies]
    for name in ('stacking_adapter.py','stacking_policy.py','stacking_eval_resume.py','evaluation_resume.py',
        'evaluate_stacking_adapter.py','evaluate_b2.py','stacking_training.py','multi_event.py','official_onset_domain.py','output_support_policy.py',
        'structured_action_deploy.py','structured_action_adapter.py','geometry_readout.py','structured_features.py','cache_joint_prefix.py'):
        files.append(WORKSPACE_ROOT/'cx_tta'/name)
    sources={str(f):digest(f) for f in files}
    manifest=dict(entry='stacking_adapter_resumable_v1',task=args.task,condition=args.condition,
        episodes=args.episodes,seed_start=args.seed_start,max_attempts=args.max_attempts,
        task_config=args.task_config,instruction_type=args.instruction_type,training_run=str(args.training_run),
        checkpoint_sha256=digest(checkpoint),event_config=config.to_dict(),event_cache_limit=args.event_cache_limit,
        official_task_action_budget=budget,runtime_source=str(runtime),source_sha256=sources,
        no_video=True,history_mode='current_only_adapter',gate_trained=False,
        candidate_stage='expert_initialization_only',comparison='Resumed process strata retained; no bitwise equivalence to an uninterrupted compiler process',
        checkpoint_caption='stacking_adapter_seed0',task_runtime_sha256=read(ref/'provenance.json')['runtime_sha256'])
    manifest['evaluation_identity']=identity(manifest)
    return dict(training_run=str(args.training_run),source_sha256=sources),ref,tokenizer,manifest


def choose_run(args,manifest):
    parent=WORKSPACE_ROOT/'outputs/stacking_adapter_eval'/args.task/args.condition
    if args.resume_run:
        run=output_path(args.resume_run.resolve())
        if run.parent!=parent or read(run/'manifest.json')['evaluation_identity']!=manifest['evaluation_identity']:
            raise ValueError('Explicit resume identity/path differs')
        return run
    found=[r for r in parent.glob('eval_*') if (r/'manifest.json').exists()
           and read(r/'manifest.json')['evaluation_identity']==manifest['evaluation_identity']]
    if len(found)>1:raise ValueError('Multiple matching datasets; specify --resume-run; never concatenate')
    if found:return found[0]
    run=new_run('eval',group=('stacking_adapter_eval',args.task,args.condition))
    write_json(run/'manifest.json',manifest);return run


def final_result(run,manifest):
    state=load_journal(run,manifest);eps=state['episodes']
    if len(eps)!=manifest['episodes']:raise ValueError('Not all episodes complete')
    # An interrupt after the final episode commit but before the final summary
    # needs only CPU provenance checks, not another rollout.
    for name,sha in manifest['source_sha256'].items():
        if digest(Path(name))!=sha:raise ValueError('Changed evaluation source')
    for session in {ep['process_session'] for ep in eps}:
        folder=output_path(run/'sessions'/session)
        provenance=read(folder/'provenance.json')
        for name,sha in provenance['runtime_sha256'].items():
            if digest(folder/'robotwin'/name)!=sha:raise ValueError('Private session runtime changed')
    return dict(status='complete',complete=True,task=manifest['task'],condition=manifest['condition'],
        requested=manifest['episodes'],completed=len(eps),successes=sum(ep['success'] for ep in eps),
        success_rate=sum(ep['success'] for ep in eps)/len(eps),attempted=len(state['attempts']),censored=0,
        valid_seeds=[ep['seed'] for ep in eps],process_strata=dict(Counter(ep['process_session'] for ep in eps)),
        checkpoint_sha256=manifest['checkpoint_sha256'],evaluation_identity=manifest['evaluation_identity'],
        continuation='Complete episodes retained; same seed/index for interrupted episode; compiler processes separately recorded')


def main(execute,argv=None):
    args=options(argv);p,ref,tokenizer,manifest=preflight(args)
    if args.dry_run:
        print(json.dumps(dict(manifest,gpu_started=False),ensure_ascii=False,indent=2));return
    lockdir=WORKSPACE_ROOT/'outputs/evaluation_locks';lockdir.mkdir(exist_ok=True)
    with (lockdir/(manifest['evaluation_identity']+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=choose_run(args,manifest);state=load_journal(run,manifest)
        args.dataset_run=run;args.initial_completed=len(state['episodes'])
        if len(state['episodes'])==args.episodes:
            result=final_result(run,manifest);replace_json(run/'result.json',result)
            publish_job(run,'finished',ready=True,result=str(run/'result.json'),last_episode=state['episodes'][-1],initial_completed=args.initial_completed,**result)
            print('REUSED_COMPLETE_EVALUATION',run,flush=True);return
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,JAX_PLATFORMS='cpu')
        from .native_entry_storage import NativeEntryStorage
        from .evaluate_structured_low_storage import _terminate
        storage=NativeEntryStorage(runtime_source=ref/'robotwin');atexit.register(storage.cleanup)
        signal.signal(signal.SIGTERM,_terminate)
        session=run/'sessions'/('process_'+uuid.uuid4().hex[:12]);session.mkdir(parents=True)
        for name in ('queries','onsets'):(session/name).mkdir()
        write_json(session/'manifest.json',dict(manifest,initial_completed=args.initial_completed,pid=os.getpid()))
        publish_job(run,'initializing',ready=False,task=args.task,condition=args.condition,requested=args.episodes,completed=args.initial_completed,initial_completed=args.initial_completed)
        original_out,original_err=sys.stdout,sys.stderr
        with (session/'terminal.log').open('w',buffering=1) as log:
            stdout,stderr=Tee(original_out,log),Tee(original_err,log);sys.stdout,sys.stderr=stdout,stderr
            try:
                print('EVALUATION',run,'SESSION',session,'KEPT',args.initial_completed,flush=True)
                execute(args,p,ref,tokenizer,manifest,storage,session)
                result=final_result(run,manifest);replace_json(run/'result.json',result)
                replace_json(run/'progress.json',dict(phase='finished',**result))
                publish_job(run,'finished',ready=True,result=str(run/'result.json'),last_episode=load_journal(run,manifest)['episodes'][-1],initial_completed=args.initial_completed,**result)
            except BaseException:
                failure=dict(complete=False,traceback=traceback.format_exc(),session=str(session))
                replace_json(session/'failure.json',failure);replace_json(run/'last_failure.json',failure)
                publish_job(run,'failed',ready=False,error=failure['traceback']);raise
            finally:
                sys.stdout,sys.stderr=original_out,original_err;stdout.close();stderr.close()
