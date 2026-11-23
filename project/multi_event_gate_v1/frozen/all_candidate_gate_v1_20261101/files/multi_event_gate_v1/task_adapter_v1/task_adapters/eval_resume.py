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

from cx_tta.audit import digest
from cx_tta.paths import ROOT,WORKSPACE_ROOT,new_run,output_path,write_json,replace_json
from cx_tta.evaluate_stacking import read,Tee,REFERENCE,OFFICIAL_CONSOLE
from cx_tta.evaluation_resume import identity,load_journal
from cx_tta.multi_event import EventConfig
from cx_tta.stacking_support import task_contract
from cx_tta.gpu_job_status import publish_job


def options(argv=None):
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--task',required=True)
    ap.add_argument('--condition',choices=('baseline','Z_zero','M_first_full','M_event_full'),default='M_event_full')
    ap.add_argument('--training-run',type=Path,required=True)
    ap.add_argument('--resume-run',type=Path,help='Optional explicit dataset root; otherwise resume matching identity automatically')
    ap.add_argument('--gpu',default='0');ap.add_argument('--episodes',type=int,default=100)
    starts=ap.add_mutually_exclusive_group();starts.add_argument('--seed',type=int,default=0);starts.add_argument('--seed-start',type=int)
    ap.add_argument('--max-attempts',type=int,default=1000)
    ap.add_argument('--task-config',choices=('demo_randomized','demo_clean'),default='demo_randomized')
    ap.add_argument('--instruction-type',choices=('unseen',),default='unseen')
    ap.add_argument('--tokenizer-path',type=Path)
    ap.add_argument('--event-config',type=Path,default=WORKSPACE_ROOT/'configs/events.json')
    ap.add_argument('--event-cache-limit',type=int,default=0)
    ap.add_argument('--no-color',action='store_true');ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args(argv)
    if not a.gpu.isdigit() or not 1<=a.episodes<=a.max_attempts<=1000 or not 0<=a.event_cache_limit<=100:ap.error('Invalid counts/GPU')
    if a.seed_start is None:a.seed_start=100000*(1+a.seed)
    if a.seed<0 or a.seed_start<0:ap.error('Nonnegative seed required')
    if a.condition=='baseline':a.condition='Z_zero'
    a.training_run=output_path(a.training_run.resolve());a.event_config=a.event_config.resolve();a.smoke_queries=None
    return a


def preflight(args):
    from .adapter import read_checkpoint
    from .protocol import load,task_contract,local_sources,OUTPUT
    checkpoint=args.training_run/'adapter_seed0.pt';payload=read_checkpoint(checkpoint);meta=payload['metadata']
    plan=read(args.training_run/'training_plan.json');dataset=Path(plan['dataset']);protocol=load(dataset)
    if not args.training_run.is_relative_to(OUTPUT):raise ValueError('Use a new task-adapter training run')
    if args.task not in plan['training_tasks']:raise ValueError('Task not covered by this adapter; explicit transfer experiment required')
    for split,spec in protocol['splits'].items():
        if max(args.seed_start,spec['start'])<min(args.seed_start+args.max_attempts,spec['stop']):
            raise ValueError('Evaluation seed window overlaps '+split)
    runtime=Path(protocol['runtime_source']);ref=runtime.parent
    tokenizer=Path(args.tokenizer_path or protocol['tokenizer']).resolve()
    if digest(tokenizer)!=plan['policy_identity']['tokenizer_sha256']:raise ValueError('Tokenizer differs from training')
    contract=task_contract(runtime,args.task,args.task_config)
    config=EventConfig(**read(args.event_config))
    files=[OFFICIAL_CONSOLE,tokenizer,checkpoint,args.training_run/'training_plan.json',
           args.training_run/'training_audit.json',args.training_run/'cache_manifest.json',
           Path(meta['projection_path']),Path(meta['robot_config_path']),args.event_config,
           dataset/'protocol.json',*local_sources()]
    sources={str(f):digest(f) for f in files}
    sources.update(protocol['source_sha256']);sources.update(contract['source_sha256'])
    manifest=dict(entry='task_adapter_resumable_v1',task=args.task,condition=args.condition,
        episodes=args.episodes,seed_start=args.seed_start,max_attempts=args.max_attempts,
        task_config=args.task_config,instruction_type=args.instruction_type,training_run=str(args.training_run),
        checkpoint_sha256=digest(checkpoint),event_config=config.to_dict(),event_cache_limit=args.event_cache_limit,
        official_task_action_budget=contract['native_budget'],runtime_source=str(runtime),source_sha256=sources,
        no_video=True,history_mode='current_only_adapter',gate_trained=False,
        candidate_stage='expert_initialization_only',training_tasks=plan['training_tasks'],
        comparison='Held-out scenes in trained tasks, not zero-shot task generalization',
        checkpoint_caption='task_adapter_seed0',
        task_runtime_sha256={str(f.relative_to(runtime)):digest(f)
                             for part in ('envs','task_config','description')
                             for f in (runtime/part).rglob('*.py') if '__pycache__' not in f.parts})
    manifest['evaluation_identity']=identity(manifest)
    return dict(training_run=str(args.training_run),source_sha256=sources),ref,tokenizer,manifest


def choose_run(args,manifest):
    parent=WORKSPACE_ROOT/'outputs/task_adapter_eval'/args.task/args.condition
    if args.resume_run:
        run=output_path(args.resume_run.resolve())
        if run.parent!=parent or read(run/'manifest.json')['evaluation_identity']!=manifest['evaluation_identity']:
            raise ValueError('Explicit resume identity/path differs')
        return run
    found=[r for r in parent.glob('eval_*') if (r/'manifest.json').exists()
           and read(r/'manifest.json')['evaluation_identity']==manifest['evaluation_identity']]
    if len(found)>1:raise ValueError('Multiple matching datasets; specify --resume-run; never concatenate')
    if found:return found[0]
    run=new_run('eval',group=('task_adapter_eval',args.task,args.condition))
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
    lockdir=WORKSPACE_ROOT/'outputs/evaluation_locks';lockdir.mkdir(parents=True,exist_ok=True)
    with (lockdir/(manifest['evaluation_identity']+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=choose_run(args,manifest);state=load_journal(run,manifest)
        args.dataset_run=run;args.initial_completed=len(state['episodes'])
        if len(state['episodes'])==args.episodes:
            result=final_result(run,manifest);replace_json(run/'result.json',result)
            publish_job(run,'finished',ready=True,result=str(run/'result.json'),last_episode=state['episodes'][-1],initial_completed=args.initial_completed,**result)
            print('REUSED_COMPLETE_EVALUATION',run,flush=True);return
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,JAX_PLATFORMS='cpu')
        from cx_tta.native_entry_storage import NativeEntryStorage
        from cx_tta.evaluate_structured_low_storage import _terminate
        storage=NativeEntryStorage(runtime_source=Path(manifest['runtime_source']));atexit.register(storage.cleanup)
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
