"""Frozen full-event evaluation lifecycle; official seed groups and whole-episode resume."""
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
from .frozen_event_support import REGISTRY, resolve_adapter, audited_adapter, deployed_sources
import re


def options(argv=None):
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--task',required=True)
    ap.add_argument('--condition',choices=('baseline','Z_zero','M_event_full'),default='M_event_full')
    ap.add_argument('--adapter',choices=('auto','legacy4','stacking2'),default='auto')
    ap.add_argument('--resume-run',type=Path)
    ap.add_argument('--expected-identity',help=argparse.SUPPRESS)
    ap.add_argument('--gpu',default='0');ap.add_argument('--episodes',type=int,default=100)
    ap.add_argument('--seed',type=int,default=0,help='Official seed group: first scene = 100000*(1+seed)')
    ap.add_argument('--max-attempts',type=int,default=1000)
    ap.add_argument('--task-config',choices=('demo_randomized','demo_clean'),default='demo_randomized')
    ap.add_argument('--instruction-type',choices=('seen','unseen'),default='unseen')
    ap.add_argument('--tokenizer-path',type=Path)
    ap.add_argument('--no-color',action='store_true');ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args(argv)
    if not re.fullmatch(r'[a-z][a-z0-9_]*',a.task):ap.error('Invalid task name')
    if not a.gpu.isdigit() or not 1<=a.episodes<=a.max_attempts<=100000:ap.error('Invalid counts/GPU')
    if a.seed<0:ap.error('seed must be a nonnegative official seed group')
    a.seed_start=100000*(1+a.seed)
    if a.seed_start+a.max_attempts>=2**32:ap.error('Scene seed exceeds NumPy seed range')
    # Existing independent TRAIN/calibration windows must never become EVAL.
    if a.seed_start<7105000 and a.seed_start+a.max_attempts>7100000:
        ap.error('Requested group overlaps reserved stacking TRAIN/calibration scenes')
    try:a.adapter=resolve_adapter(a.task,a.adapter)
    except ValueError as exc:ap.error(str(exc))
    if a.condition=='baseline':a.condition='Z_zero'
    a.training_run=REGISTRY[a.adapter]['run']
    a.event_config=WORKSPACE_ROOT/'configs/events.json'
    a.event_cache_limit=0;a.smoke_queries=None
    return a


def preflight(args):
    from .cache_expert import RT
    from .collect_smoke import make_config
    from .frozen_event_support import verify_base
    meta,plan,files=audited_adapter(args.adapter)
    files=list(files)
    for name,sha in plan['source_sha256'].items():
        file=Path(name)
        if file.is_relative_to(RT) or file.suffix=='.urdf':
            if digest(file)!=sha:raise ValueError('Training/deployment code or embodiment changed: '+name)
            files.append(file)
    reference=read(REFERENCE)
    reference_row=reference['reference'].get(args.task,reference['reference']['beat_block_hammer'])
    ref=Path(reference_row['run']);runtime=ref/'robotwin'
    tokenizer=(args.tokenizer_path or ROOT/'outputs/base_cache_20260926_024808_872087/cache/openpi/big_vision/paligemma_tokenizer.model').resolve()
    extra,assets,budget=task_contract(runtime,args.task)
    config=EventConfig(**read(args.event_config))
    if config.to_dict()!=EventConfig().to_dict():raise ValueError('M_event_full event configuration is frozen; restore original configs/events.json')
    resolved=make_config(runtime,args,WORKSPACE_ROOT/'outputs/frozen_event_eval/preflight')
    # No simulator/GPU initialization; verify the actual selected YAML resolves.
    if resolved.get('task_name')!=args.task:raise ValueError('Resolved task differs')
    policy_identity=verify_base(args.adapter,tokenizer)
    if policy_identity.get('live_identity',policy_identity).get('tokenizer_sha256')!=digest(tokenizer):
        raise ValueError('Tokenizer differs from training policy')
    files += [REFERENCE,OFFICIAL_CONSOLE,ref/'provenance.json',tokenizer,args.event_config,
        Path(meta['projection_path']),Path(meta['robot_config_path']),
        runtime/'description/utils/generate_episode_instructions.py',
        runtime/'task_config'/f'{args.task_config}.yml',runtime/'task_config/_embodiment_config.yml',
        runtime/'task_config/_camera_config.yml',*extra,*assets,*deployed_sources(),
        WORKSPACE_ROOT/'scripts/eval_event_full_worker.sh']
    checkpoint=args.training_run/REGISTRY[args.adapter]['filename']
    sources={str(f):digest(f) for f in dict.fromkeys(files)}
    manifest=dict(entry='frozen_event_full_official_v1',task=args.task,condition=args.condition,
        episodes=args.episodes,seed=args.seed,seed_start=args.seed_start,max_attempts=args.max_attempts,
        task_config=args.task_config,instruction_type=args.instruction_type,adapter=args.adapter,
        training_run=str(args.training_run),checkpoint_path=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        adapter_training_tasks=list(REGISTRY[args.adapter]['tasks']),
        task_seen_in_adapter_training=args.task in REGISTRY[args.adapter]['tasks'],
        event_config=config.to_dict(),event_cache_limit=0,arm_cap_rad=.1,gripper_residual=0,
        official_task_action_budget=budget,runtime_source=str(runtime),source_sha256=sources,
        no_video=True,no_image_or_feature_cache=True,history_mode=REGISTRY[args.adapter]['history'],
        gate_trained=False,tta=False,candidate_stage=REGISTRY[args.adapter]['stage'],
        comparison='Independent model/simulator/compiler processes; matched seed lists must be checked; no identical-inference pairing claim',
        checkpoint_caption=args.adapter+'_seed0',policy_identity=policy_identity,
        task_runtime_sha256=read(ref/'provenance.json')['runtime_sha256'])
    manifest['evaluation_identity']=identity(manifest)
    return dict(training_run=str(args.training_run),source_sha256=sources),ref,tokenizer,manifest


def choose_run(args,manifest):
    parent=WORKSPACE_ROOT/'outputs/frozen_event_eval'/args.task/args.task_config/args.adapter/args.condition
    if args.resume_run:
        run=output_path(args.resume_run.resolve())
        if run.parent!=parent or read(run/'manifest.json')['evaluation_identity']!=manifest['evaluation_identity']:
            raise ValueError('Explicit resume identity/path differs')
        return run
    found=[r for r in parent.glob('eval_*') if (r/'manifest.json').exists()
           and read(r/'manifest.json')['evaluation_identity']==manifest['evaluation_identity']]
    if len(found)>1:raise ValueError('Multiple matching datasets; specify --resume-run; never concatenate')
    if found:return found[0]
    run=new_run('eval',group=('frozen_event_eval',args.task,args.task_config,args.adapter,args.condition))
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
        task_config=manifest['task_config'],adapter=manifest['adapter'],seed=manifest['seed'],
        seed_start=manifest['seed_start'],gate_trained=False,no_video=True,
        continuation='Complete episodes retained; same seed/index for interrupted episode; compiler processes separately recorded')


def main(execute,argv=None):
    args=options(argv);p,ref,tokenizer,manifest=preflight(args)
    if args.expected_identity and args.expected_identity!=manifest['evaluation_identity']:
        raise ValueError('Worker inputs differ from the scheduler preflight; no GPU evaluation started')
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
            print('REUSED_COMPLETE_EVALUATION',run,flush=True)
            print(f"{args.task} / {args.condition}: {result['successes']}/{result['completed']} ({result['success_rate']:.1%})",flush=True);return
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
