"""Causal timing equivalence smoke and new whole-scene paired process blocks."""
import argparse
import atexit
import importlib
import os
from pathlib import Path
import signal
import time
import traceback
import uuid
import numpy as np

from .paths import ROOT,new_run,output_path,write_json,replace_json
from .audit import digest
from .native_history_cache import read,verify,require
from .causal_arm_gate import CausalArmPolicy,OneShotClosingGate,CONDITIONS,SMOKE_CONDITIONS
from .causal_arm_analysis import order,summarize
from .evaluate_structured_attribution import PhysicalPolicy,CompactTrace,pilot_reference,steady_probe
from .train_output_support import BASE
from .causal_arm_protocol import TRAIN,TASKS


def CPU_preflight(protocol):
    from .structured_attribution_analysis import _load_trace
    p=read(protocol);verify(p['source_sha256']);known=read(ROOT/'outputs/reports/attribution/local_components_accepted_20261006_042530_435693/result.json')
    checks=[]
    for row in known['causal_timing_proxy_exploration']:
        source=Path(row['source']);condition='C00_zero' if 'local_components' in str(source) else 'L0'
        ep=next(e for e in read(source/'episodes.json') if e['seed']==row['seed'] and e['condition']==condition and e['repeat']==0)
        t=_load_trace(source,ep);gate=OneShotClosingGate();states=[]
        for i in range(len(t['rows'])):states.append(gate.step(i,t['arrays']['state'][i].copy(),t['arrays']['base'][i].copy()))
        require(gate.window==row['known_effective_window'],'Stored numerical alignment changed')
        # Prefix replay with no accessible later observations/base chunks.
        for n in range(min(4,len(states))):
            prefix=OneShotClosingGate()
            for i in range(n+1):current=prefix.step(i,t['arrays']['state'][i],t['arrays']['base'][i])
            require(current==states[n],'Prefix decision depends on unavailable later data')
        checks.append(dict(task=row['task'],seed=row['seed'],source=str(source),first_event=gate.first_event,window=gate.window,all_prefixes_match=True))
    write_json(Path(protocol).parent/'CPU_causality_audit.json',dict(passed=True,GPU_started=False,checks=checks,
        current_fields_only=True,no_future_state_or_task_seed_GT_fields=True,
        scene_isolation_passed=read(p['isolation'])['passed'],known_alignment_not_new_success=True,
        source_sha256={str(protocol):digest(Path(protocol))}))
    print('CPU_CAUSALITY_PASS',len(checks),flush=True)


class CausalTrace(CompactTrace):
    def __init__(self,*args,registry_row=None):
        super().__init__(*args);self.captures=[];self.indices=[];self.registry_row=registry_row
    def record(self,**kwargs):
        if self.first is None and self.registry_row is not None:
            from .collect_recovery import signature
            from .eval_diagnostics import json_hash
            require(signature(kwargs['observation'])==self.registry_row['initial_observation_signature'],'Frozen new-scene initial RGB/command differs')
            require(json_hash(self.policy.before)==self.registry_row['initial_physical_sha256'],'Frozen new-scene initial physics differs')
        super().record(**kwargs);x=self.policy.policy.last_legal_inputs
        if x is not None:
            self.captures.append({k:v.copy() for k,v in x.items()});self.indices.append(self.policy.policy.query_count-1)
            require(len(self.indices)<=7,'Causal boundary capture budget')
            file=self.path.parent.parent/'legal_inputs'/(self.stem+'.npz');file.parent.mkdir(exist_ok=True);tmp=file.with_suffix('.npz.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,query_indices=np.asarray(self.indices,np.int64),**{k:np.stack([r[k] for r in self.captures]) for k in x})
            tmp.replace(file)


def registry_check(protocol,registry):
    p=read(protocol);m=read(registry/'manifest.json');r=read(registry/'result.json');rows=read(registry/'records.json')
    require(m['protocol_sha256']==digest(Path(protocol)) and r['status']=='complete' and r['all_fixed_seeds_attempted'] and r['policy_sampler_calls']==0,'Completed pre-outcome registry required')
    verify(r['source_sha256'])
    expected={(t,s) for t in p['tasks'] for s in p['seeds']}
    require(len(rows)==len(expected) and {(r['task'],r['seed']) for r in rows}==expected,'No missing/replaced registry scenes')
    return rows


def execute(args,storage):
    p=read(args.protocol);verify(p['source_sha256']);require(read(args.protocol.parent/'CPU_causality_audit.json')['passed'],'CPU gate')
    require(p['stage']=='causal_arm_confirmation' and args.task in p['tasks'],'Registered task')
    smoke=args.mode=='smoke';_,legacy,_,evidence=pilot_reference(args.task,BASE)
    if smoke:
        seeds=[min(legacy_seed for legacy_seed in (100380,100382 if args.task=='place_empty_cup' else 100387))]
        rows=None;conditions=SMOKE_CONDITIONS;fixed=p['smoke_fixed_windows'][args.task]
    else:
        require(args.registry is not None,'Pre-outcome registry required')
        rows=registry_check(args.protocol,args.registry);seeds=[s for s in p['seeds'] if next(r for r in rows if r['task']==args.task and r['seed']==s)['eligible']]
        conditions=CONDITIONS;fixed=None
    schedule=order(seeds,conditions,args.block)
    manifest=dict(stage=p['stage'],task=args.task,protocol=str(args.protocol),protocol_sha256=digest(args.protocol),
        training_run=str(TRAIN),mode=args.mode,seeds=seeds,attempted_seeds=seeds if smoke else p['seeds'],
        invalid_scenes=[] if smoke else [r for r in rows if r['task']==args.task and not r['eligible']],
        conditions=list(conditions),block_id=args.block,max_chunks=args.max_chunks if smoke else None,fixed_window=fixed,
        registry=None if smoke else str(args.registry),registry_records_sha256=None if smoke else digest(args.registry/'records.json'),
        order=[dict(seed=s,repeat=b,condition=c) for s,b,c in schedule],
        process_instance_id=str(uuid.uuid4()),pid=os.getpid(),no_video=True,compiled=True,internal_noise=True,
        same_sampler_in_block=True,independent_process_block=True,scope=p['scientific_scope'])
    if args.dry_run:print('DRY_RUN',dict(manifest=manifest,full_rollouts_requested=len(schedule),GPU_started=False),flush=True);return
    import torch
    from .cache_expert import create_base_policy
    from .evaluate_b2 import configure_inference,resolve_tokenizer,run_episode
    from .evaluate_structured_action import check_training_policy_identity,SmokeBudgetReached
    from .output_support_policy import OutputSupportPolicy
    from .collect_smoke import make_config
    require(torch.cuda.is_available(),'Visible CUDA before creating run')
    run=new_run('paired_smoke' if smoke else 'block',group=('diagnostics','causal_arm',args.task));print('RUN',run,flush=True)
    write_json(run/'manifest.json',manifest);episodes=[];began=time.monotonic()
    try:
        require(bool(seeds),'No eligible fixed scenes; report invalid registry, no replacement')
        tokenizer=resolve_tokenizer();identity=check_training_policy_identity(BASE,tokenizer)
        storage.runtime_source=Path(evidence['run'])/'robotwin';runtime=storage.prepare_runtime(run);storage.configure_runtime(run,runtime,tokenizer)
        args.task_config,args.instruction_type='demo_randomized','unseen';config=make_config(runtime,args,run)
        config.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0);write_json(run/'resolved_config.json',config)
        old=read(Path(evidence['run'])/'provenance.json');rt={str(f.relative_to(runtime)):digest(f) for f in runtime.rglob('*.py') if '__pycache__' not in f.parts}
        require(rt==old['runtime_sha256'],'Frozen official runtime identity')
        for name,sha in old['code_sha256'].items():require(digest(Path(name))==sha,'Frozen source identity')
        write_json(run/'provenance.json',dict(runtime_sha256=rt,source_sha256={str(f):digest(f) for f in (ROOT/'cx_tta').glob('*.py')},identity=identity['live_identity'],policy_source_sha256=identity['policy_source_sha256']))
        task=getattr(importlib.import_module('envs.'+args.task),args.task)();configure_inference('compiled',0);torch.set_num_threads(2)
        replace_json(run/'progress.json',dict(phase='loading_shared_base',completed=0,requested=len(schedule)))
        base=create_base_policy();base._model.eval().requires_grad_(False)
        wrappers={c:CausalArmPolicy(base,TRAIN,c,fixed if c=='A_fixed_arm' else None) for c in conditions}
        for w in wrappers.values():w.check_robot_config(config)
        zero=CausalArmPolicy(base,TRAIN,'Z_zero');true_l0=OutputSupportPolicy(base,TRAIN,'L0')
        write_json(run/'hardware.json',dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
            compiled=hasattr(base._sample_actions,'_torchdynamo_orig_callable'),global_dynamo_disabled=bool(torch._dynamo.config.disable),
            shared_sampler=all(w.sampler is base._sample_actions for w in wrappers.values()),sample_kwargs=base._sample_kwargs,
            process_instance_id=manifest['process_instance_id'],pid=os.getpid(),compiler_scratch=read(run/'compiler_temp.json')['path']))
        def timeout(*_):raise TimeoutError('Causal block declared scene/compile timeout')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(args.scene_timeout)
        instruction=legacy[seeds[0]]['instruction'] if smoke else next(r['instruction'] for r in rows if r['task']==args.task and r['seed']==seeds[0])
        replace_json(run/'progress.json',dict(phase='common_warmup',completed=0,requested=len(schedule)))
        probe=steady_probe(task,config,seeds[0],instruction,{'L0':true_l0,'S0_zero':zero})
        np.savez_compressed(run/'steady_probe.npz',base=probe.pop('base_values'));write_json(run/'steady_probe.json',probe);require(probe['passed'],'CausalZ exact originalL0 probe');signal.alarm(0)
        for seed,block,condition in schedule:
            signal.alarm(args.scene_timeout);w=wrappers[condition];w.reset_history();w.episode_seed=seed
            reg=None if smoke else next(r for r in rows if r['task']==args.task and r['seed']==seed)
            instruction=legacy[seed]['instruction'] if smoke else reg['instruction'];index=seeds.index(seed) if smoke else p['seeds'].index(seed)
            replace_json(run/'progress.json',dict(phase='causal_block',completed=len(episodes),requested=len(schedule),seed=seed,condition=condition))
            physical=PhysicalPolicy(w,task,args.max_chunks if smoke else None);stem=f'seed{seed}_b{block}_{condition}';trace=CausalTrace(run,stem,physical,registry_row=reg)
            try:
                ep=run_episode(task,config,runtime,seed,index,physical,True,instruction_type='unseen',instruction_override=instruction,trace=trace,rollout_protocol='official');ep['complete']=True
                if smoke:ep['observed_native_success']=ep.pop('success');ep.update(complete=False,success=None,reason='causal_smoke_no_success_claim')
            except SmokeBudgetReached as exc:ep=exc.record
            except Exception as exc:ep=dict(seed=seed,eligible=True,complete=False,success=None,reason='error',error_type=type(exc).__name__,traceback=traceback.format_exc())
            ep.update(repeat=block,condition=condition,queries=w.query_count,trace_stem=stem);episodes.append(ep);replace_json(run/'episodes.json',episodes)
            print('EPISODE',seed,block,condition,ep.get('success'),ep.get('executed_actions'),flush=True);signal.alarm(0)
            if ep.get('reason')=='error':break
        result=summarize(run);result['elapsed_s']=time.monotonic()-began;write_json(run/'result.json',result)
        write_json(run/'source_runtime_audit.json',storage.verify_source());replace_json(run/'progress.json',dict(phase='finished',interface_passed=result['interface_passed']))
        for c,v in result['observed_outcomes'].items():
            rate='N/A' if v['ratio'] is None else f"{v['ratio']:.1%}"
            print(args.task,c,'success',f"{v['successes']}/{v['completed']}",'rate',rate,'censored',v['censored'],flush=True)
        require(result['interface_passed'],'Causal interface/registry/frame audit failed')
    except BaseException:
        if not (run/'failure.json').exists():write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed_records=len(episodes)))
        raise
    finally:signal.alarm(0)


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--protocol',required=True,type=Path);ap.add_argument('--registry',type=Path)
    ap.add_argument('--task',required=True,choices=TASKS);ap.add_argument('--mode',choices=('smoke','block'),default='block');ap.add_argument('--block',type=int,choices=(0,1),default=0)
    ap.add_argument('--max-chunks',type=int,default=5);ap.add_argument('--gpu',default='0');ap.add_argument('--scene-timeout',type=int,default=1800);ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args();a.protocol=output_path(a.protocol)
    if a.registry is not None:a.registry=output_path(a.registry)
    require(a.gpu.isdigit() and a.scene_timeout>=60 and a.max_chunks in (5,6),'GPU/timeout/bounded smoke')
    from .native_entry_storage import NativeEntryStorage
    from .evaluate_structured_low_storage import _terminate
    storage=NativeEntryStorage()
    if not a.dry_run:
        atexit.register(storage.cleanup);signal.signal(signal.SIGTERM,_terminate);os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;os.environ['JAX_PLATFORMS']='cpu'
    execute(a,storage)

if __name__=='__main__':main()
