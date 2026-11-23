"""Same-process frozenA always vs two bounded temporal windows; no video/RGB."""
import argparse
import atexit
import importlib
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np

from .audit import digest
from .paths import ROOT,new_run,output_path,write_json,replace_json
from .native_history_cache import read,verify,require
from .local_intervention import CONDITIONS,choose_windows,LocalInterventionPolicy
from .local_intervention_analysis import order,summarize
from .structured_attribution_analysis import _load_trace
from .evaluate_structured_attribution import CompactTrace,PhysicalPolicy,pilot_reference,steady_probe
from .train_output_support import BASE

TRAIN=ROOT/'outputs/effect_learning/output_support_20261005_082307_793921'
ACCEPTED=ROOT/'outputs/reports/attribution/output_support_accepted_20261005_123653_284756'
TASKS=('beat_block_hammer','place_empty_cup')


def prepare_protocol():
    accepted=read(ACCEPTED/'result.json');require(accepted['status']=='passed','Accepted S2o evidence')
    sources={};windows={};references={};budgets={}
    for task in TASKS:
        run=Path(accepted['tasks'][task]['run']);es=read(run/'episodes.json');windows[task]={}
        for ep in es:
            if ep['condition']=='L0' and ep['repeat']==0:
                t=_load_trace(run,ep);windows[task][str(ep['seed'])]=choose_windows(t['rows'])
                for suffix in ('.npz','.jsonl.gz'):
                    path=run/'traces'/(ep['trace_stem']+suffix);sources[str(path)]=digest(path)
        _,_,_,evidence=pilot_reference(task,BASE);references[task]=evidence
        cfg=Path(evidence['run'])/'robotwin/task_config/_eval_step_limit.yml';budgets[task]=digest(cfg);sources[str(cfg)]=digest(cfg)
        sources[str(run/'episodes.json')]=digest(run/'episodes.json')
    q=read(TRAIN/'training_protocol.json');sources.update(q['source_sha256'])
    paths=[ACCEPTED/'result.json',TRAIN/'A_bounded_clipped_seed0.pt',TRAIN/'training_audit.json',
        ROOT/'cx_tta/local_intervention.py',ROOT/'cx_tta/local_intervention_analysis.py',Path(__file__),
        ROOT/'scripts/eval_local_intervention.sh',ROOT/'tests/test_local_intervention.py']
    sources.update({str(p):digest(p) for p in paths})
    out=new_run('local_intervention',group=('reports','protocols'))
    p=dict(stage='local_intervention',status='frozen_before_live_validation',training_run=str(TRAIN),
        checkpoint_sha256=digest(TRAIN/'A_bounded_clipped_seed0.pt'),conditions=list(CONDITIONS),windows=windows,
        tasks=list(TASKS),seeds={task:[int(s) for s in windows[task]] for task in TASKS},repeats=2,
        reference=references,budget_config_sha256=budgets,scales=q['reference_scales_native_units'],robot_joint_limits=q['robot_joint_limits'],
        window_rule='First L0 commanded gripper closure crossing0.5; previous/current2-query window and subsequent2-query window. Same rule for both tasks/scenes, frozen once, no outcome search.',
        scientific_scope='Offline source-derived diagnostic schedule, not learned local router or task-specific deployed lookup; two reused development scenes/task.',
        original_budget_unchanged=True,noise='internal',compiled=True,observations='after each action',
        arm_and_gripper_strength_when_active=1,zero_identity_when_off=True,no_boundary_smoothing=True,
        max_local_queries=2,max_local_native_actions=64,no_new_training=True,no_video=True,
        sources_missing_from_original_training=33,source_sha256=sources)
    write_json(out/'protocol.json',p);print('PROTOCOL',out/'protocol.json',flush=True);return out/'protocol.json'


class LocalTrace(CompactTrace):
    def __init__(self,*args):
        super().__init__(*args);self.captures=[];self.indices=[]
    def record(self,**kwargs):
        super().record(**kwargs);actual=self.policy.policy.last_legal_inputs
        if actual is not None:
            self.captures.append({k:v.copy() for k,v in actual.items()});self.indices.append(self.policy.policy.query_count-1)
            require(len(self.indices)<=6,'Small registered capture budget')
            path=self.path.parent.parent/'legal_inputs'/(self.stem+'.npz');path.parent.mkdir(exist_ok=True)
            temp=path.with_suffix('.npz.tmp')
            with temp.open('wb') as f:
                np.savez_compressed(f,query_indices=np.asarray(self.indices,np.int64),
                    **{k:np.stack([r[k] for r in self.captures]) for k in actual})
            temp.replace(path)


def execute(args,storage):
    p=read(args.protocol);verify(p['source_sha256']);require(args.task in TASKS and p['stage']=='local_intervention','Registered task')
    _,reference,_,evidence=pilot_reference(args.task,BASE)
    seeds=p['seeds'][args.task][:1] if args.max_chunks is not None else p['seeds'][args.task]
    schedule=order(seeds,args.max_chunks is not None)
    manifest=dict(stage='local_intervention',task=args.task,protocol=str(args.protocol),protocol_sha256=digest(args.protocol),
        training_run=p['training_run'],seeds=seeds,conditions=list(CONDITIONS),max_chunks=args.max_chunks,
        order=[dict(seed=s,repeat=r,condition=c) for s,r,c in schedule],no_video=True,compiled=True,
        original_native_budget=True,noise_source='model_internal_torch',shared_base_models=1,shared_compiled_samplers=1,
        scope=p['scientific_scope'])
    if args.dry_run:print('DRY_RUN',manifest,flush=True);return
    import torch
    from .cache_expert import create_base_policy
    from .evaluate_structured_action import check_training_policy_identity,SmokeBudgetReached
    from .evaluate_b2 import configure_inference,resolve_tokenizer,run_episode
    from .output_support_policy import OutputSupportPolicy
    from .collect_smoke import make_config
    require(torch.cuda.is_available(),'CUDA required; no run before visibility check')
    run=new_run('paired_smoke' if args.max_chunks is not None else 'paired',group=('diagnostics','local_intervention',args.task));print('RUN',run,flush=True)
    write_json(run/'manifest.json',manifest);episodes=[];began=time.monotonic()
    try:
        tokenizer=resolve_tokenizer();identity=check_training_policy_identity(BASE,tokenizer)
        storage.runtime_source=Path(evidence['run'])/'robotwin';runtime=storage.prepare_runtime(run);storage.configure_runtime(run,runtime,tokenizer)
        args.task_config,args.instruction_type='demo_randomized','unseen';config=make_config(runtime,args,run)
        config.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0);write_json(run/'resolved_config.json',config)
        old=read(Path(evidence['run'])/'provenance.json');rt={str(f.relative_to(runtime)):digest(f) for f in runtime.rglob('*.py') if '__pycache__' not in f.parts}
        require(rt==old['runtime_sha256'],'Frozen runtime identity')
        for name,sha in old['code_sha256'].items():require(digest(Path(name))==sha,'Frozen source changed')
        write_json(run/'provenance.json',dict(runtime_sha256=rt,source_sha256={str(f):digest(f) for f in (ROOT/'cx_tta').glob('*.py')},
            identity=identity['live_identity'],policy_source_sha256=identity['policy_source_sha256']))
        task=getattr(importlib.import_module('envs.'+args.task),args.task)();configure_inference('compiled',0);torch.set_num_threads(2)
        replace_json(run/'progress.json',dict(phase='loading_shared_base',completed=0,requested=len(schedule)))
        base=create_base_policy();base._model.eval().requires_grad_(False)
        wrappers={c:LocalInterventionPolicy(base,TRAIN,c) for c in CONDITIONS}
        for wrapper in wrappers.values():wrapper.check_robot_config(config)
        zero=OutputSupportPolicy(base,TRAIN,'A_bounded_clipped',force_zero=True)
        wrappers['L0'].set_windows(p['windows'][args.task][str(seeds[0])])
        write_json(run/'hardware.json',dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
            compiled=hasattr(base._sample_actions,'_torchdynamo_orig_callable'),global_dynamo_disabled=bool(torch._dynamo.config.disable),
            shared_sampler=all(w.sampler is base._sample_actions for w in wrappers.values()),sample_kwargs=base._sample_kwargs))
        def timeout(*_):raise TimeoutError('Registered local scene timeout')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(args.scene_timeout)
        replace_json(run/'progress.json',dict(phase='common_warmup',completed=0,requested=len(schedule)))
        probe=steady_probe(task,config,seeds[0],reference[seeds[0]]['instruction'],{'L0':wrappers['L0'],'S0_zero':zero})
        np.savez_compressed(run/'steady_probe.npz',base=probe.pop('base_values'));write_json(run/'steady_probe.json',probe)
        require(probe['passed'],'Fixed-input zero probe');signal.alarm(0)
        for seed,repeat,condition in schedule:
            signal.alarm(args.scene_timeout);wrapper=wrappers[condition];wrapper.reset_history();wrapper.episode_seed=seed;wrapper.set_windows(p['windows'][args.task][str(seed)])
            replace_json(run/'progress.json',dict(phase='local_windows',completed=len(episodes),requested=len(schedule),seed=seed,condition=condition))
            physical=PhysicalPolicy(wrapper,task,args.max_chunks);stem=f'seed{seed}_r{repeat}_{condition}';trace=LocalTrace(run,stem,physical)
            try:
                ep=run_episode(task,config,runtime,seed,seeds.index(seed),physical,condition!='L0',instruction_type='unseen',instruction_override=reference[seed]['instruction'],trace=trace,rollout_protocol='official')
                ep['complete']=True
                if args.max_chunks is not None:ep['observed_native_success']=ep.pop('success');ep.update(success=None,complete=False,reason='interface_smoke_no_rate_claim')
            except SmokeBudgetReached as exc:ep=exc.record
            except Exception as exc:ep=dict(seed=seed,complete=False,success=None,reason='error',error_type=type(exc).__name__,traceback=traceback.format_exc())
            ep.update(repeat=repeat,condition=condition,queries=wrapper.query_count,trace_stem=stem)
            episodes.append(ep);replace_json(run/'episodes.json',episodes);print('EPISODE',seed,repeat,condition,ep.get('success'),ep.get('executed_actions'),flush=True);signal.alarm(0)
            if ep.get('reason')=='error':break
        result=summarize(run);result['elapsed_s']=time.monotonic()-began;write_json(run/'result.json',result)
        write_json(run/'source_runtime_audit.json',storage.verify_source());replace_json(run/'progress.json',dict(phase='finished',interface_passed=result['interface_passed']))
        for c,v in result['observed_outcomes'].items():print(args.task,c,'success',str(v['successes'])+'/'+str(v['completed']),'rate',v['ratio'],'censored',v['censored'],flush=True)
        require(result['interface_passed'],'Local numeric/prefix audit failed; inspect result')
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed_records=len(episodes)));raise
    finally:signal.alarm(0)


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--prepare-protocol',action='store_true');ap.add_argument('--protocol',type=Path)
    ap.add_argument('--task',choices=TASKS);ap.add_argument('--gpu',default='0');ap.add_argument('--max-chunks',type=int);ap.add_argument('--scene-timeout',type=int,default=1800);ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.prepare_protocol:prepare_protocol();return
    require(args.protocol is not None and args.task is not None and args.gpu.isdigit(),'Protocol/task/GPU required')
    require(args.max_chunks is None or 5<=args.max_chunks<=6,'Smoke must cover window return-to-zero:5..6 queries')
    args.protocol=output_path(args.protocol)
    from .native_entry_storage import NativeEntryStorage
    from .evaluate_structured_low_storage import _terminate
    storage=NativeEntryStorage()
    if not args.dry_run:
        atexit.register(storage.cleanup);signal.signal(signal.SIGTERM,_terminate);os.environ['CUDA_VISIBLE_DEVICES']=args.gpu;os.environ['JAX_PLATFORMS']='cpu'
    execute(args,storage)


if __name__=='__main__':main()
