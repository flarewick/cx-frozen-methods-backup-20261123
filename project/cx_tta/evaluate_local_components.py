"""Fixed cup window: same-process arm/gripper2x2, frozenA and native budget."""
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
from .local_component_intervention import CONDITIONS,FACTORS,GRIPPERS,LocalComponentPolicy,component_actions
from .local_component_analysis import episode_order,summarize
from .structured_action_adapter import ARM_INDICES
from .evaluate_local_intervention import LocalTrace
from .evaluate_structured_attribution import PhysicalPolicy,pilot_reference,steady_probe
from .train_output_support import BASE,read_checkpoint

TRAIN=ROOT/'outputs/effect_learning/output_support_20261005_082307_793921'
DESIGN=ROOT/'outputs/reports/attribution/local_intervention_accepted_20261005_152036_339948/next_experiments.json'
TASK='place_empty_cup'


def real_input_preflight(protocol):
    import torch
    from .native_history_cache import arrays
    from .structured_attribution_analysis import _load_trace
    from .output_support_adapter import OutputSupportAdapter,validate_native_arms
    p=read(protocol);verify(p['source_sha256'])
    payload=read_checkpoint(TRAIN/'A_bounded_clipped_seed0.pt','A_bounded_clipped')
    model=OutputSupportAdapter(32,payload['metadata']['geometry_mean'],'A_bounded_clipped').eval().requires_grad_(False)
    model.load_state_dict(payload['model']);source=Path(p['accepted_temporal_run']);packets=c11_matches=0
    for ep in read(source/'episodes.json'):
        if ep['repeat']!=0 or ep['condition'] not in ('L0','A_close_window'):continue
        t=_load_trace(source,ep);cap=arrays(source/'legal_inputs'/(ep['trace_stem']+'.npz'));indices=cap.pop('query_indices').tolist()
        for j,i in enumerate(indices):
            base=cap['base'][j].copy();x={k:torch.from_numpy(v[j].copy()) for k,v in cap.items() if k!='base'}
            with torch.inference_mode():
                out=model(x,torch.from_numpy(base),enabled=True);zero=model(x,torch.from_numpy(base),enabled=False)
            raw=out['delta'][0].numpy().copy();b=base[0]
            expected_both=out['actions'][0].numpy() if p['window'][0]<=i<p['window'][1] else b
            for c in CONDITIONS:
                a,d,arm,grip,_=component_actions(b,raw,c,i,p['window'])
                validate_native_arms(a,p['robot_joint_limits'])
                if c=='C00_zero':require(a.tobytes()==zero['actions'][0].numpy().tobytes(),'Real00 not exact original model bypass')
                if c=='C11_both':require(a.tobytes()==expected_both.tobytes(),'Real11 not original local A assembly')
                for factor,cols in ((arm,ARM_INDICES),(grip,GRIPPERS)):
                    if not factor:require(a[:,cols].tobytes()==b[:,cols].tobytes(),'Real disabled column modified')
            if ep['condition']=='A_close_window':
                require(np.max(np.abs(raw-t['arrays']['raw_delta'][i]))<2e-5,'OriginalA actual packet reload')
                legacy=t['arrays']['base'][i]+t['arrays']['applied_delta'][i] if t['rows'][i]['window_active'] else t['arrays']['base'][i]
                require(expected_both.tobytes()==legacy.tobytes(),'11 not saved old close-window command')
                c11_matches+=1
            packets+=1
    require(packets>0 and c11_matches>0,'Actual packet coverage empty')
    write_json(Path(protocol).parent/'CPU_preflight.json',dict(passed=True,GPU_started=False,actual_source_packets=packets,
        original_close_commands_reproduced=c11_matches,zero_control_semantic_exact=True,original_local11_semantic_exact=True,
        independently_disabled_channels_exact=True,robot_domain_unchanged=True,
        source_sha256={str(protocol):digest(Path(protocol))}))
    print('REAL_CPU_PREFLIGHT_PASS',packets,c11_matches,flush=True)


def prepare_protocol():
    d=read(DESIGN);verify(d['source_sha256']);require(d['status']=='design_prepared_not_executed','Sealed unexecuted component design')
    require(d['window']==[1,3] and d['scene_seeds']==[100380,100382] and d['primary_next_task']==TASK,'Fixed design/window/scenes')
    accepted=read(Path(d['accepted_results']));require(accepted['status']=='passed','Accepted local evidence')
    oldrun=Path(accepted['tasks'][TASK]['run']);parent=read(read(oldrun/'manifest.json')['protocol'])
    require(all(parent['windows'][TASK][str(s)]['A_close_window']==d['window'] for s in d['scene_seeds']),'Keep original window')
    _,_,_,reference=pilot_reference(TASK,BASE);q=read(TRAIN/'training_protocol.json')
    budget_file=Path(reference['run'])/'robotwin/task_config/_eval_step_limit.yml'
    paths=[DESIGN,Path(d['accepted_results']),TRAIN/'A_bounded_clipped_seed0.pt',TRAIN/'training_audit.json',budget_file,
        Path(__file__),ROOT/'cx_tta/local_component_intervention.py',ROOT/'cx_tta/local_component_analysis.py',
        ROOT/'scripts/eval_local_components.sh',ROOT/'tests/test_local_component_intervention.py',
        ROOT/'cx_tta/evaluate_local_intervention.py',ROOT/'cx_tta/evaluate_structured_attribution.py',
        ROOT/'cx_tta/output_support_policy.py',ROOT/'cx_tta/native_entry_storage.py']
    sources=dict(q['source_sha256']);sources.update({str(f):digest(f) for f in paths})
    out=new_run('local_components',group=('reports','protocols'))
    p=dict(stage='local_component_factorial',status='frozen_before_GPU_validation',design=str(DESIGN),task=TASK,
        conditions=d['conditions'],window=d['window'],seeds=d['scene_seeds'],repeats=2,training_run=str(TRAIN),
        checkpoint_sha256=digest(TRAIN/'A_bounded_clipped_seed0.pt'),reference=reference,
        accepted_temporal_run=str(oldrun),arm_indices=list(ARM_INDICES),gripper_indices=list(GRIPPERS),
        budget_config_sha256=digest(budget_file),scales=q['reference_scales_native_units'],robot_joint_limits=q['robot_joint_limits'],
        raw_proposal='All4 cells compute exactly same frozenA head on their reached state; both applied factors only0/1',
        disabled_semantics='Exact current base bytes on every disabled column; full identity outside window',
        native_budget='Original task yml and every real native counter/terminal checked; no resets',
        compiled=True,internal_noise=True,per_action_observations=True,shared_base_models=1,
        unchanged=['Weights/PCA/9 fields+actual base/history/geometry prior',
            'Original URDF and native per-tick gripper clipping', 'Original total step budget and success predicates',
            'Window2queries/64actions/no temporal shift or smoothing'],
        no_new_training=True,no_router_or_TTA=True,no_RGB_or_video=True,
        joint_index_masks_not_task_arm_selection=True,source_sha256=sources,
        scientific_scope='Two reused cup development scenes; offline fixed-window factor diagnostic, not learned gate/population performance')
    write_json(out/'protocol.json',p);print('PROTOCOL',out/'protocol.json',flush=True)
    real_input_preflight(out/'protocol.json');return out/'protocol.json'


def execute(args,storage):
    p=read(args.protocol);verify(p['source_sha256']);require(p['stage']=='local_component_factorial' and args.task==p['task']==TASK,'Registered task/stage')
    require(p['conditions']=={c:dict(arm_strength=a,gripper_strength=g) for c,(a,g) in FACTORS.items()},'Registered component factors')
    require(read(args.protocol.parent/'CPU_preflight.json')['passed'],'CPU semantic gate')
    require(digest(TRAIN/'A_bounded_clipped_seed0.pt')==p['checkpoint_sha256'],'FrozenA checkpoint')
    _,reference,_,evidence=pilot_reference(TASK,BASE)
    seeds=p['seeds'][:1] if args.max_chunks is not None else p['seeds'];schedule=episode_order(seeds,args.max_chunks is not None)
    manifest=dict(stage=p['stage'],task=TASK,protocol=str(args.protocol),protocol_sha256=digest(args.protocol),
        training_run=str(TRAIN),seeds=seeds,conditions=list(CONDITIONS),window=p['window'],max_chunks=args.max_chunks,
        order=[dict(seed=s,repeat=r,condition=c) for s,r,c in schedule],no_video=True,compiled=True,
        original_native_budget=True,noise_source='model_internal_torch',shared_base_models=1,shared_compiled_samplers=1,
        scope=p['scientific_scope'])
    if args.dry_run:
        print('DRY_RUN',dict(manifest=manifest,requested_rollouts=len(schedule),GPU_started=False),flush=True);return
    import torch
    from .cache_expert import create_base_policy
    from .evaluate_b2 import configure_inference,resolve_tokenizer,run_episode
    from .evaluate_structured_action import check_training_policy_identity,SmokeBudgetReached
    from .output_support_policy import OutputSupportPolicy
    from .collect_smoke import make_config
    require(torch.cuda.is_available(),'CUDA required before creating run')
    run=new_run('paired_smoke' if args.max_chunks is not None else 'paired',group=('diagnostics','local_components',TASK));print('RUN',run,flush=True)
    write_json(run/'manifest.json',manifest);episodes=[];began=time.monotonic()
    try:
        tokenizer=resolve_tokenizer();identity=check_training_policy_identity(BASE,tokenizer)
        storage.runtime_source=Path(evidence['run'])/'robotwin';runtime=storage.prepare_runtime(run);storage.configure_runtime(run,runtime,tokenizer)
        args.task_config,args.instruction_type='demo_randomized','unseen';config=make_config(runtime,args,run)
        config.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0);write_json(run/'resolved_config.json',config)
        old=read(Path(evidence['run'])/'provenance.json');rt={str(f.relative_to(runtime)):digest(f) for f in runtime.rglob('*.py') if '__pycache__' not in f.parts}
        require(rt==old['runtime_sha256'],'Frozen runtime identity')
        for path,sha in old['code_sha256'].items():require(digest(Path(path))==sha,'Frozen source changed')
        write_json(run/'provenance.json',dict(runtime_sha256=rt,source_sha256={str(f):digest(f) for f in (ROOT/'cx_tta').glob('*.py')},
            identity=identity['live_identity'],policy_source_sha256=identity['policy_source_sha256']))
        task=getattr(importlib.import_module('envs.'+TASK),TASK)();configure_inference('compiled',0);torch.set_num_threads(2)
        replace_json(run/'progress.json',dict(phase='loading_shared_base',completed=0,requested=len(schedule)))
        base=create_base_policy();base._model.eval().requires_grad_(False)
        wrappers={c:LocalComponentPolicy(base,TRAIN,c,p['window']) for c in CONDITIONS}
        for wrapper in wrappers.values():wrapper.check_robot_config(config)
        true_l0=OutputSupportPolicy(base,TRAIN,'L0')
        write_json(run/'hardware.json',dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
            compiled=hasattr(base._sample_actions,'_torchdynamo_orig_callable'),global_dynamo_disabled=bool(torch._dynamo.config.disable),
            shared_sampler=all(w.sampler is base._sample_actions for w in wrappers.values()),sample_kwargs=base._sample_kwargs))
        def timeout(*_):raise TimeoutError('Registered local component scene/probe timeout')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(args.scene_timeout)
        replace_json(run/'progress.json',dict(phase='common_warmup',completed=0,requested=len(schedule)))
        probe=steady_probe(task,config,seeds[0],reference[seeds[0]]['instruction'],{'L0':true_l0,'S0_zero':wrappers['C00_zero']})
        np.savez_compressed(run/'steady_probe.npz',base=probe.pop('base_values'));write_json(run/'steady_probe.json',probe)
        require(probe['passed'],'ActualL0 vs00 exact fixed-input probe');signal.alarm(0)
        for seed,repeat,condition in schedule:
            signal.alarm(args.scene_timeout);wrapper=wrappers[condition];wrapper.reset_history();wrapper.episode_seed=seed
            replace_json(run/'progress.json',dict(phase='local_component_factorial',completed=len(episodes),requested=len(schedule),seed=seed,condition=condition))
            physical=PhysicalPolicy(wrapper,task,args.max_chunks);stem=f'seed{seed}_r{repeat}_{condition}';trace=LocalTrace(run,stem,physical)
            try:
                ep=run_episode(task,config,runtime,seed,seeds.index(seed),physical,True,instruction_type='unseen',instruction_override=reference[seed]['instruction'],trace=trace,rollout_protocol='official')
                ep['complete']=True
                if args.max_chunks is not None:ep['observed_native_success']=ep.pop('success');ep.update(success=None,complete=False,reason='component_smoke_no_rate_claim')
            except SmokeBudgetReached as exc:ep=exc.record
            except Exception as exc:ep=dict(seed=seed,eligible=True,complete=False,success=None,reason='error',error_type=type(exc).__name__,traceback=traceback.format_exc())
            ep.update(repeat=repeat,condition=condition,queries=wrapper.query_count,trace_stem=stem)
            episodes.append(ep);replace_json(run/'episodes.json',episodes);print('EPISODE',seed,repeat,condition,ep.get('success'),ep.get('executed_actions'),flush=True);signal.alarm(0)
            if ep.get('reason')=='error':break
        result=summarize(run);result['elapsed_s']=time.monotonic()-began;write_json(run/'result.json',result)
        write_json(run/'source_runtime_audit.json',storage.verify_source());replace_json(run/'progress.json',dict(phase='finished',interface_passed=result['interface_passed']))
        for c,v in result['observed_outcomes'].items():
            rate='N/A' if v['ratio'] is None else f"{v['ratio']:.1%}"
            print(TASK,c,'success',f"{v['successes']}/{v['completed']}",'rate',rate,'censored',v['censored'],flush=True)
        require(result['interface_passed'],'Component numeric/prefix audit failed; inspect result')
    except BaseException:
        if not (run/'failure.json').exists():write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed_records=len(episodes)))
        raise
    finally:signal.alarm(0)


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--prepare-protocol',action='store_true');ap.add_argument('--protocol',type=Path)
    ap.add_argument('--task',choices=(TASK,),default=TASK);ap.add_argument('--gpu',default='0');ap.add_argument('--max-chunks',type=int)
    ap.add_argument('--scene-timeout',type=int,default=1800);ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.prepare_protocol:prepare_protocol();return
    require(args.protocol is not None and args.gpu.isdigit() and args.scene_timeout>=60,'Protocol/GPU/timeout required')
    require(args.max_chunks is None or args.max_chunks in (5,6),'Smoke must observe return-to-zero at5..6queries')
    args.protocol=output_path(args.protocol)
    from .native_entry_storage import NativeEntryStorage
    from .evaluate_structured_low_storage import _terminate
    storage=NativeEntryStorage()
    if not args.dry_run:
        atexit.register(storage.cleanup);signal.signal(signal.SIGTERM,_terminate);os.environ['CUDA_VISIBLE_DEVICES']=args.gpu;os.environ['JAX_PLATFORMS']='cpu'
    execute(args,storage)


if __name__=='__main__':main()
