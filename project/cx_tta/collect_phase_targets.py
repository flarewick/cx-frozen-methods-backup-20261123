"""Independent TRAIN paired bounded phase-target collection.
Offline point selection/expert references never enter neural forward.
"""
import argparse,atexit,importlib,os,signal,time,traceback
from pathlib import Path
import numpy as np
from .paths import ROOT,new_run,write_json,replace_json,output_path
from .native_history_cache import read,require,verify
from .audit import digest
from .phase_target_policy import AllQuerySource,PhaseGripperPolicy,functional_state,CASES
from .causal_arm_protocol import TRAIN
from .geometry_labels import episode_definition
from .local_benefit_cache import AllQueryCache
from .evaluate_structured_attribution import CompactTrace,PhysicalPolicy
from .attribution_physics import physical_snapshot
from .eval_diagnostics import json_hash,array_hash
from .recovery_contract import rng_capture,rng_restore,rng_fingerprint,execute_chunk
from .deploy import unpack_observation,POLICY_CAMERA_KEYS

def observation_hash(obs):
    images,state=unpack_observation(obs)
    return json_hash({**{k:array_hash(v) for k,v in zip(POLICY_CAMERA_KEYS,images)},'command_state':array_hash(state)})

class TargetTrace(CompactTrace):
    def __init__(self,run,stem,physical,taskname,defs,names):
        super().__init__(run,stem,physical);self.cache=AllQueryCache(run/'legal_cache'/stem)
        self.defs,self.names,self.taskname=defs,names,taskname;self.rows=[];self.controls=[]
    def record(self,**kwargs):
        super().record(**kwargs);w=self.policy.policy;before=self.policy.before;after=physical_snapshot(self.policy.task)
        row=dict(w.last_diagnostics,native_before_sha256=json_hash(before),native_after_sha256=json_hash(after),
            executed=kwargs['executed'],observation_sha256=observation_hash(kwargs['observation']))
        self.rows.append(row);self.controls.append(kwargs['actions'].copy())
        self.cache.append(w.query_count-1,w.last_legal_inputs,w.last_numeric['raw_delta'],
            kwargs['delta'],kwargs['actions'],kwargs['executed'],
            dict(before=functional_state(before,self.taskname,self.defs,self.names),
                 after=functional_state(after,self.taskname,self.defs,self.names),scope='offline_only,notquerybenefit'))

def start(task,cfg,reg,run):
    from .evaluate_b2 import seed_all
    seed_all(reg['seed']);task.setup_demo(now_ep_num=reg['scene_index'],seed=reg['seed'],
        **dict(cfg,need_plan=True,save_data=False,is_test=True,save_path=str(run/'scene')))
    task.set_instruction(instruction=reg['instruction']);obs=task.get_obs();native=physical_snapshot(task)
    from .collect_recovery import signature
    require(json_hash(native)==reg['initial_physical_sha256'] and signature(obs)==reg['initial_observation_signature'],'Frozen scene initial signatures')
    return obs

def episode_record(seed,condition,w,task,trace,complete,point=None):
    return dict(seed=seed,condition=condition,complete=complete,success=bool(task.eval_success) if complete else None,
        executed_actions=int(task.take_action_cnt),queries=w.query_count,trace_stem=trace.stem,
        cache=str(trace.cache.folder),point_query=None if point is None else point['query'])

def save_point(run,stem,point):
    import torch
    folder=run/'points'/stem;folder.mkdir(parents=True)
    old=point['previous']
    np.savez_compressed(folder/'history.npz',joint=np.empty(0) if old is None else old[0],state=np.empty(0) if old is None else old[1])
    rng=point['rng'];torch.save(dict(cpu=rng['torch_cpu'],cuda=rng['torch_cuda']),folder/'rng_tensors.pt')
    write_json(folder/'rng_meta.json',dict(python=rng['python'],numpy=dict(name=rng['numpy'][0],keys=rng['numpy'][1].tolist(),position=rng['numpy'][2],gaussian=rng['numpy'][3],cached=rng['numpy'][4])))
    write_json(folder/'point.json',{k:v for k,v in point.items() if k not in ('rng','previous','base','raw')})
    np.savez_compressed(folder/'reference_query.npz',base=point['base'],raw_delta=point['raw'])

def source_rollout(task,cfg,reg,run,w,defs,names,mode,maxqueries):
    from .evaluate_b2 import step_chunk_official
    w.reset_history();w.episode_seed=reg['seed'];obs=start(task,cfg,reg,run)
    stem=f"seed{reg['seed']}_{w.condition}";physical=PhysicalPolicy(w,task)
    trace=TargetTrace(run,stem,physical,cfg['task_name'],defs,names);point=None
    try:
        while task.take_action_cnt<task.step_lim and not task.eval_success:
            q=w.query_count
            if maxqueries is not None and q>=maxqueries:break
            native=physical_snapshot(task);state=functional_state(native,cfg['task_name'],defs,names)
            window=w.detector.window
            forced=mode=='smoke' and q==1
            eligible=mode=='full' and window is not None and q>=window[1] and state['geometry_satisfied'] and not state['native_success']
            proposal=None
            if point is None and (forced or eligible):
                proposal=dict(query=q,native_count=int(task.take_action_cnt),native_before_sha256=json_hash(native),
                    observation_sha256=observation_hash(obs),rng=rng_capture(),
                    previous=None if w.previous is None else tuple(v.copy() for v in w.previous),
                    source_condition=w.condition,source_state=state,forced_interface_only=forced)
            _,_,obs=step_chunk_official(task,physical,obs,reg['instruction'],True,trace=trace)
            if proposal is not None:
                row=trace.rows[-1];proposal.update(base=w.last_numeric['base'].copy(),raw=w.last_numeric['raw_delta'].copy(),
                    structured_input_sha256=row['structured_input_sha256'],rng_before_query=row['rng_before_query'],
                    rng_after_sampler=row['rng_after_sampler'],base_chunk_sha256=row['base_chunk_sha256'],
                    raw_delta_sha256=row['raw_delta_sha256'],prefix_queries=q)
                require(rng_fingerprint(proposal['rng'])==proposal['rng_before_query'],'Point RNG capture exact')
                require(not row['window_active'],'Late point baseline must already have no residual')
                point=proposal
            replace_json(run/'progress.json',dict(phase='source',seed=reg['seed'],condition=w.condition,queries=w.query_count,native_count=task.take_action_cnt))
        complete=mode=='full' and (task.eval_success or task.take_action_cnt==task.step_lim)
        trace.cache.finish(complete=bool(complete),success=bool(task.eval_success) if complete else None,role='diagnostic_confirmation' if mode=='smoke' else 'independent_train_candidate')
        ep=episode_record(reg['seed'],w.condition,w,task,trace,bool(complete),point)
        if point is not None:
            point.update(source_episode=ep,reference_teacher_grippers=reg['expert_terminal_grippers'])
            save_point(run,stem,point)
        return ep,trace,point
    finally:task.close_env()

def probe_rollout(task,cfg,reg,run,w,defs,names,source,point,mode,maxqueries):
    from .evaluate_b2 import step_chunk_official
    w.reset_history();w.episode_seed=reg['seed'];w.point_query=point['query']
    w.teacher_goal=np.asarray(reg['expert_terminal_grippers'],np.float32) if w.condition=='T_stage_grip' else None
    obs=start(task,cfg,reg,run)
    try:
        for i in range(point['query']):
            row=source.rows[i];require(json_hash(physical_snapshot(task))==row['native_before_sha256'] and observation_hash(obs)==row['observation_sha256'],'Replay prefix native/RGB')
            obs,n=execute_chunk(task,source.controls[i]);require(n==row['executed'] and json_hash(physical_snapshot(task))==row['native_after_sha256'],'Replay prefix actual clock/physics')
        require(json_hash(physical_snapshot(task))==point['native_before_sha256'] and observation_hash(obs)==point['observation_sha256'],'Exact selected fork')
        w.previous=None if point['previous'] is None else tuple(v.copy() for v in point['previous']);w.query_count=point['query']
        rng_restore(point['rng']);require(rng_fingerprint(rng_capture())==point['rng_before_query'],'Exact restored sampler state')
        stem=f"seed{reg['seed']}_{point['source_condition']}_{w.condition}"
        physical=PhysicalPolicy(w,task);trace=TargetTrace(run,stem,physical,cfg['task_name'],defs,names)
        # Cache history starts at a native mid-episode query; use a separate retained
        # prefix of actual observable packets without inventing teacher history.
        with np.load(source.cache.folder/'queries.npz',allow_pickle=False) as z:
            offline=read(source.cache.folder/'offline_labels.json')['queries']
            for i in range(point['query']):
                packet={k:z[k][i:i+1].copy() for k in w.model_forward_keys} if hasattr(w,'model_forward_keys') else None
                from .structured_action_adapter import FORWARD_KEYS
                packet={k:z[k][i:i+1].copy() for k in FORWARD_KEYS|{'base'}}
                trace.cache.append(i,packet,z['raw_delta'][i],z['applied_delta'][i],z['actions'][i],int(z['valid'][i].sum()),offline[i])
        first=True
        while task.take_action_cnt<task.step_lim and not task.eval_success:
            if maxqueries is not None and w.query_count>=maxqueries:break
            _,_,obs=step_chunk_official(task,physical,obs,reg['instruction'],True,trace=trace)
            if first:
                row=trace.rows[0]
                for key in ('structured_input_sha256','rng_before_query','rng_after_sampler','base_chunk_sha256','raw_delta_sha256','native_before_sha256','observation_sha256'):
                    require(row[key]==point[key],'Paired actual fork changed: '+key)
                first=False
            replace_json(run/'progress.json',dict(phase='target_probe',seed=reg['seed'],condition=w.condition,queries=w.query_count,native_count=task.take_action_cnt))
        require(not first,'Fork probe not executed')
        complete=mode=='full' and (task.eval_success or task.take_action_cnt==task.step_lim)
        trace.cache.finish(complete=bool(complete),success=bool(task.eval_success) if complete else None,role='diagnostic_confirmation' if mode=='smoke' else 'independent_train_candidate')
        ep=episode_record(reg['seed'],w.condition,w,task,trace,bool(complete),point);ep.update(source_condition=point['source_condition'],prefix_queries=point['query'],probe_queries=len(trace.rows))
        return ep
    finally:task.close_env()

def registry_check(protocol,registry):
    p=read(protocol);m=read(registry/'manifest.json');r=read(registry/'result.json')
    require(m['protocol_sha256']==digest(protocol) and r['status']=='complete' and r['all_fixed_seeds_attempted'] and r['policy_sampler_calls']==0,'Prospective complete registry')
    verify(r['source_sha256']);rows=read(registry/'records.json')
    require({(r['task'],r['seed']) for r in rows}=={(t,s) for t in p['tasks'] for s in p['seeds']},'No missing/replaced attempts')
    for row in rows:
        if row['eligible']:require(len(row['expert_terminal_grippers'])==2 and all(0<=v<=1 for v in row['expert_terminal_grippers']),'Source-backed expert goals')
    return rows

def execute(a,storage):
    p=read(a.protocol);verify(p['source_sha256']);require(p['stage']=='phase_target_qualification','Phase-target protocol')
    regs=registry_check(a.protocol,a.registry);seeds=[s for s in p['seeds'] if next(r for r in regs if r['task']==a.task and r['seed']==s)['eligible']]
    if a.mode=='smoke':seeds=seeds[:1]
    if a.dry_run:print('PHASE_DRYRUN',a.task,seeds,a.mode,'maxfull',len(seeds)*6,'GTonlyoffline,9fields,perquerycapture');return
    import torch
    from .evaluate_b2 import resolve_tokenizer,configure_inference
    from .cache_expert import create_base_policy
    from .collect_smoke import make_config
    from .evaluate_structured_attribution import steady_probe
    from .output_support_policy import OutputSupportPolicy
    require(torch.cuda.is_available() and bool(seeds),'CUDA/eligiblefixedscenes')
    run=new_run('smoke' if a.mode=='smoke' else 'batch',group=('phase_targets',a.task));print('RUN',run,flush=True)
    (run/'legal_cache').mkdir();eps=[];panels=[];start_time=time.monotonic()
    write_json(run/'manifest.json',dict(stage=p['stage'],protocol=str(a.protocol),protocol_sha256=digest(a.protocol),registry=str(a.registry),
        registry_records_sha256=digest(a.registry/'records.json'),task=a.task,seeds=seeds,attempted_seeds=p['seeds'],mode=a.mode,
        max_source_queries=5 if a.mode=='smoke' else None,max_full_rollouts=len(seeds)*6,
        no_RGB_or_video=True,training_approved=False,source_modes=['Z_zero','A_causal_arm'],
        source_selection='First geometric incomplete goal after initial armwindow in EACH prescribedsource; no outcomes used to pickpoint',teacher_privileged=True))
    try:
        storage.runtime_source=Path(p['reference'][a.task]['run'])/'robotwin';runtime=storage.prepare_runtime(run);storage.configure_runtime(run,runtime,resolve_tokenizer())
        cfg=make_config(runtime,argparse.Namespace(task=a.task,task_config='demo_randomized'),run)
        cfg.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0);write_json(run/'resolved_config.json',cfg)
        original=read(Path(p['reference'][a.task]['run'])/'provenance.json')
        require({str(f.relative_to(runtime)):digest(f) for f in runtime.rglob('*.py') if '__pycache__' not in f.parts}==original['runtime_sha256'],'Original frozen runtime')
        configure_inference('compiled',0);torch.set_num_threads(2);base=create_base_policy();base._model.eval().requires_grad_(False)
        sources={c:AllQuerySource(base,TRAIN,c) for c in ('Z_zero','A_causal_arm')}
        probes={c:PhaseGripperPolicy(base,TRAIN,c) for c in ('A_grip','T_stage_grip')}
        for w in [*sources.values(),*probes.values()]:w.check_robot_config(cfg)
        task=getattr(importlib.import_module('envs.'+a.task),a.task)()
        write_json(run/'hardware.json',dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
            compiled=hasattr(base._sample_actions,'_torchdynamo_orig_callable'),global_dynamo_disabled=bool(torch._dynamo.config.disable),
            sample_kwargs=base._sample_kwargs,shared_sampler=all(w.sampler is base._sample_actions for w in [*sources.values(),*probes.values()])))
        firstreg=next(r for r in regs if r['task']==a.task and r['seed']==seeds[0])
        zero=OutputSupportPolicy(base,TRAIN,'A_bounded_clipped',force_zero=True);true_l0=OutputSupportPolicy(base,TRAIN,'L0')
        probe=steady_probe(task,cfg,seeds[0],firstreg['instruction'],{'L0':true_l0,'S0_zero':zero})
        np.savez_compressed(run/'steady_probe.npz',base=probe.pop('base_values'));write_json(run/'steady_probe.json',probe);require(probe['passed'],'Commonzero originalL0')
        for si,seed in enumerate(seeds):
            reg=next(r for r in regs if r['task']==a.task and r['seed']==seed);defs,deps=episode_definition(runtime,a.task,reg['grounded_objects'])
            names=[reg['grounded_objects']['{A}'].split('/')[0],reg['grounded_objects']['{B}'].split('/')[0] if a.task=='place_empty_cup' else 'box']
            source_order=['Z_zero'] if a.mode=='smoke' else list(sources) if si%2==0 else list(reversed(sources))
            for cond in source_order:
                e,trace,point=source_rollout(task,cfg,reg,run,sources[cond],defs,names,a.mode,5 if a.mode=='smoke' else None)
                eps.append(e);replace_json(run/'episodes.json',eps)
                if point is None:
                    panels.append(dict(seed=seed,source_condition=cond,point_absent=True,qualified_teacher_recovery=False));continue
                children={}
                for case in (('A_grip','T_stage_grip') if si%2==0 else ('T_stage_grip','A_grip')):
                    child=probe_rollout(task,cfg,reg,run,probes[case],defs,names,trace,point,a.mode,5 if a.mode=='smoke' else None)
                    eps.append(child);children[case]=child;replace_json(run/'episodes.json',eps)
                full=all(x['complete'] for x in [e,*children.values()])
                panels.append(dict(seed=seed,source_condition=cond,point_query=point['query'],native_count=point['native_count'],
                    forced_interface_only=point['forced_interface_only'],outcomes={'C_base':e['success'],**{k:v['success'] for k,v in children.items()}},
                    qualified_teacher_recovery=bool(full and not e['success'] and children['T_stage_grip']['success']),
                    teacher_preserves_success=bool(full and e['success'] and children['T_stage_grip']['success']),
                    teacher_harms_success=bool(full and e['success'] and not children['T_stage_grip']['success']),
                    teacher_success_not_adapter_improvement=True,paired_fork_checked=True))
                replace_json(run/'panels.json',panels)
        result=dict(status='complete',interface_passed=True,mode=a.mode,records=len(eps),full_rollouts=sum(e['complete'] for e in eps),
            censored=sum(not e['complete'] for e in eps),panels=panels,qualified_teacher_recoveries=sum(p.get('qualified_teacher_recovery',False) for p in panels),
            distinct_qualified_scenes=len({p['seed'] for p in panels if p.get('qualified_teacher_recovery')}),
            training_approved=False,teacher_success_not_adapter_improvement=True,elapsed_s=time.monotonic()-start_time,
            source_sha256={str(a.protocol):digest(a.protocol),str(a.registry/'records.json'):digest(a.registry/'records.json')})
        write_json(run/'result.json',result);write_json(run/'source_runtime_audit.json',storage.verify_source())
        replace_json(run/'progress.json',dict(phase='finished',records=len(eps)));print('PHASE_TARGET_DONE',result['qualified_teacher_recoveries'],'teacherrecoveries;',result['censored'],'censored',flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),records=len(eps),training_approved=False));raise

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--task',choices=('place_empty_cup','beat_block_hammer'),required=True);p.add_argument('--mode',choices=('full','smoke'),default='full')
    p.add_argument('--gpu',default='0');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    a.protocol=output_path(a.protocol);a.registry=output_path(a.registry);require(a.gpu.isdigit(),'Numeric GPU')
    if a.dry_run:execute(a,None);return
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;os.environ['JAX_PLATFORMS']='cpu'
    from .native_entry_storage import NativeEntryStorage
    from .evaluate_structured_low_storage import _terminate
    storage=NativeEntryStorage();atexit.register(storage.cleanup);signal.signal(signal.SIGTERM,_terminate)
    def timeout(*args):raise TimeoutError('Stagequalification declared3hour processbound')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(10800)
    try:execute(a,storage)
    finally:signal.alarm(0)
if __name__=='__main__':main()

