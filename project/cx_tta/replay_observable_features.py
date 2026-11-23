"""Recover observable features by exact recorded-command replay, no VLA sampler."""
import argparse,atexit,importlib,os,signal,time,traceback
from pathlib import Path
import numpy as np
from .paths import ROOT,new_run,output_path,write_json,replace_json
from .native_history_cache import read,require,verify,BASE
from .audit import digest
from .structured_attribution_analysis import _load_trace
from .structured_action_adapter import ARM_INDICES
from .deploy import unpack_observation,POLICY_CAMERA_KEYS
from .eval_diagnostics import array_hash,json_hash
from .structured_features import pack_observable
from .local_benefit_cache import AllQueryCache
from .attribution_physics import physical_snapshot

def execute(args,storage):
    m=read(args.source/'manifest.json');saved=read(args.source/'result.json')
    require(saved['gate_passed'] and m['stage']=='causal_arm_confirmation','Accepted full original panel')
    verify(saved['source_sha256']);episode=next(e for e in read(args.source/'episodes.json') if e['seed']==args.seed and e['condition']==args.condition)
    t=_load_trace(args.source,episode);limit=min(args.max_queries,len(t['rows']))
    import torch
    from .cache_expert import create_base_policy
    from .cache_joint_prefix import encode
    from .robot_kinematics import RobotFK
    from .evaluate_b2 import resolve_tokenizer,seed_all,configure_inference
    from .collect_smoke import make_config
    from .recovery_contract import rng_capture,rng_fingerprint,execute_chunk
    from .train_output_support import read_checkpoint
    from .output_support_adapter import OutputSupportAdapter
    require(torch.cuda.is_available(),'CUDA visible')
    run=new_run('replay',group=('diagnostics','feature_recovery',m['task']));print('RUN',run,flush=True)
    manifest=dict(source=str(args.source),source_result_sha256=digest(args.source/'result.json'),source_trace_stem=episode['trace_stem'],
        task=m['task'],seed=args.seed,condition=args.condition,max_queries=limit,
        training_approved=False,scope='Existing confirmation trajectory feature reconstruction only,not newSR orTRAIN',no_RGB_or_video=True,VLA_sampler_calls=0,
        code_sha256={str(ROOT/'cx_tta'/n):digest(ROOT/'cx_tta'/n) for n in ('replay_observable_features.py','local_benefit_cache.py','structured_features.py','cache_joint_prefix.py')})
    write_json(run/'manifest.json',manifest);task=None;setup=False;start=time.monotonic()
    try:
        storage.runtime_source=args.source/'robotwin';runtime=storage.prepare_runtime(run);storage.configure_runtime(run,runtime,resolve_tokenizer())
        taskargs=argparse.Namespace(task=m['task'],task_config='demo_randomized')
        cfg=make_config(runtime,taskargs,run);cfg.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
        write_json(run/'resolved_config.json',cfg);prov=read(args.source/'provenance.json')
        require({str(f.relative_to(runtime)):digest(f) for f in runtime.rglob('*.py') if '__pycache__' not in f.parts}==prov['runtime_sha256'],'Source runtime unchanged')
        configure_inference('compiled',0);torch.set_num_threads(2)
        replace_json(run/'progress.json',dict(phase='loading_frozen_encoder',queries=0,total=limit))
        base=create_base_policy();base._model.eval().requires_grad_(False)
        def forbidden(*a,**k):raise AssertionError('Recorded replay must never query VLA actions')
        base.infer=forbidden;base._sample_actions=forbidden
        payload=read_checkpoint(Path(read(m['protocol'])['training_run'])/'A_bounded_clipped_seed0.pt','A_bounded_clipped')
        model=OutputSupportAdapter(32,payload['metadata']['geometry_mean'],'A_bounded_clipped').eval().requires_grad_(False);model.load_state_dict(payload['model'])
        projection={k:v.copy() for k,v in np.load(payload['metadata']['projection_path']).items()};robots=[RobotFK(cfg,s) for s in ('left','right')]
        task=getattr(importlib.import_module('envs.'+m['task']),m['task'])()
        registry=read(Path(m['registry'])/'records.json');reg=next(r for r in registry if r['task']==m['task'] and r['seed']==args.seed)
        opts=dict(cfg,need_plan=True,save_data=False,is_test=True,save_path=str(run/'scene'))
        seed_all(args.seed);setup=True;task.setup_demo(now_ep_num=reg['scene_index'],seed=args.seed,**opts);task.set_instruction(instruction=episode['instruction'])
        obs=task.get_obs();previous=None;cache=AllQueryCache(run/'observable_cache');rows=[];errors=[]
        write_json(run/'hardware.json',dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,base_sampler_disabled=True,model_training=False))
        for i in range(limit):
            expected=t['rows'][i];native=physical_snapshot(task)
            require(json_hash(native)==expected['native_before_sha256'],'Replay native-before mismatch at query'+str(i))
            images,state=unpack_observation(obs);components={k:array_hash(v) for k,v in zip(POLICY_CAMERA_KEYS,images)};components['command_state']=array_hash(state)
            require(components==expected['observation_components'],'Replay RGB/command mismatch at query'+str(i))
            rng=rng_fingerprint(rng_capture());c=base._model.paligemma_with_expert.paligemma.language_model.config;att=c._attn_implementation
            try:joint,meta=torch.compiler.disable(encode)(base,images,state,episode['instruction'])
            finally:c._attn_implementation=att
            require(rng==rng_fingerprint(rng_capture()),'Replay prefix encoding changed RNG')
            old_joint,old_state=previous if previous is not None else (np.zeros_like(joint),np.zeros_like(state))
            x=pack_observable(joint[None],state[None],old_joint[None],old_state[None],np.asarray([previous is not None]),projection,robots)
            require({k:array_hash(v) for k,v in x.items()}==expected['structured_input_sha256'],'Exact original9fields differ at query'+str(i))
            recorded=t['arrays']['base'][i]
            with torch.inference_mode():pred=model({k:torch.from_numpy(v.copy()) for k,v in x.items()},torch.from_numpy(recorded[None]),True)['delta'][0].numpy()
            error=float(np.abs(pred-t['arrays']['raw_delta'][i]).max());require(error<2e-5,'Original candidate mismatch')
            actions=recorded.copy();delta=t['arrays']['applied_delta'][i]
            if expected['window_active']:actions[:,ARM_INDICES]=recorded[:,ARM_INDICES]+delta[:,ARM_INDICES]
            require(array_hash(actions)==expected['action_chunk_sha256'],'Recorded actions reconstructed incorrectly')
            obs,executed=execute_chunk(task,actions);after=physical_snapshot(task)
            require(executed==expected['executed'] and json_hash(after)==expected['native_after_sha256'],'Replay native-after mismatch at query'+str(i))
            cache.append(i,{**x,'base':recorded[None].copy()},t['arrays']['raw_delta'][i],delta,actions,executed,
                {'scope':'Offline reconstructed confirmation state,nottraining', 'source_query':i,
                 'native_before_sha256':expected['native_before_sha256'],'native_after_sha256':expected['native_after_sha256'],
                 'source_episode_success':episode['success'],'episode_outcome_not_query_benefit':True})
            previous=joint.copy(),state.copy();errors.append(error);rows.append(dict(query=i,executed=executed,native_before_exact=True,native_after_exact=True,RGB_command_exact=True,nine_fields_exact=True,candidate_reload_error=error))
            replace_json(run/'progress.json',dict(phase='exact_recorded_replay',queries=i+1,total=limit,executed=task.take_action_cnt))
            print('REPLAY_QUERY',i,'native_count',task.take_action_cnt,'all_hashes_exact','candidate_error',error,flush=True)
        complete=limit==len(t['rows']);cache.finish(complete=complete,success=episode['success'] if complete else None,role='diagnostic_confirmation')
        write_json(run/'result.json',dict(status='feature_recovery_passed',source=str(args.source),seed=args.seed,condition=args.condition,
            queries=len(rows),native_actions=sum(r['executed'] for r in rows),whole_source_trajectory_replayed=complete,
            existing_source_episode_success=episode['success'],new_policy_evaluation=False,VLA_sampler_calls=0,
            all_native_RGB_9field_hashes_exact=True,CPU_candidate_max_error=max(errors),checks=rows,training_approved=False,
            raw_RGB_or_video_saved=False,elapsed_s=time.monotonic()-start,source_sha256={str(run/'manifest.json'):digest(run/'manifest.json'),str(run/'observable_cache/queries.npz'):digest(run/'observable_cache/queries.npz')}))
        replace_json(run/'progress.json',dict(phase='finished',queries=len(rows),passed=True));print('REPLAY_FEATURE_RECOVERY_PASS',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),new_policy_evaluation=False,VLA_sampler_calls=0));raise
    finally:
        if setup:task.close_env()
        write_json(run/'source_runtime_audit.json',storage.verify_source())

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--condition',choices=('Z_zero','A_causal_arm'),required=True);p.add_argument('--max-queries',type=int,default=16);p.add_argument('--gpu',default='0');a=p.parse_args()
    a.source=output_path(a.source);require(a.gpu.isdigit() and 1<=a.max_queries<=16,'Bounded replay')
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;os.environ['JAX_PLATFORMS']='cpu'
    from .native_entry_storage import NativeEntryStorage
    from .evaluate_structured_low_storage import _terminate
    storage=NativeEntryStorage();atexit.register(storage.cleanup);signal.signal(signal.SIGTERM,_terminate)
    def timeout(*args):raise TimeoutError('Singletrajectory1200second replay limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
    try:execute(a,storage)
    finally:signal.alarm(0)
if __name__=='__main__':main()

