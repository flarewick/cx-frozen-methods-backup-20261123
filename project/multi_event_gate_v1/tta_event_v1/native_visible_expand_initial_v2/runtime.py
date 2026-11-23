"""Actual raw q0 only; no model, actions, expert replay or independent scene."""
import argparse,importlib,os,time
import numpy as np
from .paths import read,write,sha,verify
from native_expert_smoke_v2.observe import observe

def execute(spec,root,manifest,prepared,barrier):
 import torch
 from cx_tta_feedback_confirm5.lanes import Coordinator,Lane,gripper_only
 from cx_tta_feedback_confirm5.throughput import replay_prefix
 from cx_tta.collect_smoke import make_config
 from cx_tta.evaluate_b2 import seed_all
 from native_physics import load_native
 from collection_native_step import load as load_step
 from cx_tta import recovery_contract as recovery
 from envs.utils.create_actor import UnStableError
 assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==manifest['gpu_uuid']
 torch.set_num_threads(2);c=Coordinator(cuda=True,native=load_native(),native_step=load_step());c.rng.install()
 import envs.robot.robot as robot
 original_planner=robot.CuroboPlanner;robot.CuroboPlanner=gripper_only(original_planner)
 saved={name:getattr(recovery,name)for name in('rng_capture','rng_restore')}
 for name,fn in saved.items():setattr(recovery,name,c.routed(fn))
 tasks=[];prepared['GPU_keepalive']=dict(coordinator=c,tasks=tasks);rows=[];warmed=set()
 try:
  for i,item in enumerate(spec['scenes']):
   cls=getattr(importlib.import_module('envs.'+item['task']),item['task']);lane=Lane(str(i))
   cfg=make_config(prepared['runtime'],argparse.Namespace(task=item['task'],task_config='demo_randomized'),prepared['session'])
   cfg.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
   with c.rng.bind(lane):
    task=c.call(lane,lambda:c.prepare_task(cls(),lane));tasks.append(task)
    if i==0:
     write(barrier/'cuda_ready.json',dict(inner_pid=os.getpid(),gpu_uuid=manifest['gpu_uuid']))
     began=time.monotonic()
     while not(barrier/'owned_pid_permit.json').exists():
      if time.monotonic()-began>45:raise TimeoutError('No verified initial-render GPU owner permit')
      time.sleep(.03)
     assert read(barrier/'owned_pid_permit.json')['gpu_uuid']==manifest['gpu_uuid']
    raw=lambda:c.call(lane,observe,task);routed_seed=c.routed(seed_all);began=time.monotonic()
    warmup_failure=None
    if item['task']not in warmed:
     try:replay_prefix(task,item,[],cfg,raw,routed_seed)
     except UnStableError as exc:warmup_failure=dict(type=type(exc).__name__,reason=str(exc),phase='scheduled_warmup',target_case_id=item['case_id'],no_policy_actions=True)
     finally:task.close_env(clear_cache=False)
     warmed.add(item['task'])
    try:
     actual=replay_prefix(task,item,[],cfg,raw,routed_seed)
     assert task.take_action_cnt==actual['native_action_count']==0 and int(task.step_lim)>0
     folder=root/('case%03d'%i);folder.mkdir();file=folder/'initial.npz'
     with file.open('xb')as f:np.savez_compressed(f,**{'cam'+str(j):im for j,im in enumerate(actual['rgb'])},command_state=actual['command_state'])
     row=dict(case_id=item['case_id'],frame_available=True,warmup_failure=warmup_failure,instruction=item['instruction'],file=str(file),sha256=sha(file),
      physical_sha256=actual['physical_sha256'],rng_sha256=actual['rng_sha256'],native_step_limit=int(task.step_lim),
      actual_native_action_count=0,query=0,tick=0,initialization_eval_mode=True,
      seconds_including_task_warmup=time.monotonic()-began,old_expert_masks_NOT_imported=True,new_independent_scene=False)
     write(folder/'capture.json',row);rows.append(row)
     print('DEVELOPER_ACTUAL_EVAL_Q0',item['case_id'],row['seconds_including_task_warmup'],flush=True)
    except UnStableError as exc:
     folder=root/('case%03d'%i);folder.mkdir(exist_ok=True)
     row=dict(case_id=item['case_id'],instruction=item['instruction'],frame_available=False,file=None,sha256=None,warmup_failure=warmup_failure,initialization_failure=dict(type=type(exc).__name__,reason=str(exc),phase='scheduled_actual_initialization',target_case_id=item['case_id']),actual_native_action_count=0,query=0,tick=0,initialization_eval_mode=True,new_independent_scene=False,no_substitution=True)
     write(folder/'capture_failure.json',row);rows.append(row)
     print('FIXED_DEVELOPER_Q0_NATIVE_INITIALIZATION_MISSING',item['case_id'],str(exc),flush=True)
    finally:task.close_env(clear_cache=False)
  verify(spec);write(root/'GPU_frames_complete.json',dict(complete=True,rows=rows,protocol_identity=spec['identity'],new_independent_scenes=0,online_updates=0,policy_actions=0))
 finally:
  for name,fn in saved.items():setattr(recovery,name,fn)
  robot.CuroboPlanner=original_planner;c.rng.uninstall()
