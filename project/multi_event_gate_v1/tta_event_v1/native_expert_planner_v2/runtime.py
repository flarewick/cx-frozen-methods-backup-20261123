"""Actual paired initialization and original expert cached-planner replay, offline only."""
from pathlib import Path
import argparse,importlib,os,time,types
import numpy as np
from native_expert_planner_v2.paths import read,write,sha
from native_expert_planner_v2.observe import observe,unrecorded
from native_expert_planner_v2.recording import Sink

def diagnostic(task,names):
 from cx_tta.recording import measured_state
 from cx_tta.stacking_recording import actor_bindings
 values={}
 for k,obj in actor_bindings(task):
  if k in names:
   pose=obj.get_pose();values[k]=np.asarray(list(pose.p)+list(pose.q),np.float32)
 assert set(values)==set(names)
 arms,grip=measured_state(task.robot)
 return dict(arm_qpos=arms,gripper_qpos=grip,**{'object_'+k:values[k]for k in names})

def execute(spec,root,manifest,prepared,barrier):
 import torch
 from cx_tta_feedback_confirm5.lanes import Coordinator,Lane,gripper_only
 from cx_tta_feedback_confirm5.throughput import discard_observation
 from cx_tta.collect_smoke import make_config
 from cx_tta.evaluate_b2 import seed_all
 from native_physics import load_native
 from collection_native_step import load as load_step
 from cx_tta import recovery_contract as recovery
 from cx_tta.stacking_recording import install_recording
 from envs.utils.create_actor import UnStableError
 assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==manifest['gpu_uuid']
 torch.set_num_threads(2);c=Coordinator(cuda=True,native=load_native(),native_step=load_step());c.rng.install()
 import envs.robot.robot as robot
 old_planner=robot.CuroboPlanner;robot.CuroboPlanner=gripper_only(old_planner)
 saved={name:getattr(recovery,name)for name in('rng_capture','rng_restore')}
 for name,fn in saved.items():setattr(recovery,name,c.routed(fn))
 tasks={};raw_observers={};lanes={};warmed=set();rows=[];prepared['GPU_keepalive']=dict(c=c,tasks=tasks,lanes=lanes)
 try:
  for item in spec['scenes']:
   if item['task']not in lanes:lanes[item['task']]=Lane(item['task'])
   lane=lanes[item['task']];cfg=make_config(prepared['runtime'],argparse.Namespace(task=item['task'],task_config='demo_randomized'),prepared['session'])
   cfg.update(eval_video_log=False,eval_video_save_dir=None,render_freq=0)
   assert cfg['save_freq']==15
   with c.rng.bind(lane):
    if item['task']not in tasks:
     cls=getattr(importlib.import_module('envs.'+item['task']),item['task'])
     task=c.call(lane,lambda:c.prepare_task(cls(),lane));raw_observers[item['task']]=task.get_obs;install_recording(task,effect_schema=True);tasks[item['task']]=task
    task=tasks[item['task']]
    if not rows:
     write(barrier/'cuda_ready.json',dict(inner_pid=os.getpid(),gpu_uuid=manifest['gpu_uuid']))
     began=time.monotonic()
     while not(barrier/'owned_pid_permit.json').exists():
      if time.monotonic()-began>45:raise TimeoutError('No verified planner replay GPU owner permit')
      time.sleep(.03)
    began=time.monotonic();folder=root/('case%03d'%len(rows));folder.mkdir();mode_rows=[]
    with np.load(item['reference_file'],allow_pickle=False)as z:reference={k:z[k].copy()for k in z.files}
    paths={arm:[dict(status=v['status'],position=reference[v['position_key']].copy(),velocity=reference[v['velocity_key']].copy())for v in parts]for arm,parts in item['segments'].items()}
    def setup(mode):
     task._cx_planner_trace_active=False;c.call(lane,seed_all,item['seed'])
     opts=dict(cfg,eval_mode=mode,need_plan=mode,save_data=not mode,save_path=str(folder/'original_picture_sink'))
     if mode:opts['is_test']=True
     task.setup_demo(now_ep_num=item['episode_index'],seed=item['seed'],**opts)
     assert task.FRAME_IDX==0 and task.take_action_cnt==0
     if not mode:task.set_path_lst(dict(need_plan=False,**paths))
     task.set_instruction(instruction=item['instruction'])
    def initial(mode):
     # Same extra read as prior policy smoke, wrapped passively; no planner or policy action.
     discard_observation(task)
     with unrecorded(task,raw_observers[item['task']]):value=c.call(lane,observe,task)
     extra=diagnostic(task,item['object_names'])
     assert task._cx_actor_registry is None
     file=folder/('initial_eval_'+str(mode).lower()+'.npz')
     with file.open('xb')as f:np.savez_compressed(f,**{'cam'+str(i):v for i,v in enumerate(value['rgb'])},command_state=value['command_state'],**extra)
     return dict(eval_mode=mode,file=str(file),sha256=sha(file),native_setup_ticks=int(task._cx_tick),native_action_count=0,physics_sha256=value['physical_sha256'],rng_sha256=value['rng_sha256'],texture_info=task.info['texture_info'],offline_privileged_identity_only=True)
    try:
     if item['task']not in warmed:
      setup(True);c.call(lane,observe,task);task.close_env(clear_cache=False);warmed.add(item['task'])
     setup(True);mode_rows.append(initial(True));task.close_env(clear_cache=False)
     setup(False);mode_rows.append(initial(False));assert not task.need_plan and not task.eval_mode
     sink=Sink(task,item['object_names']);original_picture=task._take_picture;fn=original_picture.__func__
     private=types.FunctionType(fn.__code__,dict(fn.__globals__,save_pkl=sink),fn.__name__,fn.__defaults__,fn.__closure__);private.__kwdefaults__=fn.__kwdefaults__
     task._take_picture=types.MethodType(private,task)
     task._cx_actual_trace=[];task._cx_planner_trace_active=True;first_tick=int(task._cx_tick)
     try:info=task.play_once()
     finally:task._cx_planner_trace_active=False;task._take_picture=original_picture
     success=bool(task.plan_success and task.check_success());arrays=sink.arrays();trace=task._cx_actual_trace
     assert trace and all(trace[i]['tick_before']==first_tick+i for i in range(len(trace)))
     arrays.update({'actual_'+k:np.stack([r[k]for r in trace])for k in trace[0]})
     arrays.update(dense_ticks=np.asarray([v[0]for v in task._cx_dense],np.int64),dense_dt=np.asarray([v[1]for v in task._cx_dense],np.float64),dense_commands=np.stack([v[2]for v in task._cx_dense]))
     file=folder/'original_planner_execution.npz'
     with file.open('xb')as f:np.savez_compressed(f,**arrays)
     row=dict(case_id=item['case_id'],index=len(rows),available=True,initializations=mode_rows,file=str(file),sha256=sha(file),
      native_planner_success=bool(task.plan_success),native_task_success=success,frames=len(sink.frames),original_picture_calls=int(task.FRAME_IDX),
      first_motion_tick=first_tick,last_physics_tick=int(task._cx_tick),actual_recorded_motion_steps=len(trace),
      left_paths_used=int(task.left_cnt),right_paths_used=int(task.right_cnt),left_paths_available=len(paths['left_joint_path']),right_paths_available=len(paths['right_joint_path']),
      policy_take_action_calls=int(task.take_action_cnt),seconds=time.monotonic()-began,
      old_velocity_inferred=False,no_VLA_or_semantic_model_loaded=True,native32_policy_cache_ready=False)
    except UnStableError as exc:
     row=dict(case_id=item['case_id'],index=len(rows),available=False,reason='native_unstable_initialization_retained',initializations=mode_rows,error=str(exc),seconds=time.monotonic()-began)
    write(folder/'capture.json',row);rows.append(row);task.close_env(clear_cache=False)
    print('ORIGINAL_PLANNER_CASE',row['case_id'],row['available'],row.get('native_task_success'),row['seconds'],flush=True)
  write(root/'GPU_frames_complete.json',dict(complete=True,rows=rows,protocol_identity=spec['identity'],requested=4,
   no_VLA_or_semantic_model_loaded=True,native_training_cache_ready=False,online_updates=0,TTA_utility_proven=False))
 finally:
  for name,fn in saved.items():setattr(recovery,name,fn)
  robot.CuroboPlanner=old_planner;c.rng.uninstall()
