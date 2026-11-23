"""Four fixed FIT native command-reference episodes; no actor, feedback learner or new scenes."""
from pathlib import Path
import argparse,importlib,os,time
import numpy as np
from native_expert_smoke_v2.paths import read,write,sha
from native_expert_smoke_v2.observe import observe
from native_expert_smoke_v2.capture import capture

def execute(spec,root,manifest,prepared,barrier):
 import torch
 from cx_tta_feedback_confirm5.lanes import Coordinator,Lane,gripper_only
 from native_expert_smoke_v2.setup import replay_prefix
 from cx_tta.collect_smoke import make_config
 from cx_tta.evaluate_b2 import seed_all
 from native_physics import load_native
 from collection_native_step import load as load_step
 from cx_tta import recovery_contract as recovery
 assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==manifest['gpu_uuid']
 torch.set_num_threads(2);c=Coordinator(cuda=True,native=load_native(),native_step=load_step());c.rng.install()
 import envs.robot.robot as robot
 from envs.utils.create_actor import UnStableError
 old_planner=robot.CuroboPlanner;robot.CuroboPlanner=gripper_only(old_planner)
 saved={name:getattr(recovery,name)for name in('rng_capture','rng_restore')}
 for name,fn in saved.items():setattr(recovery,name,c.routed(fn))
 tasks={};lanes={};warmed=set();rows=[];prepared['GPU_keepalive']=dict(c=c,tasks=tasks,lanes=lanes)
 try:
  for item in spec['scenes']:
   if item['task']not in lanes:lanes[item['task']]=Lane(item['task'])
   lane=lanes[item['task']];cfg=make_config(prepared['runtime'],argparse.Namespace(task=item['task'],task_config='demo_randomized'),prepared['session'])
   cfg.update(eval_mode=False,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
   with c.rng.bind(lane):
    if item['task']not in tasks:
     cls=getattr(importlib.import_module('envs.'+item['task']),item['task'])
     tasks[item['task']]=c.call(lane,lambda:c.prepare_task(cls(),lane))
    task=tasks[item['task']]
    if not rows:
     write(barrier/'cuda_ready.json',dict(inner_pid=os.getpid(),gpu_uuid=manifest['gpu_uuid']))
     began=time.monotonic()
     while not(barrier/'owned_pid_permit.json').exists():
      if time.monotonic()-began>45:raise TimeoutError('No fresh verified expert replay GPU owner permit')
      time.sleep(.03)
    began=time.monotonic();folder=root/('case%03d'%len(rows));folder.mkdir();chunks=[];observations=[]
    def save_obs(value,q,tick):
     file=folder/('query%03d.npz'%q)
     with file.open('xb')as f:np.savez_compressed(f,**{'cam'+str(i):v for i,v in enumerate(value['rgb'])},command_state=value['command_state'])
     row={k:v for k,v in value.items()if k not in('rgb','command_state')}
     row.update(file=str(file),sha256=sha(file),query=q,global_tick=tick);observations.append(row)
    try:
     raw=lambda:c.call(lane,observe,task);routed_seed=c.routed(seed_all)
     if item['task']not in warmed:
      replay_prefix(task,item,[],cfg,raw,routed_seed);task.close_env(clear_cache=False);warmed.add(item['task'])
     current=replay_prefix(task,item,[],cfg,raw,routed_seed)
     from native_expert_planner_v2.runtime import diagnostic
     identity=diagnostic(task,item['object_names'])
     with np.load(item['matched_initial_file'],allow_pickle=False)as z:
      assert all(np.array_equal(current['rgb'][i],z['cam'+str(i)])for i in range(3)),'Expert scene q0 RGB differs from actual cached-planner initialization'
      assert np.array_equal(current['command_state'],z['command_state'])
      assert all(np.array_equal(v,z[k])for k,v in identity.items()),'Expert scene q0 object/robot state differs'
     save_obs(current,0,0)
     with np.load(item['file'],allow_pickle=False)as z:commands=z['absolute_source_commands'].copy()
     total_ticks=0
     for q,start in enumerate(range(0,len(commands),32)):
      if task.eval_success or int(task.take_action_cnt)>=int(task.step_lim):break
      a,receipt=capture(task,commands[start:start+32],max_ticks=spec['max_ticks_per_chunk'])
      file=folder/('chunk%03d.npz'%q)
      with file.open('xb')as f:np.savez_compressed(f,**a)
      receipt.update(query=q,source_start=start,source_count=len(a['source_commands']),global_start_tick=total_ticks,file=str(file),sha256=sha(file))
      chunks.append(receipt);total_ticks+=receipt['committed_ticks']
      current=raw();save_obs(current,q+1,total_ticks)
      print('EXPERT_NATIVE_CHUNK',item['case_id'],q,receipt['committed_ticks'],receipt['reason'],flush=True)
      if receipt['reason']!='source_chunk_returned':break
     row=dict(case_id=item['case_id'],index=len(rows),available=True,chunks=chunks,observations=observations,
      actual_native_action_count=int(task.take_action_cnt),native_step_limit=int(task.step_lim),native_success_diagnostic_only=bool(task.eval_success),
      source_commands=len(commands),committed_ticks=total_ticks,seconds=time.monotonic()-began,
      matched_initial_RGB_command_object_robot_exact=True,initialization_eval_mode=False,step_limit_assigned_only_after_setup=True,
      source_exhausted=bool(chunks and chunks[-1]['source_start']+len(chunks[-1]['receipts'])==len(commands)),
      no_VLA_or_semantic_model_loaded=True,no_feedback_or_training_labels=True)
    except UnStableError as exc:
     row=dict(case_id=item['case_id'],index=len(rows),available=False,reason='native_unstable_initialization_retained',error=str(exc),seconds=time.monotonic()-began)
    write(folder/'capture.json',row);rows.append(row);task.close_env(clear_cache=False)
    print('EXPERT_NATIVE_CASE',row['case_id'],row['available'],row['seconds'],flush=True)
  write(root/'GPU_frames_complete.json',dict(complete=True,rows=rows,protocol_identity=spec['identity'],requested=4,
   no_VLA_or_semantic_model_loaded=True,native_training_cache_ready=False,online_updates=0,TTA_utility_proven=False))
 finally:
  for name,fn in saved.items():setattr(recovery,name,fn)
  robot.CuroboPlanner=old_planner;c.rng.uninstall()
