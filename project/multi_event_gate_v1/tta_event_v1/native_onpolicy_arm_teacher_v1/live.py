"""Offline current-geometry recovery while independent full lanes use the GPU."""
from pathlib import Path
import argparse,importlib,weakref,gc
import numpy as np
from native_policy_execution_v1.common import write
from native_onpolicy_recovery_v3.common import read,sha,mapped
from native_onpolicy_recovery_v3.state import arrays,require_same
ARM=np.array([0,1,2,3,4,5,7,8,9,10,11,12])
def recover(spec,root,case,c,prepared,bridge,number,*,lane,p):
 from cx_tta_feedback_confirm5.throughput import replay_prefix
 from cx_tta.collect_smoke import make_config
 from cx_tta.evaluate_b2 import seed_all
 from cx_tta import recovery_contract as recovery
 from cx_tta.expert_screening import run_expert_plan,ExpertPlanningFailure,planning_failure_record
 from native_initial_frame_v3.observe import physical
 from native_expert_smoke_v2.observe import observe
 from native_arm_phase_feedback_v1.capture import capture
 from .record import record
 import torch
 import envs.robot.robot as robot
 schedule=prepared['live_teacher_schedule'];assert c.rng.current()is lane
 folder=root/'recoveries'/case['case_id'];folder.mkdir(parents=True)
 row=dict(case_id=case['case_id'],task=case['task'],query=p['query'],application_mode=p['choice']['application_mode'],packet=p['choice']['packet'],teacher_kind='live_current_geometry_open_replan',label_available=False,provisional_until_entire_parent_batch_qualifies=True,optimizer_steps=0,no_actor_or_online_teacher=True,online_GRIP_remains_frozen=True)
 cfg=make_config(prepared['runtime'],argparse.Namespace(task=case['task'],task_config='demo_randomized'),prepared['session']);cfg.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
 with schedule.serial_screen_lock,c.rng.bind(lane):
  phase=getattr(lane,'phase',None);lane.phase='expert_screen'
  def create():
   from shared6_task_bookkeeping import install_task
   value=getattr(importlib.import_module('envs.'+case['task']),case['task'])();install_task(value);return c.prepare_task(value,lane)
  task=c.call(lane,create);original_setup=task.setup_demo
  def setup(**kwargs):
   def owner_setup():
    saved=robot.CuroboPlanner;assert saved is schedule.replay
    robot.CuroboPlanner=schedule.pooled
    try:return original_setup(**kwargs)
    finally:robot.CuroboPlanner=saved
   owner_setup._cx_stage='setup';owner_setup._cx_priority=5;return c.call(lane,owner_setup)
  task.setup_demo=setup;raw=lambda:c.call(lane,observe,task);seed=c.routed(seed_all);primary_error=False
  def check(value,label):
   diagnostic=c.call(lane,lambda:dict(physical_sha256=physical(task),rng_sha256=recovery.rng_fingerprint(recovery.rng_capture())))
   proof=require_same(value,diagnostic,p);f=folder/(label+'_state.npz')
   with f.open('xb')as stream:np.savez_compressed(stream,**{'cam'+str(i):v for i,v in enumerate(value['rgb'])},command_state=value['command_state'],native_action_count=np.asarray(value['native_action_count']))
   write(folder/(label+'_same_state.json'),dict(**proof,**diagnostic,state_file=str(f),state_sha256=sha(f)))
  def prefix(label):
   current=replay_prefix(task,case,[],cfg,raw,seed)
   for i,part in enumerate(p['prefix']):
    bridge.begin_case(case['case_id']+'/'+label+str(i),number)
    try:actual,meta,_=capture(task,part['actions'],reader=raw,initial=current,kind=part['kind'],collect_dense=bool(part['metadata']['dense_frame_count']),on_arrival=None,check_cancel=c.check_cancel)
    finally:audited=bridge.finish_case()
    assert actual.keys()==part['actual_trace'].keys()and all(np.array_equal(actual[k],part['actual_trace'][k],equal_nan=True)for k in actual)
    assert all(meta[k]==part['metadata'][k]for k in('receipts','reason','committed_ticks','readouts'))and audited['native_calls']==meta['committed_ticks'];current=raw()
   check(current,label);return current
  try:
   # Original constructor warmup consumes no hidden q0 randomness: reseed/setup
   # and the entire actual policy prefix follow it, with strict state equality.
   try:seed(case['seed']);setup(now_ep_num=case['episode_index'],seed=case['seed'],**dict(cfg,need_plan=True,save_data=False,is_test=True))
   finally:task.close_env(clear_cache=False)
   prefix('before_plan')
   def operation():
    from envs.utils import ArmTag
    task.move(task.open_gripper(ArmTag('left')),task.open_gripper(ArmTag('right')))
    return run_expert_plan(task)
   bridge.begin_case(case['case_id']+'/live_raw_expert',number)
   try:_,receipt,values=record(task,operation,folder)
   finally:audited=bridge.finish_case();write(folder/'expert_plan_readout_bridge.json',audited)
   assert audited['native_calls']==receipt['physics_steps']
   success=c.call(lane,lambda:bool(task.plan_success and task.check_success()));write(folder/'expert_plan_receipt.json',dict(**receipt,original_plan_and_task_success=success,teacher_offline_only=True))
   if not success:row['reason']='live_current_geometry_expert_failed_retained'
   else:
    assert values is not None and receipt['commands']>0;commands=values['absolute_source_commands'];task.close_env(clear_cache=False);current=prefix('before_replay');ticks=0;returned=0;first=None
    bridge.begin_case(case['case_id']+'/live_native_teacher',number)
    try:
     for i,start in enumerate(range(0,len(commands),32)):
      source=commands[start:start+32].copy();trace,meta,_=capture(task,source,reader=raw,initial=current,kind=None,collect_dense=False,on_arrival=None,check_cancel=c.check_cancel)
      f=folder/f'teacher_execution{i:03d}.npz'
      with f.open('xb')as h:np.savez_compressed(h,**trace)
      write(folder/f'teacher_execution{i:03d}.json',dict(**meta,actual_trace_file=str(f),actual_trace_sha256=sha(f)))
      valid=np.asarray([r['offset']for r in meta['receipts']if r['status']=='returned'],np.int64);ticks+=meta['committed_ticks'];returned+=len(valid)
      if i==0:first=(source,valid)
      if meta['reason']!='source_chunk_returned':break
      current=raw()
    finally:audited=bridge.finish_case();write(folder/'teacher_readout_bridge.json',audited)
    assert audited['native_calls']==ticks;source,valid=first;native=c.call(lane,lambda:bool(task.eval_success));original=c.call(lane,lambda:bool(task.check_success()))
    row.update(native_teacher_replay_success=native,original_task_success=original,returned_commands=returned,first_action_returned=len(valid))
    if native and original and len(valid):
     target=folder/'teacher_target.npz';base=p['once_policy']['base']
     with target.open('xb')as h:np.savez_compressed(h,valid_offsets=valid,teacher_ARM=source[valid][:,ARM],once_base_ARM=base[valid][:,ARM],direction_target_ARM=np.clip(source[valid][:,ARM]-base[valid][:,ARM],-.1,.1).astype(np.float32))
     row.update(label_available=True,target_file=str(target),target_sha256=sha(target),reason='same_state_live_current_geometry_native_recovery_success',teacher_vs_base_first_GRIP_difference_diagnostic_only=float(np.max(np.abs(source[valid][:,[6,13]]-base[valid][:,[6,13]]))))
    else:row['reason']='live_native_teacher_replay_failed_retained'
  except ExpertPlanningFailure as error:row.update(reason='live_expert_planning_failure_retained',planning_failure=planning_failure_record(error))
  except BaseException:
   import traceback
   primary_error=True;write(folder/'primary_failure.json',dict(traceback=traceback.format_exc(),no_dataset_qualification=True));raise
  finally:
   task.close_env(clear_cache=False);task_ref=weakref.ref(task);scene_ref=weakref.ref(task.scene);task=None;raw=None;original_setup=None;lane.phase=phase
   def release():
    with c.scene_gate.write():gc.collect();torch.cuda.empty_cache()
    if not primary_error:
     assert task_ref()is None and scene_ref()is None
     from native_expert_case_release_v1.release import retire_live_group
     proof=retire_live_group(schedule,lane,c);write(folder/'closed_live_planner_proxy_release.json',proof)
   c.call(lane,release)
 write(folder/'result.json',row);print('PROVISIONAL_LIVE_NATIVE_RECOVERY',case['case_id'],row['label_available'],row['reason'],flush=True);return row

def audit(spec,root,case,row,p):
 folder=root/'recoveries'/case['case_id']
 proof=read(folder/'closed_live_planner_proxy_release.json');assert proof['live_closed_proxy_group_released']and proof['original_shared_planners_unchanged']
 for label in ['before_plan']+(['before_replay']if 'native_teacher_replay_success'in row else[]):
  proof=read(folder/(label+'_same_state.json'));a=arrays(proof['state_file']);assert sha(mapped(proof['state_file']))==proof['state_sha256'];require_same(dict(rgb=[a[f'cam{i}']for i in range(3)],command_state=a['command_state'],native_action_count=int(a['native_action_count'])),proof,p)
 if 'native_teacher_replay_success'in row:
  plan=read(folder/'expert_plan_receipt.json');assert plan['original_plan_and_task_success']and plan['sampling_stride']==15 and sha(mapped(plan['file']))==plan['sha256'];v=arrays(plan['file']);ix=np.arange(14,len(v['actual_dt_s']),15,dtype=np.int64);assert np.array_equal(v['source_dense_indices'],ix)
  cmd=np.column_stack((v['actual_arm_position_target'][ix,:6],v['actual_gripper_logical_command'][ix,0],v['actual_arm_position_target'][ix,6:],v['actual_gripper_logical_command'][ix,1])).astype(np.float32);assert np.array_equal(cmd,v['absolute_source_commands'])
  meta=read(folder/'teacher_execution000.json');trace=arrays(meta['actual_trace_file']);assert sha(mapped(meta['actual_trace_file']))==meta['actual_trace_sha256']and np.array_equal(trace['source_commands'],cmd[:32]);valid=np.asarray([r['offset']for r in meta['receipts']if r['status']=='returned'],np.int64)
  assert row['label_available']==bool(row['native_teacher_replay_success']and row['original_task_success']and len(valid))
  if row['label_available']:
   a=arrays(row['target_file']);assert sha(mapped(row['target_file']))==row['target_sha256']and np.array_equal(a['valid_offsets'],valid)and np.array_equal(a['teacher_ARM'],cmd[valid][:,ARM])and np.array_equal(a['once_base_ARM'],p['once_policy']['base'][valid][:,ARM])and np.array_equal(a['direction_target_ARM'],np.clip(a['teacher_ARM']-a['once_base_ARM'],-.1,.1).astype(np.float32))
 if not row['label_available']:assert not(folder/'teacher_target.npz').exists()
