"""Passive actual commanded-drive reference during original successful expert plan."""
from pathlib import Path
import numpy as np
from .common import write,sha

def sample_commands(rows,dt,*,stride=15):
 if stride!=15 or len(rows)!=len(dt)or not rows:raise ValueError('Original15-physics-step expert sampling required')
 if not np.isfinite(dt).all()or np.any(np.asarray(dt)<=0):raise ValueError('Actual positive finite physics durations required')
 indices=np.arange(stride-1,len(rows),stride,dtype=np.int64)
 if len(indices)==0:raise ValueError('Expert plan has no complete original sampling interval')
 commands=np.zeros((len(indices),14),np.float32)
 for j,i in enumerate(indices):
  row=rows[int(i)];a=np.asarray(row['arm_position_target']);g=np.asarray(row['gripper_logical_command'])
  if a.shape!=(12,)or g.shape!=(2,)or not np.isfinite(a).all()or not np.isfinite(g).all():raise ValueError('Exact original commanded positions/grips required')
  commands[j,:6]=a[:6];commands[j,6]=g[0];commands[j,7:13]=a[6:];commands[j,13]=g[1]
 return dict(absolute_source_commands=commands,source_dense_indices=indices,source_physics_ticks=indices+1,source_time_s=np.cumsum(np.asarray(dt,np.float64))[indices])

def record(task,operation,*,root,case_id):
 from actuation_telemetry import ACTIVE as TRACE,Recorder,capture
 from cx_tta_clock.clock import ACTIVE as CLOCK
 if CLOCK.get()is not None or TRACE.get()is not None:raise ValueError('Expert reference cannot borrow a live actor clock')
 # The official expert take_dense_action calls scene.step directly, bypassing
 # policy-only _cx_diag_step. Observe the actual scene step for this call only.
 original=task.scene.step;trace=Recorder(limit=100000)
 def observed_step():
  row=capture(task.robot);dt=float(task.scene.get_timestep());value=original();trace.append(row,dt);return value
 task.scene.step=observed_step
 try:info=operation()
 finally:task.scene.step=original
 if trace.failed:raise ValueError('Actual expert drive trace failed')
 receipt=dict(complete=True,case_id=case_id,commands=0,actual_expert_physics_steps=len(trace.rows),expert_plan_returned_normally=True,sampling_stride_physics_steps=15,source='Actual expert commanded drive targets; not sampled VLA actions',actual_native_replay_still_required=True,source_velocity_not_fabricated=True,policy_or_response_training_steps=0,scene_step_restored_after_record=True,expert_raw_scene_step_observer=True)
 if not trace.rows:
  receipt.update(file=None,sha256=None,no_physics_steps=True)
  return info,receipt
 # Partial intervals cannot be promoted to a full original15-step sample.
 if len(trace.rows)<15:
  receipt.update(file=None,sha256=None,no_complete_sampling_interval=True)
  return info,receipt
 arrays=sample_commands(trace.rows,trace.dt);arrays.update({'actual_'+k:np.stack([r[k]for r in trace.rows])for k in trace.rows[0]});arrays['actual_dt_s']=np.asarray(trace.dt,np.float64)
 folder=root/'expert_references';folder.mkdir(exist_ok=True);file=folder/(case_id+'.npz')
 with file.open('xb')as f:np.savez_compressed(f,**arrays)
 receipt.update(file=str(file),sha256=sha(file),commands=len(arrays['absolute_source_commands']))
 return info,receipt

def finalize(root,case_id,receipt,*,expert_plan_success,expert_task_success):
 if expert_plan_success and expert_task_success and not receipt['commands']:
  raise ValueError('Successful expert has no actual complete15-step reference: capture must be fixed, never skip it')
 value=dict(receipt,expert_plan_success=bool(expert_plan_success),expert_task_success=bool(expert_task_success),success_checked_only_by_original_screen=True)
 write(root/'expert_references'/(case_id+'.json'),value)
 return value
