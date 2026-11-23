"""Original raw scene-step teacher drive capture at an already verified state."""
import numpy as np
from native_policy_execution_v1.common import write,sha
from native_object_expert_train_v3.expert_reference import sample_commands

def record(task,operation,folder):
 from actuation_telemetry import ACTIVE as TRACE,Recorder,capture
 from cx_tta_clock.clock import ACTIVE as CLOCK
 if CLOCK.get()is not None or TRACE.get()is not None:raise ValueError('Teacher plan must not borrow a live policy clock')
 original=task.scene.step;trace=Recorder(limit=100000)
 def observed():
  row=capture(task.robot);dt=float(task.scene.get_timestep());value=original();trace.append(row,dt);return value
 task.scene.step=observed
 try:result=operation()
 finally:task.scene.step=original
 assert not trace.failed
 arrays=None
 if len(trace.rows)>=15:
  arrays=sample_commands(trace.rows,trace.dt);arrays.update({'actual_'+k:np.stack([r[k]for r in trace.rows])for k in trace.rows[0]});arrays['actual_dt_s']=np.asarray(trace.dt,np.float64)
  file=folder/'expert_plan_commands.npz'
  with file.open('xb')as stream:np.savez_compressed(stream,**arrays)
  receipt=dict(file=str(file),sha256=sha(file),physics_steps=len(trace.rows),commands=len(arrays['absolute_source_commands']),sampling_stride=15,scene_step_restored=True)
 else:receipt=dict(file=None,sha256=None,physics_steps=len(trace.rows),commands=0,sampling_stride=15,scene_step_restored=True)
 return result,receipt,arrays
