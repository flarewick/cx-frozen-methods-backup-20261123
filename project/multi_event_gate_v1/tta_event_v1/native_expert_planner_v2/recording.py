"""Original expert picture/clock calls, with a numeric sink and actual drive velocity observation."""
import ast
from pathlib import Path
import numpy as np
from actuation_telemetry import capture

def instrument(source):
 assert '_cx_record_step'not in source and 'self.scene.step()'in source
 out='from native_expert_planner_v2.recording import counted_step as _cx_record_step\n'+source.replace('self.scene.step()','_cx_record_step(self)')
 ast.parse(out);return out

def counted_step(task):
 from cx_tta.recording import counted_step as original
 record=bool(getattr(task,'_cx_planner_trace_active',False))
 if record:
  tick=int(task._cx_tick);row=capture(task.robot);dt=float(task.scene.get_timestep())
 value=original(task)
 if record:
  assert int(task._cx_tick)==tick+1
  task._cx_actual_trace.append(dict(tick_before=tick,dt_s=dt,**row))
 return value

class Sink:
 def __init__(self,task,names):self.task=task;self.names=names;self.frames=[];self.first_rgb=None;self.first_camera={}
 def __call__(self,file,obs):
  # Called at the original save_pkl site, after the original get_obs, with no new observation or draw.
  assert Path(file).name==str(len(self.frames))+'.pkl'
  m=obs['cx_measurement'];p=obs['cx_privileged'];assert all(k in p for k in self.names)
  row=dict(tick=int(m['physics_tick']),dt=float(m['physics_dt_s']),command=np.asarray(obs['joint_action']['vector']).copy(),arm_qpos=np.asarray(m['arm_qpos']).copy(),gripper_qpos=np.asarray(m['gripper_joint_qpos']).copy(),objects={k:np.asarray(p[k]).copy()for k in self.names})
  self.frames.append(row)
  if self.first_rgb is None:
   self.first_rgb={k:np.asarray(obs['observation'][k]['rgb']).copy()for k in('head_camera','right_camera','left_camera')}
   self.first_camera={k:np.asarray(obs['observation'][k]['extrinsic_cv']).copy()for k in self.first_rgb}
 def arrays(self):
  assert self.frames
  f=self.frames;a={name:np.asarray([r[key]for r in f])for name,key in(('frame_ticks','tick'),('frame_dt','dt'),('frame_commands','command'),('arm_qpos','arm_qpos'),('gripper_qpos','gripper_qpos'))}
  a.update({'object_'+k:np.stack([r['objects'][k]for r in f])for k in self.names})
  a.update({'initial_'+k+'_rgb':v for k,v in self.first_rgb.items()});a.update({'initial_'+k+'_extrinsic':v for k,v in self.first_camera.items()})
  return a
