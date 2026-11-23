"""Passive per-native-action clocks/physics, and signed executed arrays."""
import functools,gzip,json
from pathlib import Path
import numpy as np
from cx_tta.attribution_physics import physical_snapshot
from cx_tta.eval_diagnostics import state_snapshot,json_hash,array_hash
class NativeTap:
 def __init__(self,native,coordinator):self.native=native;self.c=coordinator
 def __getattr__(self,name):return getattr(self.native,name)
 def control(self,*args,**kwargs):
  result=self.native.control(*args,**kwargs)
  lane=self.c.rng.current();task=getattr(lane,'pilot_task',None)
  if task is not None and getattr(task,'_pilot_recording',False):task._pilot_ticks+=int(result[3])
  return result

def install_scene_clock(task):
 # Native C++ advances PhysxSystem directly; the official Python fallback
 # advances Scene.step instead. Count exactly those two disjoint paths.
 scene=task.scene;original_step=scene.step
 def step(*args,**kwargs):
  result=original_step(*args,**kwargs)
  if task._pilot_recording:task._pilot_ticks+=1
  return result
 scene.step=step

def contacts(task):
 rows=[]
 for item in task.scene.physx_system.get_contacts():
  bodies=[]
  for body in item.bodies:
   entity=body.entity;bodies.append(str(entity.name))
  points=[dict(position=np.asarray(p.position).tolist(),normal=np.asarray(p.normal).tolist(),impulse=np.asarray(p.impulse).tolist(),separation=float(p.separation))for p in item.points]
  points.sort(key=lambda p:json.dumps(p,sort_keys=True));rows.append(dict(bodies=bodies,points=points))
 rows.sort(key=lambda r:json.dumps(r,sort_keys=True));return rows

def success_parts(task):
 if type(task).__name__=='lift_pot':
  pose=task.pot.get_pose();left=np.asarray(task.robot.get_left_tcp_pose()[:3]);right=np.asarray(task.robot.get_right_tcp_pose()[:3]);lg=np.asarray(task.pot.get_contact_point(0)[:3]);rg=np.asarray(task.pot.get_contact_point(1)[:3])
  upright=float(task.check_success.__func__.__globals__['get_face_prod'](pose.q,[0,0,1],[0,0,1]));dl=float(np.linalg.norm(left-lg));dr=float(np.linalg.norm(right-rg))
  return dict(pot_z=float(pose.p[2]),left_contact_distance=dl,right_contact_distance=dr,upright=upright,height_ok=bool(pose.p[2]>.82),left_ok=dl<.03,right_ok=dr<.03,upright_ok=upright>.8)
 z=float(task.deskbin.get_pose().p[2]);zs=[float(s.get_pose().p[2])for s in task.sphere_lst]
 return dict(deskbin_z=z,height_ok=z>=1.,garbage_z=zs,garbage_in_target=[.13<=x<=.25 for x in zs])

def compact(task):return dict(state=state_snapshot(task),contacts=contacts(task),task_success_components=success_parts(task))

def install(task,lane,c):
 lane.pilot_task=task;task._pilot_recording=False;task._pilot_ticks=0;task._pilot_action_rows=[]
 previous=task.take_action
 @functools.wraps(previous)
 def action(command,*args,**kwargs):
  if not task._pilot_recording:return previous(command,*args,**kwargs)
  start=int(task.take_action_cnt);ticks=task._pilot_ticks;dt=float(task.scene.get_timestep())
  with c.scene_gate.read():before=compact(task)
  value=previous(command,*args,**kwargs)
  if int(task.take_action_cnt)!=start:
   with c.scene_gate.read():after=compact(task)
   task._pilot_action_rows.append(dict(action_index=start,command=np.asarray(command).tolist(),tick_before=ticks,tick_after=task._pilot_ticks,dt_s=dt,time_before_s=ticks*dt,time_after_s=task._pilot_ticks*dt,offline_only=dict(before=before,after=after)))
  return value
 task.take_action=action

def chunk_recorder(original,session):
 session=Path(session)
 def step(task,wrapper,observation,instruction,*args,**kwargs):
  if not task._pilot_recording:raise RuntimeError('Formal trace not initialized')
  q=wrapper.query_count;task._pilot_action_rows=[]
  before=physical_snapshot(task)
  result=original(task,wrapper,observation,instruction,*args,**kwargs)
  after=physical_snapshot(task)
  if not before['usable']or not after['usable']:raise RuntimeError('Incomplete passive physical snapshot')
  count=result[0];numeric=wrapper.last_numeric
  actions=np.asarray(numeric['actions']);base=np.asarray(numeric['base']);delta=np.asarray(numeric['applied_delta'])
  if len(task._pilot_action_rows)!=count or not np.array_equal(np.asarray([v['command']for v in task._pilot_action_rows],np.float32).reshape(count,14),actions[:count]):raise RuntimeError('Executed action recording differs')
  arrays={('legal__'+key):np.asarray(value).copy()for key,value in wrapper.last_legal_inputs.items()}
  arrays.update(base=base.copy(),actions=actions.copy(),raw_delta=np.asarray(numeric['raw_delta']).copy(),applied_delta=delta.copy(),executed_actions=actions[:count].copy(),executed_base=base[:count].copy(),executed_delta=delta[:count].copy())
  folder=session/'pilot_traces';folder.mkdir(exist_ok=True);stem=task._pilot_stem
  npz=folder/(stem+f'_q{q:03d}.npz');np.savez_compressed(npz,**arrays)
  record=dict(query=q,executed=count,native_count=int(task.take_action_cnt),physics_tick=task._pilot_ticks,legal_array_sha256={key:array_hash(v)for key,v in arrays.items()},offline_only=dict(physical_before=before,physical_after=after,actions=task._pilot_action_rows))
  with gzip.open(folder/(stem+'.jsonl.gz'),'at',encoding='utf-8')as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
  task._pilot_evidence.extend([npz,folder/(stem+'.jsonl.gz')])
  return result
 return step
