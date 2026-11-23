"""Feedback after real requested-origin completion, not after first grip motion.

All inputs are once-known planned base and actually committed command receipts.
No measured joints, object poses, contact, task, seed or outcome is accepted.
The queue never advances the simulator or borrows later unrelated commands.
"""
import hashlib
import numpy as np
KINDS={'close_left':(0,1),'close_right':(1,1),'open_left':(0,-1),'open_right':(1,-1)}
class CompletionQueue:
 def __init__(self):self.pending={};self.closed=[];self.tick=-1;self.origin=None
 def register(self,request_id,origin,kind,initial_grip,once_sampled_base,*,decision_tick,frame_id):
  if kind not in KINDS or type(decision_tick)is not int or decision_tick<0 or decision_tick<self.tick or request_id in self.pending:raise ValueError('Unique known event at arrived decision required')
  grip=np.asarray(initial_grip,float);base=np.asarray(once_sampled_base)
  if grip.shape!=(2,) or base.shape!=(32,14) or not np.isfinite(grip).all() or not np.isfinite(base).all() or not isinstance(frame_id,str):raise ValueError('Own finite initial command, once-sampled32base and RGB id required')
  side,sign=KINDS[kind];g0=float(np.clip(grip[side],0,1));planned=np.clip(base[:,[6,13]][:,side],0,1);goal=float(planned.min() if sign==1 else planned.max());amplitude=sign*(g0-goal)
  if amplitude<.1:raise ValueError('No unambiguous requested gripper excursion of .1')
  self.tick=decision_tick;self.origin=origin;self.pending[request_id]=dict(request_id=request_id,origin=origin,kind=kind,side=side,sign=sign,initial_grip=g0,planned_goal=goal,planned_amplitude=amplitude,planned_base_sha256=hashlib.sha256(base.tobytes()).hexdigest(),decision_tick=decision_tick,initial_frame_id=frame_id,onset_tick=None,requested95_tick=None,actual_ticks=0,last_tick=decision_tick,origin_completed_tick=None)
 def after_actuation(self,origin,gripper_logical_command,*,committed_tick,dt_s):
  grip=np.asarray(gripper_logical_command,float)
  if type(committed_tick)is not int or committed_tick<=self.tick or grip.shape!=(2,) or not np.isfinite(grip).all() or not np.isfinite(dt_s) or dt_s<=0:raise ValueError('Strictly newer actual committed step required')
  self.tick=committed_tick;self.origin=origin
  for r in self.pending.values():
   if r['origin']!=origin or r['origin_completed_tick'] is not None:continue
   current=float(np.clip(grip[r['side']],0,1));change=r['sign']*(r['initial_grip']-current);r['actual_ticks']+=1;r['last_tick']=committed_tick
   if r['onset_tick'] is None and change>=.1:r['onset_tick']=committed_tick
   if r['requested95_tick'] is None and r['onset_tick'] is not None and change/r['planned_amplitude']>=.95:r['requested95_tick']=committed_tick
 def finish_origin(self,origin,*,arrived_tick,commands_returned,native_censored):
  if type(arrived_tick)is not int or arrived_tick!=self.tick or type(commands_returned)is not int or not 0<=commands_returned<=32 or type(native_censored)is not bool:raise ValueError('Truthful current native-origin completion receipt required')
  for key,r in list(self.pending.items()):
   if r['origin']!=origin:continue
   if native_censored or commands_returned!=32 or not r['actual_ticks']:
    self.closed.append(dict(r,status='censored',reason='original_native_command_not_completed',commands_returned=commands_returned));del self.pending[key]
   elif r['requested95_tick'] is None:
    self.closed.append(dict(r,status='censored',reason='requested_target_not_observed',commands_returned=32));del self.pending[key]
   else:r['origin_completed_tick']=arrived_tick
 def observe_rgb(self,origin,*,observation_tick,decision_tick,frame_id):
  if type(observation_tick)is not int or observation_tick!=decision_tick or observation_tick!=self.tick or not isinstance(frame_id,str):raise ValueError('Only actual currently arrived RGB')
  ready=[]
  for key,r in list(self.pending.items()):
   if r['origin']!=origin:continue
   if r['origin_completed_tick'] is None:continue
   if r['origin_completed_tick']!=observation_tick:raise ValueError('Cannot credit unrelated later actions to completed origin')
   row=dict(r,status='feedback_ready',observation_tick=observation_tick,frame_id=frame_id,physical_contact_certified=False,only_actual_prefix=True);ready.append(row);self.closed.append(row);del self.pending[key]
  return ready
 def reset(self):self.pending.clear();self.closed.clear();self.tick=-1;self.origin=None
