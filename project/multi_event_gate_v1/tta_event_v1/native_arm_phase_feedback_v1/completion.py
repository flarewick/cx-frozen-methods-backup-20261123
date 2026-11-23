"""Requested path progress of all moving arms, from own native motor targets."""
import copy,hashlib
import numpy as np
ARM=np.r_[np.arange(6),np.arange(7,13)]
class ArmPathCompletion:
 def __init__(self,initial,commands):
  commands=np.asarray(commands);initial=np.asarray(initial)
  if commands.shape!=(32,14)or commands.dtype!=np.float32 or initial.shape!=(14,)or not np.isfinite(commands).all()or not np.isfinite(initial).all():raise ValueError('Known finite initial own state and actually issued32 commands required')
  points=np.r_[initial[None,ARM],commands[:,ARM]].astype(np.float64).reshape(33,2,6)
  self.starts=points[:-1];self.delta=np.diff(points,axis=0);self.length=np.linalg.norm(self.delta,axis=2);self.prefix=np.r_[np.zeros((1,2)),self.length.cumsum(axis=0)];self.total=self.prefix[-1];self.moving=self.total>1e-6
  if not self.moving.any():raise ValueError('No actual requested ARM path')
  self.index=-1;self.tick=0;self.side95=[None,None];self.closed=[];self.credited=np.zeros(2);self.fraction_max=np.zeros(2)
  self.pending={'event':dict(request_id='event',origin='origin',kind='initial_own_ARM_path',requested95_tick=None,onset_tick=None,actual_ticks=0,last_tick=0,origin_completed_tick=None,planned_base_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),moving_arms=np.flatnonzero(self.moving).tolist(),known_planned_path_length=self.total.tolist(),physical_contact_certified=False,own_command_targets_only=True,all_moving_arms_required=True)}
 def command_begin(self,index):
  if type(index)is not int or index!=self.index+1 or not 0<=index<32 or not self.pending:raise ValueError('One contiguous actually issued command index required')
  if self.index>=0:self.credited+=self.fraction_max*self.length[self.index]
  self.fraction_max[:]=0.;self.index=index
 def after_arm_actuation(self,targets,*,committed_tick,dt_s):
  targets=np.asarray(targets)
  if not self.pending or self.index<0 or targets.shape!=(12,)or not np.isfinite(targets).all()or type(committed_tick)is not int or committed_tick!=self.tick+1 or not np.isfinite(dt_s)or abs(float(dt_s)-.004)>1e-8:raise ValueError('Exactly next actual native own-target step required')
  self.tick=committed_tick;r=self.pending['event'];r['actual_ticks']+=1;r['last_tick']=committed_tick
  i=self.index;v=targets.astype(float).reshape(2,6)-self.starts[i];denom=self.length[i]**2
  fraction=np.divide(np.sum(v*self.delta[i],axis=1),denom,out=np.zeros(2),where=denom>1e-20);fraction=np.clip(fraction,0.,1.)
  self.fraction_max=np.maximum(self.fraction_max,fraction)
  progress=self.credited+self.fraction_max*self.length[i]
  ratio=np.divide(progress,self.total,out=np.ones(2),where=self.moving)
  if r['onset_tick']is None and np.any(ratio[self.moving]>=.1):r['onset_tick']=committed_tick
  for arm in np.flatnonzero(self.moving):
   if self.side95[arm]is None and ratio[arm]>=.95:self.side95[arm]=committed_tick
  if r['requested95_tick']is None and all(self.side95[a]is not None for a in np.flatnonzero(self.moving)):r['requested95_tick']=max(self.side95[a]for a in np.flatnonzero(self.moving))
 def finish_origin(self,origin,*,arrived_tick,commands_returned,native_censored):
  if origin!='origin'or arrived_tick!=self.tick or type(commands_returned)is not int or not 0<=commands_returned<=32 or type(native_censored)is not bool or not self.pending:raise ValueError('Truthful current native receipt required once')
  r=self.pending['event'];r['commands_returned']=commands_returned;r['side95_ticks']=self.side95.copy()
  if native_censored or commands_returned!=32 or self.index!=31 or r['requested95_tick']is None:
   self.closed.append(dict(r,status='censored',reason='actual32_or_all_ARM_targets_not_completed'));self.pending.clear()
  else:r['origin_completed_tick']=arrived_tick
 def observe_rgb(self,origin,*,observation_tick,decision_tick,frame_id):
  if origin!='origin'or observation_tick!=decision_tick or observation_tick!=self.tick or not isinstance(frame_id,str):raise ValueError('Only actual current arrived RGB boundary')
  if not self.pending:return[]
  r=self.pending['event']
  if r['origin_completed_tick']!=observation_tick:raise ValueError('No feedback before all actual32 returned')
  row=dict(r,status='feedback_ready',reason='all_actual_own_ARM_path95_milestones',observation_tick=observation_tick,frame_id=frame_id)
  self.closed.append(row);self.pending.clear();return[row]
