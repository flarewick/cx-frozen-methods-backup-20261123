"""Once per accepted fast revision, only its newly visible ARM difference."""
import copy,numpy as np
from native_response_action_v1.compose import ARM
from cx_tta.multi_event import apply_arm_candidate
class FeedbackApplication:
 def __init__(self):self.pending=None;self.seen=set();self.records=[]
 def step(self,composer,base,reference,direction,actions,delta,event,update,*,condition):
  q=event['query'];entry=dict(query=q,condition=condition,applied=False,pending=False)
  if condition=='TTA'and update.get('accepted'):
   if update.get('mode')!='TTA' or update.get('query')!=q:raise ValueError('Actual accepted current-query TTA receipt required')
   key=(update['source_query'],update['query'],update['fast_after'])
   if key in self.seen:raise ValueError('Accepted feedback revision cannot be borrowed or reused')
   self.seen.add(key)
   if self.pending is not None:self.records.append(dict(self.pending,reason='superseded_by_newer_accepted_fast_revision',applied=False))
   self.pending=dict(source_query=update['source_query'],accepted_query=q,fast_after=update['fast_after'],deadline_query=q+2)
  if condition!='TTA':
   if self.pending is not None:raise ValueError('Fast application must be episode/branch local')
   return actions,delta,event,entry
  if self.pending is None:return actions,delta,event,entry
  event=copy.deepcopy(event);entry.update(self.pending,pending=True)
  if q>self.pending['deadline_query']:
   self.records.append(dict(entry,reason='expired_without_legal_current_query'));self.pending=None;return actions,delta,event,entry
  if event['decision']['active']:
   if event['adaptation_applied']:
    entry.update(applied=True,reason='original_event_already_applied_current_fast_direction');self.records.append(entry);self.pending=None
   return actions,delta,event,entry
  control=composer.controller;cfg=control.config
  if control.interventions>=cfg.max_interventions:entry['reason']='shared_episode_budget_exhausted';return actions,delta,event,entry
  if control.last_intervention is not None and q-control.last_intervention<=cfg.cooldown_queries:entry['reason']='original_base_only_cooldown';return actions,delta,event,entry
  if direction is None or direction.shape!=(32,12)or direction.dtype!=np.float32 or not np.isfinite(direction).all():raise ValueError('Fresh finite trusted fast direction required')
  correction=np.zeros_like(base);correction[:,ARM]=direction-reference[:,ARM]
  if np.any(correction[:,[6,13]])or not np.isfinite(correction).all()or float(np.max(np.abs(correction)))>.020001:raise ValueError('Fast-only correction exceeds original trust shift')
  if not np.any(correction):entry['reason']='no_actual_ARM_visible_fast_difference';return actions,delta,event,entry
  # No frozen residual is added at this new opportunity. Only this difference
  # exists, hence zero/shadow/no accepted update is exactly the original full.
  fresh,applied=apply_arm_candidate(base,correction,True)
  control.interventions+=1;control.last_intervention=q
  event['decision'].update(active=True,interventions=control.interventions,phase='accepted_feedback_fast_difference')
  event['adaptation_applied']=True;entry.update(applied=True,pending=False,reason='accepted_feedback_fast_difference',maximum_ARM_difference=float(np.abs(correction).max()),shared_intervention_count=control.interventions)
  event['feedback_fast_difference']=entry.copy();self.records.append(entry);self.pending=None
  return fresh,applied,event,entry
 def finish(self):return dict(records=copy.deepcopy(self.records),unapplied_pending=copy.deepcopy(self.pending),accepted_revisions=len(self.seen),zero_update_additional_action_exact_zero=True,original_shared_budget_and_cooldown_preserved=True)
