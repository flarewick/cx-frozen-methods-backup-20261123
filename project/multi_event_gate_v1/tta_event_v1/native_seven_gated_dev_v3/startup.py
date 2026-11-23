"""Release real lanes on completed original queries, never on a timer."""
import threading,time
class StagedLanes:
 def __init__(self,cancel,emit,*,lanes=10):
  assert lanes in(4,8,9,10,12)
  self.cancel=cancel;self.emit=emit;self.total=lanes;self.four=threading.Event();self.eight=threading.Event();self.lock=threading.Lock();self.started=time.monotonic()
 def wait(self,number):
  if not 0<=number<self.total:raise ValueError('Invalid lane')
  while number>=4 and not self.eight.wait(.05):
   if self.cancel.is_set():return False
  return not self.cancel.is_set()
 def release_q0(self,initial,case_id,content_key):
  # Scheduling evidence only: no work.mark_ready or fabricated actor query.
  import numpy as np
  assert initial['native_action_count']==0 and len(initial['rgb'])==3
  assert all(np.asarray(im).shape==(240,320,3)and np.asarray(im).dtype==np.uint8 for im in initial['rgb'])
  assert np.asarray(initial['command_state']).shape==(14,)and np.isfinite(initial['command_state']).all()
  assert isinstance(content_key,str)and len(content_key)==64
  assert isinstance(case_id,str)and case_id
  with self.lock:
   if self.eight.is_set():return False
   self.emit(dict(reason='actual_verified_native_q0_RGB_command_prefetch',case_id=case_id,content_key=content_key,native_action_count=0,seconds_since_runtime_start=time.monotonic()-self.started,released_lanes=list(range(4,self.total)),continuous_resource_clock=True,no_original_actor_query_fabricated=True))
   self.eight.set();return True
 def release(self,reason,*,lane,query=None,tick=None):
  if reason not in('actual_once_policy_query_complete','actual_case_terminal'):raise ValueError('Actual native evidence required')
  if reason=='actual_once_policy_query_complete'and(query!=0 or tick!=0):return False
  if not 0<=lane<self.total:raise ValueError('Invalid lane')
  if lane>=4:return False
  with self.lock:
   if self.eight.is_set():return False
   self.emit(dict(reason=reason,lane=lane,query=query,tick=tick,seconds_since_runtime_start=time.monotonic()-self.started,released_lanes=list(range(4,self.total)),continuous_resource_clock=True))
   self.four.set();self.eight.set();return True

def actor_class(original,gate,case_to_lane,work):
 class StagedOriginalActor(original):
  def query(self,observed,record,folder):
   value=super().query(observed,record,folder)
   if record['query']==0 and record['global_tick']==0:work.mark_case_ready(folder.parent.name)
   gate.release('actual_once_policy_query_complete',lane=case_to_lane[folder.parent.name],query=record['query'],tick=record['global_tick'])
   return value
 return StagedOriginalActor
