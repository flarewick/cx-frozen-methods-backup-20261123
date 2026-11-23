"""Release real lanes on completed original queries, never on a timer."""
import threading,time
class StagedLanes:
 def __init__(self,cancel,emit,*,lanes=10):
  assert lanes in(8,10)
  self.cancel=cancel;self.emit=emit;self.total=lanes;self.four=threading.Event();self.eight=threading.Event();self.lock=threading.Lock();self.started=time.monotonic()
 def wait(self,number):
  if not 0<=number<self.total:raise ValueError('Invalid lane')
  while number>=2 and not self.eight.wait(.05):
   if self.cancel.is_set():return False
  return not self.cancel.is_set()
 def release(self,reason,*,lane,query=None,tick=None):
  if reason not in('actual_once_policy_query_complete','actual_case_terminal'):raise ValueError('Actual native evidence required')
  if reason=='actual_once_policy_query_complete'and(query!=0 or tick!=0):return False
  if not 0<=lane<self.total:raise ValueError('Invalid lane')
  if lane>=2:return False
  with self.lock:
   if self.eight.is_set():return False
   self.emit(dict(reason=reason,lane=lane,query=query,tick=tick,seconds_since_runtime_start=time.monotonic()-self.started,released_lanes=list(range(2,self.total)),continuous_resource_clock=True))
   self.four.set();self.eight.set();return True

def actor_class(original,gate,case_to_lane):
 class StagedOriginalActor(original):
  def query(self,observed,record,folder):
   value=super().query(observed,record,folder)
   gate.release('actual_once_policy_query_complete',lane=case_to_lane[folder.parent.name],query=record['query'],tick=record['global_tick'])
   return value
 return StagedOriginalActor
