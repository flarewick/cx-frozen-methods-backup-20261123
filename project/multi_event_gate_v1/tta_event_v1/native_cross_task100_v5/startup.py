"""Release real lanes on completed original queries, never on a timer."""
import threading,time
class StagedLanes:
 def __init__(self,cancel,emit):
  self.cancel=cancel;self.emit=emit;self.four=threading.Event();self.eight=threading.Event();self.lock=threading.Lock();self.started=time.monotonic()
 def wait(self,number):
  if not 0<=number<8:raise ValueError('Invalid lane')
  event=None if number<2 else self.four if number<4 else self.eight
  while event is not None and not event.wait(.05):
   if self.cancel.is_set():return False
  return not self.cancel.is_set()
 def release(self,reason,*,lane,query=None,tick=None):
  if reason not in('actual_once_policy_query_complete','actual_case_terminal'):raise ValueError('Actual native evidence required')
  if reason=='actual_once_policy_query_complete'and(query!=0 or tick!=0):return False
  if not 0<=lane<8:raise ValueError('Invalid lane')
  if lane>=4:return False
  event=self.four if lane<2 else self.eight
  with self.lock:
   if event.is_set():return False
   if lane>=2 and not self.four.is_set():raise RuntimeError('Second stage released before first')
   self.emit(dict(reason=reason,lane=lane,query=query,tick=tick,seconds_since_runtime_start=time.monotonic()-self.started,released_lanes=[2,3]if lane<2 else[4,5,6,7],continuous_resource_clock=True))
   event.set();return True

def actor_class(original,gate,case_to_lane):
 class StagedOriginalActor(original):
  def query(self,observed,record,folder):
   value=super().query(observed,record,folder)
   gate.release('actual_once_policy_query_complete',lane=case_to_lane[folder.parent.name],query=record['query'],tick=record['global_tick'])
   return value
 return StagedOriginalActor
