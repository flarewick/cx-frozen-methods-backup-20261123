"""Admit two real lanes first; release all six after the first actual query."""
import threading,time
class StagedLanes:
 def __init__(self,cancel,emit):
  self.cancel,self.emit=cancel,emit;self.ready=threading.Event();self.lock=threading.Lock();self.started=time.monotonic()
 def wait(self,number):
  if number<2:return not self.cancel.is_set()
  while not self.ready.wait(.05):
   if self.cancel.is_set():return False
  return not self.cancel.is_set()
 def release(self,reason,**fields):
  if reason not in('actual_once_policy_query_complete','initial_case_terminal_without_query'):raise ValueError('Actual startup evidence required')
  with self.lock:
   if self.ready.is_set():return False
   self.emit(dict(reason=reason,seconds_since_runtime_start=time.monotonic()-self.started,initial_lanes=[0,1],released_lanes=[2,3,4,5],**fields))
   self.ready.set();return True
