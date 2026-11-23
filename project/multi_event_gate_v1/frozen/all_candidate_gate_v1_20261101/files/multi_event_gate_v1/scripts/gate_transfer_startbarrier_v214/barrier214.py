"""Initial genuine scene/action admission barrier; unchanged per-scene execution."""
import threading,time,functools
INSTALLED=False
class FirstActionBarrier:
 def __init__(self,c,target,note=lambda **kw:None):
  assert 1<=target<=14;self.c=c;self.target=target;self.note=note;self.condition=threading.Condition();self.seen=set();self.open=False;self.began=time.monotonic()
 def enter(self,task):
  if self.open or not getattr(task,'_pilot_recording',False):return
  if threading.get_ident()==self.c.service.owner:raise RuntimeError('Physical admission wait must not block model owner')
  with self.condition:
   self.seen.add(id(task))
   if len(self.seen)>=self.target:
    self.open=True;self.note(target=self.target,real_tasks_at_first_action=len(self.seen),wait_seconds=time.monotonic()-self.began,all_original_initializations_and_first_queries_retained=True);self.condition.notify_all();return
   while not self.open:
    self.c.check_cancel();self.condition.wait(.05)
 def wrap(self,task,original):
  @functools.wraps(original)
  def action(*args,**kwargs):self.enter(task);return original(*args,**kwargs)
  return action

def install():
 global INSTALLED
 if INSTALLED:return
 from model_service import ModelService
 import startup211
 current=ModelService.call
 assert current.__globals__ is startup211.__dict__ and hasattr(current,'__wrapped__')
 # Restore original request priorities; keep CPU full-source SHA and Warp helpers.
 ModelService.call=current.__wrapped__
 import benefit_pilot_trace as trace,benefit_memory_batches as batches,benefit_followup_protocol_v2 as p
 previous=trace.install
 def install_trace(task,lane,c):
  value=previous(task,lane,c)
  if not hasattr(c,'_first_action_barrier214'):
   with batches._LOCK:target=len(batches._INFLIGHT)
   c._first_action_barrier214=FirstActionBarrier(c,target,note=lambda **kw:p.atomic(p.ROOT/'startup_barrier214_open.json',kw))
  task.take_action=c._first_action_barrier214.wrap(task,task.take_action)
  return value
 trace.install=install_trace;INSTALLED=True
