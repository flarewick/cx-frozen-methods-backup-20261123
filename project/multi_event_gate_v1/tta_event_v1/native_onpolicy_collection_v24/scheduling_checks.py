"""Actual threads test query barriers, recovery priority and cancellation."""
import threading,time
from .priority_lock import PriorityLock
from .work import FixedWorkQueue
def run():
 cancel=threading.Event();lock=PriorityLock(cancel);order=[];errors=[]
 def worker(name,background):
  try:
   if background:
    with lock.background(),lock:order.append(name)
   else:
    with lock:order.append(name)
  except BaseException as e:errors.append(repr(e))
 def queued(count):
  deadline=time.monotonic()+3
  while True:
   with lock.condition:n=len(lock.waiting)
   if n==count:return
   assert time.monotonic()<deadline;time.sleep(.002)
 with lock:
  background=threading.Thread(target=worker,args=('future',True));background.start();queued(1)
  current=threading.Thread(target=worker,args=('live',False));current.start();queued(2)
 for t in(current,background):t.join(3);assert not t.is_alive()
 assert not errors and order==['live','future'],(errors,order)
 # Selection alone must never unblock the first cold case.
 w=FixedWorkQueue(3,('a',),{'a':dict(items=[0,1])},cancel)
 for n in range(2):assert w.take()==n;w.mark_selected(n);w.register_case(str(n),n)
 taken=[];started=threading.Event()
 def cold():started.set();taken.append(w.take())
 t=threading.Thread(target=cold);t.start();started.wait(3)
 with w.condition:assert not taken
 w.mark_case_ready('0')
 with w.condition:assert not taken
 w.mark_case_ready('1');t.join(3);assert not t.is_alive()and taken==[2]
 assert w.ready=={0,1}
 # Original non-reentrant semantics and cancellation retained.
 cancel.set()
 try:
  with lock:raise AssertionError('Cancellation did not reject acquisition')
 except InterruptedError:pass
 assert lock.owner is None and not lock.waiting
 return dict(passed=True,actual_threads=True,current_recovery_precedes_waiting_future_expert=True,
             cold_expert_waits_for_actual_cached_queries=True,selection_alone_does_not_release=True,
             cancellation_fail_closed=True,no_model_physics_or_GPU_work=True)
