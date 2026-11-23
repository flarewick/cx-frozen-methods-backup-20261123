"""Actual CPU threads test fairness, exclusion and original per-case RNG routing."""
import contextlib,threading,time,random
import numpy as np
import torch
from .priority_lock import PriorityLock,FutureTail

def run():
 from native_expert_owner_dispatch_v1.cpu_checks import Coordinator
 from concurrent_rng import Lane
 c=Coordinator();c.rng.install();lock=PriorityLock(c.cancel);rows=[];errors=[];outputs={};active=[0];maximum=[0]
 def operation(index):
  torch.manual_seed(9000+index);np.random.seed(9000+index);random.seed(9000+index)
  return np.r_[torch.rand(4).numpy(),np.random.rand(4),[random.random()for _ in range(4)]]
 def worker(index,kind):
  try:
   outer=lock.background()if kind=='future'else contextlib.nullcontext()
   inner=lock.screen(lambda:True)if kind in('screen','future')else contextlib.nullcontext()
   with outer,inner,lock,c.rng.bind(Lane('fair_'+str(index))):
    active[0]+=1;maximum[0]=max(maximum[0],active[0]);rows.append((index,kind))
    outputs[index]=c.call(c.rng.current(),operation,index);active[0]-=1
  except BaseException as error:errors.append(repr(error));c.cancel.set()
 threads=[]
 try:
  # Hold only to enqueue all real test requests in a known order.
  with lock:
   for index,kind in enumerate(['recovery']*5+['screen']*5+['future']*2):
    thread=threading.Thread(target=worker,args=(index,kind));threads.append(thread);thread.start()
    deadline=time.monotonic()+5
    while True:
     with lock.condition:queued=len(lock.waiting)
     if queued==index+1:break
     assert time.monotonic()<deadline;time.sleep(.002)
  deadline=time.monotonic()+30
  while any(t.is_alive()for t in threads):
   assert time.monotonic()<deadline;c.service.process_one(.01)
  for thread in threads:thread.join()
  assert not errors,errors
  assert maximum[0]==1 and len(rows)==12 and not lock.waiting and lock.owner is None
  for kind in ['screen','recovery','future']:
   indices=[i for i,k in rows if k==kind];assert indices==sorted(indices)
  reference=Lane('fair_reference')
  for index in outputs:assert np.array_equal(outputs[index],c.call(reference,operation,index))
  # Future requests already waiting cannot be skipped by more than one foreground grants.
  future_positions=[i for i,(_,kind)in enumerate(rows)if kind=='future']
  assert future_positions[0]<=1 and future_positions[1]-future_positions[0]<=2,rows
  for start in range(len(rows)-1):
   prefix=rows[:start];remaining={k:sum(kind==k for _,kind in rows[start:])for k in ['screen','recovery']}
   if all(remaining.values()):
    foreground=[kind for _,kind in rows[start:]if kind!='future']
    if foreground[0]=='recovery':assert foreground[1]=='screen',rows
    if len(foreground)>=3:assert foreground[:3]!=['screen']*3,rows
  rejected=[];tail=PriorityLock(threading.Event())
  def future_tail():
   try:
    with tail.background(),tail.screen(lambda:False),tail:raise AssertionError('Tail dispatched')
   except FutureTail:rejected.append(True)
  thread=threading.Thread(target=future_tail);thread.start();thread.join(5)
  assert not thread.is_alive()and rejected==[True]and not tail.waiting and tail.owner is None
  cancelled=PriorityLock(threading.Event());cancelled.cancel.set()
  try:
   with cancelled:raise AssertionError('Cancelled entered')
  except InterruptedError:pass
  assert not cancelled.waiting and cancelled.owner is None
  return dict(passed=True,actual_thread_requests=12,maximum_original_planner_concurrency=maximum[0],FIFO_within_work_kind=True,waiting_screen_burst_maximum2_and_recovery_nonstarvation=True,maximum_foreground_grants_before_required_future=1,actual_ModelService_Torch_numpy_python_RNG_byte_exact=True,late_future_rejected_before_any_compute=True,cancelled_ticket_removed=True,no_GPU_or_native_or_optimizer_work=True)
 finally:c.rng.uninstall()
