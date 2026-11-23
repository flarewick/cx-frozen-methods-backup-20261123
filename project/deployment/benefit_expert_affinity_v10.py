"""Host scheduling only: one private expert task until its fixed20 is complete.

Concentrate first on the task with the smaller existing eligibility prefix.
No policy outcomes enter this choice. Original calls, RNG and ledgers stay intact.
"""
import threading,time,json
_APPLIED=False
PREFIX='mixed_original_expert/'

class TaskAffineSemaphore:
 def __init__(self,ledger):
  self.ledger=ledger;self.token=threading.BoundedSemaphore(1)
  with ledger.lock:self.order=tuple(sorted(ledger.ledger.tasks,key=lambda t:(ledger.ledger.tasks[t]['qualified'],t)))
  self.owner=None
 def selected(self):
  with self.ledger.lock:
   return next((t for t in self.order if self.ledger.ledger.tasks[t]['qualified']<20),None)
 def acquire(self,blocking=True,timeout=None):
  name=threading.current_thread().name
  if not name.startswith(PREFIX):raise RuntimeError('Expert slot requires original task lane')
  task=name[len(PREFIX):]
  if task not in self.order:raise RuntimeError('Unknown original expert task')
  end=None if timeout is None else time.monotonic()+timeout
  while True:
   with self.ledger.lock:done=self.ledger.ledger.tasks[task]['qualified']==20
   chosen=self.selected()
   if done or chosen==task or chosen is None:
    left=-1 if end is None else max(0,end-time.monotonic())
    ok=self.token.acquire(blocking=blocking,timeout=left)if blocking else self.token.acquire(False)
    if ok:self.owner=threading.get_ident()
    return ok
   if not blocking:return False
   left=.05 if end is None else min(.05,max(0,end-time.monotonic()))
   if left<=0:return False
   time.sleep(left)
 def release(self):
  if self.owner!=threading.get_ident():raise RuntimeError('Expert slot released by another lane')
  self.owner=None;self.token.release()

def apply():
 global _APPLIED
 if _APPLIED:return
 import benefit_followup_protocol_v2 as p
 import benefit_followup_mixed_core_v3 as core,benefit_followup_mixed_manager_v3 as manager
 import benefit_memory_batches as batches
 from benefit_followup_stream_v3 import atomic
 ledger=core.state()
 if manager._CONTEXT is not None or batches._INFLIGHT:raise RuntimeError('Install before original workers')
 semaphore=TaskAffineSemaphore(ledger);manager._ACTIVE=semaphore;original=ledger.submit
 def refresh():
  with ledger.lock:pending=any(v['qualified']<20 for v in ledger.ledger.tasks.values())
  batches.MAX_CLAIMED=5 if pending else 6
 def submit(*a,**kw):
  value=original(*a,**kw);refresh();return value
 ledger.submit=submit;refresh()
 receipt=dict(pure_host_scheduling_override=True,helper_sha256=p.sha(__file__),
  immutable_membership_scope_sha256=p.sha(core.scope_path()),
  policy_cap=batches.MAX_CLAIMED,expert_cap=1 if semaphore.selected()else 0,max_real_scenes=6,
  expert_task_order=list(semaphore.order),initial_active_task=semaphore.selected(),
  task_order_uses_expert_prefix_counts_only=True,policy_outcomes_never_read=True,
  each_task_original_decision_order_and_original_private_planners_retained=True,
  second_task_waits_until_first_fixed20_complete=True,
  original_model_physics_and_RNG_functions_untouched=True,all_resource_guards_unchanged=True)
 folder=p.ROOT/'affinity_runtime_receipts_v10';folder.mkdir(exist_ok=True)
 atomic(folder/('runtime_'+str(time.time_ns())+'.json'),receipt);_APPLIED=True
