"""Fourteen real independent scenes, seven/task; one unchanged sampler owner."""
import threading,functools
from pathlib import Path
INSTALLED=False
def full(queue):
 import benefit_followup_protocol_v2 as p,benefit_memory_batches as b
 task,branch=p.decode(queue.manifest['condition'])
 return sum(Path(run).parent.name==task for run,seed in b._INFLIGHT)>=7

def install():
 global INSTALLED
 if INSTALLED:return
 import benefit_memory_batches as b,benefit_stage_runtime_v2 as v2
 from concurrent_queue import EpisodeQueue
 import shared6_eval_backend as backend,benefit_followup_protocol_v2 as p
 assert not b._INFLIGHT and b.MAX_CLAIMED==10 and v2._slots._value==10
 b.MAX_CLAIMED=14;v2._slots=threading.BoundedSemaphore(14)
 actual=b.acquire.__closure__[b.acquire.__code__.co_freevars.index('acquire')].cell_contents
 for fn in (actual,EpisodeQueue.acquire):
  assert fn.__globals__['_transfer_task_full207'].__module__=='balanced_v207'
  fn.__globals__['_transfer_task_full207']=full
 import benefit_followup_mixed_manager_v3 as manager
 assert manager._ACTIVE.capacity==10;manager._ACTIVE.capacity=14
 previous=backend.make_plan
 def plan(*a,**kw):
  value=previous(*a,**kw)
  from evaluate import host
  import json
  for job in value['jobs']:
   file=host(job['run'])/'concurrency_queue.json';state=json.loads(file.read_text())if file.exists()else{};left=job['manifest']['episodes']-len(state.get('completed',{}))
   job['lane_count']=min(7,left)if not job['complete']else 0
  mode='validate'if value['task'].endswith('validate')else'collect';p.atomic(p.ROOT/(mode+'_plan.json'),value);p.atomic(p.ROOT/'active_plan.json',value);return value
 backend.make_plan=plan;INSTALLED=True
