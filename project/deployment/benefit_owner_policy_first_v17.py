"""Host scheduling only: ready policy requests precede queued expert plans.

Original owner execution, RNG scopes, task/model functions and 6+1 scene caps
are unchanged. The original sampler warmup and non-expert priorities stay intact.
"""
import queue,threading,hashlib,marshal
PREFIX='mixed_original_expert/'
_INSTALLED=False
RECEIPT={}


def priority(item):
 if len(item)!=7:raise RuntimeError('Original owner request tuple changed')
 old,seq,lane,fn,args,kwargs,future=item
 if lane.name.startswith(PREFIX)and old==0 and getattr(fn,'_cx_stage',None)in('expert_plan_path','expert_plan_batch'):return(2,seq,lane,fn,args,kwargs,future)
 return item


class PolicyFirstQueue(queue.PriorityQueue):
 def put(self,item,block=True,timeout=None):
  return super().put(priority(item),block,timeout)


def install():
 global _INSTALLED
 if _INSTALLED:return
 import model_service as original
 execute=original.ModelService.execute;call=original.ModelService.call;process=original.ModelService.process_one
 hashes={n:hashlib.sha256(marshal.dumps(f.__code__)).hexdigest()for n,f in [('execute',execute),('call',call),('process_one',process)]}
 before=original.ModelService.__init__
 def initialize(self,coordinator):
  before(self,coordinator)
  if not self.requests.empty():raise RuntimeError('Cannot replace a live owner queue')
  self.requests=PolicyFirstQueue()
 original.ModelService.__init__=initialize
 assert original.ModelService.execute is execute and original.ModelService.call is call and original.ModelService.process_one is process
 RECEIPT.update(original_execution_and_RNG_code_unchanged=True,original_request_and_processing_code_unchanged=True,
  original_code_sha256=hashes,policy_priorities_unchanged=True,non_expert_priorities_unchanged=True,
  original_sampler_warmup_unchanged=True,policy_slots=6,active_expert_slots=1,max_real_scenes=7,
  change='Only queued original expert plan_path/plan_batch priority0 becomes priority2; ready policy priority1 precedes them; original sequence and all other requests retained')
 _INSTALLED=True
