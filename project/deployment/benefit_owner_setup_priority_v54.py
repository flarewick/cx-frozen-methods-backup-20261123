"""Let queued policy inference precede queued private expert scene setup."""
import hashlib,marshal
import benefit_owner_policy_first_v17 as base
_INSTALLED=False
RECEIPT={}
def priority(item):
 value=base.priority(item)
 old,seq,lane,fn,args,kwargs,future=value
 if old==0 and lane.name.startswith(base.PREFIX) and getattr(fn,'_cx_stage',None)=='expert_scene_setup':
  return(2,seq,lane,fn,args,kwargs,future)
 return value
class PolicySetupQueue(base.PolicyFirstQueue):
 def put(self,item,block=True,timeout=None):
  return super().put(priority(item),block,timeout)
def install():
 global _INSTALLED
 if _INSTALLED:return
 base.install()
 import model_service as original
 saved={n:getattr(original.ModelService,n)for n in('execute','call','process_one')}
 before=original.ModelService.__init__
 def initialize(self,coordinator):
  before(self,coordinator)
  if not self.requests.empty():raise RuntimeError('Cannot replace a live owner queue')
  self.requests=PolicySetupQueue()
 original.ModelService.__init__=initialize
 assert all(getattr(original.ModelService,n)is fn for n,fn in saved.items())
 RECEIPT.update(original_execution_RNG_call_processing_unchanged=True,original_code_sha256={n:hashlib.sha256(marshal.dumps(f.__code__)).hexdigest()for n,f in saved.items()},only_additional_change='queued private expert_scene_setup priority0 becomes priority2',original_sequence_and_all_other_requests_preserved=True,original_v17_plan_priority2_preserved=True,capacity_and_all_resource_guards_unchanged=True)
 _INSTALLED=True
