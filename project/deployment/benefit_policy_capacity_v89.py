"""Eight genuine policy scenes plus at most one original task-affine expert."""
import time
_APPLIED=False
def apply():
 global _APPLIED
 if _APPLIED:return
 import benefit_policy_capacity_v45 as previous;previous.apply()
 import benefit_followup_protocol_v2 as p
 import benefit_followup_mixed_core_v3 as core,benefit_followup_mixed_manager_v3 as manager,benefit_memory_batches as batches
 assert manager._CONTEXT is None and not batches._INFLIGHT
 ledger=core.state();submit_before=ledger.submit
 def submit(*a,**kw):
  result=submit_before(*a,**kw);batches.MAX_CLAIMED=8;return result
 ledger.submit=submit;batches.MAX_CLAIMED=8
 folder=p.ROOT/'capacity_runtime_receipts_v89';folder.mkdir(exist_ok=True)
 p.atomic(folder/(str(time.time_ns())+'.json'),dict(helper_sha256=p.sha(__file__),policy_pending=8,policy_after_experts=8,expert_cap=1,max_real_scenes=9,batch=1,prior_capacity45_explicitly_superseded=True,scientific83_and_guards_unchanged=True,actual_whole_episode_claims_only=True));_APPLIED=True
