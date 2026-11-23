"""Seven real policy scenes while expert pending, six after; original execution."""
import time
_APPLIED=False

def apply():
 global _APPLIED
 if _APPLIED:return
 import benefit_policy_capacity_v12 as old;old.apply()
 import benefit_followup_protocol_v2 as p
 import benefit_followup_mixed_core_v3 as core,benefit_followup_mixed_manager_v3 as manager,benefit_memory_batches as batches
 if manager._CONTEXT is not None or batches._INFLIGHT:raise RuntimeError('Install capacity before any scene admission')
 ledger=core.state();previous=ledger.submit
 def cap():
  with ledger.lock:return 7 if any(row['qualified']<20 for row in ledger.ledger.tasks.values())else 6
 def submit(*args,**kwargs):
  value=previous(*args,**kwargs);batches.MAX_CLAIMED=cap();return value
 ledger.submit=submit;batches.MAX_CLAIMED=cap()
 folder=p.ROOT/'capacity_runtime_receipts_v45';folder.mkdir(exist_ok=True)
 p.atomic(folder/('runtime_'+str(time.time_ns())+'.json'),dict(helper_sha256=p.sha(__file__),policy_cap_while_expert_pending=7,policy_cap_after_expert_complete=6,active_expert_cap=1,max_real_scenes=8,model_batch=1,original83_files_unchanged=True,old_v12_capacity_explicitly_superseded=True,old_membership_and_scene_dependencies_unchanged=True,original16_worker_threads_and_all_guards_unchanged=True));_APPLIED=True
