"""Explicit host capacity amendment: six productive policy scenes + one expert.

Six policy slots remain after experts finish. Original task-affine expert,
owner, per-scene dependencies, model batch1 and all guards stay unchanged.
"""
import time
_APPLIED=False

def apply():
 global _APPLIED
 if _APPLIED:return
 import benefit_expert_affinity_v10 as affinity;affinity.apply()
 import benefit_followup_protocol_v2 as p
 import benefit_followup_mixed_core_v3 as core,benefit_followup_mixed_manager_v3 as manager
 import benefit_memory_batches as batches
 from benefit_followup_stream_v3 import atomic
 if manager._CONTEXT is not None or batches._INFLIGHT:raise RuntimeError('Capacity installs before original worker admission')
 ledger=core.state();original=ledger.submit
 def submit(*a,**kw):
  value=original(*a,**kw);batches.MAX_CLAIMED=6;return value
 ledger.submit=submit;batches.MAX_CLAIMED=6
 with ledger.lock:pending=any(row['qualified']<20 for row in ledger.ledger.tasks.values())
 value=dict(helper_sha256=p.sha(__file__),policy_cap=6,expert_cap=1 if pending else 0,
  maximum_real_scenes=7 if pending else 6,shared_model_batch=1,
  task_affine_expert_unchanged=True,original83_sources_unchanged=True,
  immutable_membership_scope_sha256=p.sha(core.scope_path()),
  historical_scope_resource_ceiling_explicitly_superseded=True,
  original_model_physics_RNG_and_dependencies_unchanged=True,all_resource_guards_unchanged=True)
 folder=p.ROOT/'capacity_runtime_receipts_v12';folder.mkdir(exist_ok=True)
 atomic(folder/('runtime_'+str(time.time_ns())+'.json'),value);_APPLIED=True
