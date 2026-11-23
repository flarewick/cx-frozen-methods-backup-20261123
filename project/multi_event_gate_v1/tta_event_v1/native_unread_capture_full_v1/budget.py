"""Keep every used second; no automatic extension of the approved VAL cap."""
from native_policy_execution_v1.common import read,sha
from native_onpolicy_recovery_v3.common import verify
USED=12409.593
MINIMUM_PLANNED_FULL_BATCH_SECONDS=3600

def authorization(T):
 original=T/'outputs/native_onpolicy_full_ops_v2/ARM18_VAL_budget_authorization_v1.json'
 base=read(original);verify(base)
 assert base['explicit_user_authorization']and base['approved_total_seconds']==14400
 assert base['previous_total_seconds']==10800 and abs(base['all_used_seconds_retained']-10723.073)<1e-6
 amendment=T/'outputs/native_onpolicy_full_ops_v2/ARM29_VAL_budget_authorization_v1.json'
 if not amendment.exists():return original,base
 value=read(amendment);verify(value)
 assert value['explicit_user_authorization']and value['approved_total_seconds']==18000
 assert value['previous_total_seconds']==14400 and abs(value['all_used_seconds_retained']-USED)<1e-6
 assert value['scope']=='cumulative_VAL_diagnostic_only'
 assert value['input_sha256'][str(original)]==sha(original)
 return amendment,dict(base,approved_total_seconds=18000)

def require_planned_full_batch(spec):
 if spec['dataset_split']!='adapter_validation':return
 remaining=spec['stage_total_GPU_budget_seconds']-spec['inherited_stage_GPU_seconds']
 assert remaining>=MINIMUM_PLANNED_FULL_BATCH_SECONDS,(
  'Explicit finite VAL budget review required before this whole25-pair batch',remaining,
  MINIMUM_PLANNED_FULL_BATCH_SECONDS,'No GPU launch or study freeze at the current14400 cap')
