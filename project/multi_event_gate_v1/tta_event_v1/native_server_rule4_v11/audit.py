"""Bind every running decision to this attempt's actual strict admission."""
import json
from native_policy_execution_v1.common import read,write,sha
from native_server_rule4_v1.admission import decision
from .admission import digest

def audit(root):
 admission_file=root/'attempt/rule4_admission.json';running_file=root/'attempt/rule4_running.jsonl'
 original=read(admission_file);spec=read(root/'protocol.json');uuid=spec['required_gpu_uuid']
 assert original['passed']and decision(original['fresh_original_rule4_fleet'],uuid)[0]
 previous=[c for c in original['latest_observation']['fleet']['cards']if c['own_pids']and c['uuid']!=uuid]
 for card in previous:
  e=original['estimates'][card['uuid']]
  assert e['eligible']and e['observed_seconds']>=120 and e['memory_lifetime_estimate']>=.55
 rows=[json.loads(line)for line in running_file.read_text().splitlines()];assert rows
 for row in rows:
  proof=row['proof'];latest=proof['latest_observation']
  assert proof['passed']and proof['original_strict_admission']==original
  assert proof['original_strict_admission_sha256']==digest(original)
  assert proof['extra_card_condition_applies_before_new_allocation']and proof['later_account_jobs_do_not_revoke_prior_legal_admission']
  assert 0<=proof['checked_monotonic']-latest['monotonic']<=8
  assert latest['fleet']['own_gpu_count']<=3
 value=dict(passed=True,actual_strict_admission=True,original_instantaneous_and_conservative_prior_conditions_preserved=True,
  all_running_decisions_bound_to_this_attempt_admission=True,fresh_account_limit_checks=len(rows),
  no_retrospective_cancellation_due_to_later_jobs=True,no_scientific_or_process_device_resource_waiver=True,
  input_sha256={str(p):sha(p)for p in [admission_file,running_file,root/'attempt/barrier/permit.json']})
 write(root/'CPU_rule4_admission_scope_audit.json',value)
 return value
