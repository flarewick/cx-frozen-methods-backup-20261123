"""Bind each closed native expert lifetime; no failed data promotion."""
import json
from native_policy_execution_v1.common import read,write,sha
def audit(root,spec):
 logs=[root/'closed_expert_task_release.jsonl',root/'next_preparation/closed_expert_task_release.jsonl'];rows=[];bindings={}
 for file in logs:
  if not file.exists():continue
  bindings[str(file)]=sha(file);rows.extend(json.loads(line)for line in file.read_text().splitlines())
 expected=spec['cases_per_batch']-(spec.get('cached_preparation')or{}).get('cached_valid_cases',0)+spec['future_preparation_each_task']*len(spec['tasks'])
 assert len(rows)==expected,('Closed expert release population mismatch',len(rows),expected)
 for row in rows:
  assert row['closed_task_and_scene_weakrefs_dead']and row['closed_proxy_weakrefs_dead']and row['real_planner_constructors_not_repeated']and row['model_versions_and_RNG_unchanged']and row['global_renderer_cache_not_cleared']and row['no_extra_queries_or_physics']
  assert len(row['original_shared_planner_identities_preserved'])==2 and row['before']is not None and row['after']is not None
 if rows:assert len({tuple(row['original_shared_planner_identities_preserved'])for row in rows})==1
 value=dict(passed=True,closed_expert_tasks=len(rows),all_closed_scene_and_task_and_proxy_weakrefs_dead=True,one_original_shared_pair_preserved=True,original_model_and_RNG_preserved=True,no_global_renderer_cache_clear=True,input_sha256=bindings)
 write(root/'CPU_closed_expert_task_release_audit.json',value);return value
