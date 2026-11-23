"""Transferred scenes qualify only after actual whole full release and reference audit."""
import json
from .common import read,sha
from native_policy_execution_v1.common import write as global_write

def audit(root,spec):
 file=root/'q0_task_handoff.jsonl';rows=[json.loads(line)for line in file.read_text().splitlines()]if file.exists()else[]
 expected=spec['cases_per_batch']-(spec.get('cached_preparation')or{}).get('cached_valid_cases',0)
 assert len(rows)==expected and len({r['case_id']for r in rows})==expected
 inputs={str(file):sha(file)}if rows else{}
 for row in rows:
  assert row['exact_task_and_scene_retained']and row['expert_proxy_weakrefs_dead']and row['model_versions_and_RNG_unchanged']and row['no_extra_primary_queries_or_physics']and row['native_action_count']==0
  topp=row['scene_free_TOPP'];assert topp['passed']and topp['original_TOPP_function_objects_and_limits_exact']and topp['all_obsolete_scene_weakrefs_dead']and topp['current_native_scene_preserved']
  folder=root/'cases'/row['case_id']/'M_event_full'
  initial=folder.parent/'M_event_full_initialization_integrity.json';a=read(initial);assert a['exact_native_q0_task_handoff']and a['actual_passive_RGB_command_physics_RNG_exact']
  release=folder/'completed_task_release.json';r=read(release);assert r['completed_task_released']and r['completed_current_scene_released']and r['model_parameter_pointers_and_RNG_unchanged']
  for p in(initial,release):inputs[str(p)]=sha(p)
 cached=list((root/'cached_q0_revalidation').glob('*.json'));assert len(cached)==spec['cases_per_batch']-expected
 regenerated=set()
 for p in cached:
  row=read(p);assert row['old_physics_RNG_command_exact']and row['semantic_input_from_actual_current_scene']and row['native_action_count']==0 and row['no_extra_policy_queries_or_physics']and row['no_old_masks_or_policy_outputs_or_labels_imported']
  assert row['requires_whole_fresh_full_control']==(not row['old_RGB_exact'])
  if row['requires_whole_fresh_full_control']:regenerated.add(row['case_id'])
  folder=root/'cases'/row['case_id']/'M_event_full';initial=folder.parent/'M_event_full_initialization_integrity.json';a=read(initial);assert a['actual_current_cached_q0']and a['actual_policy_entry_independent_RGB_command_check_completed']and a['old_physics_RNG_commands_exact']
  release=folder/'completed_task_release.json';r=read(release);assert r['completed_task_released']and r['completed_current_scene_released']and r['model_parameter_pointers_and_RNG_unchanged']
  for f in(p,initial,release):inputs[str(f)]=sha(f)
 controls=list((root/'cases').glob('*/q0_handoff_whole_reference_audit.json'))
 cold_controls=[p for p in controls if p.parent.name not in regenerated]
 assert len(cold_controls)==len({r['case_id'].split(':')[0]for r in rows})
 assert {p.parent.name for p in controls}=={p.parent.name for p in cold_controls}|regenerated
 assert len(controls)==len(cold_controls)+len(regenerated)
 for p in controls:
  value=read(p);assert value['passed']and value['whole_native_episode_exact']and value['engineering_only_excluded_from_training'];inputs[str(p)]=sha(p);inputs.update(value['input_sha256'])
 value=dict(passed=True,transferred_q0_tasks=len(rows),cached_revalidated_cases=len(cached),regenerated_cached_cases=sorted(regenerated),cached_regeneration_whole_controls=len(regenerated),all_cached_policy_entries_byte_exact=True,all_closed_scene_and_task_and_proxy_weakrefs_dead=True,one_original_shared_pair_preserved=True,original_model_and_RNG_preserved=True,closed_after_full_not_after_screen=True,all_obsolete_MPLIB_scenes_released=True,whole_native_reference_controls=len(controls),input_sha256=inputs)
 global_write(root/'CPU_closed_expert_task_release_audit.json',value);return value
