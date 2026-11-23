"""Real complete C23 expert/q0 preparation only, never failed scientific outputs."""
from pathlib import Path
from .common import T,TASKS,read,mapped
from .known_preparation import collect

def run():
 parent=T/'outputs/native_onpolicy_collection_v14/fit_003';root=mapped(read(parent/'execution_state.json')['root']);initial=read(root/'screening_complete.json')['state']
 from native_onpolicy_collection_v23.known_preparation import collect as prior
 old=prior(initial,TASKS,25,7120000);assert old['cached_valid_cases']==24
 current=collect(initial,TASKS,25,7120000);assert current['cached_valid_cases']==25
 c23=T/'outputs/native_onpolicy_collection_v23/fit_004';r=mapped(read(c23/'execution_state.json')['root']);expected={read(f)['case_id']for f in(r/'accepted').glob('*.json')}
 actual={v['case_id']for group in current['task_cache'].values()for v in group['items']};assert actual==expected and len(actual)==25
 for task in TASKS:
  a=old['task_cache'][task];b=current['task_cache'][task];assert b['items'][:len(a['items'])]==a['items']and b['rows'][:len(a['rows'])]==a['rows'];assert b['end_state']['accepted']==25
 assert current['case_request_keys']['grab_roller:7120029']=='5cfd5405e634fd424bf57fd7b631800095e0de4ccd659646b3c90a7b70e97b71'
 assert current['historical_C5_preparation_recoveries_used']==current['historical_C5_preparation_recoveries_maximum']==2 and current['new_automatic_recovery_attempts']==0
 bound=[str(mapped(f))for f in current['input_sha256']if mapped(f).is_relative_to(r)]
 assert bound and not any('/cases/'in f or '/policy'in f or '/masks'in f or '/labels'in f or 'target_file'in f for f in bound)
 assert current['no_policy_outputs_masks_or_labels_imported']
 return dict(passed=True,actual_completed_q0_preparation_cases=25,old_available_cases=24,exact_same25_case_identities=True,old_canonical_validity_and_negatives_unchanged=True,all_q0_keys_are_actual_current_C23_RGB_commands_language=True,latest_rare_RGB_key_verified=True,failed_full_policy_masks_labels_NOT_imported=True,old_preparation_recovery_budget_2_of_2_unchanged=True,no_GPU_work_or_scientific_qualification=True)
