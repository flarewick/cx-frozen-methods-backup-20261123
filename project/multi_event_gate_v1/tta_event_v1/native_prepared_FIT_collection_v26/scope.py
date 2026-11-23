from native_prepared_FIT_collection_v24.scope import transform as previous_transform,restore_runtime as previous_restore
M={'native_onpolicy_arm_collection_v69': 'native_onpolicy_arm_collection_v71', 'native_prepared_FIT_collection_v24': 'native_prepared_FIT_collection_v26', 'native_onpolicy_calibration_data_v52': 'native_onpolicy_calibration_data_v54'}
OPEN=' from native_owner_profile_v2.profile import OwnerProfile\n owner_profile=OwnerProfile(c.service)\n'
CLOSE="  write(root/'owner_profile_runtime.json',owner_profile.close())\n"
PLAN="  v['governor_extra_margin_required']=False\n  v['documented_server_policy_user34_without_extra35_36']=True\n  v['owner_profile_observes_existing_calls_without_GPU_sync']=True\n  files=[T/'outputs/native_written_server_policy_v1/CPU_v3/CPU_contract_checks_v1.json',T/'outputs/native_onpolicy_full_ops_v2/CPU_written_policy_v3_actual_exit.json',T/'outputs/native_written_server_policy_v1/user_authority_20261107.json',T/'outputs/native_written_server_policy_v1/server_rules_snapshot_20261107.txt',T/'outputs/native_owner_profile_v2/CPU_v1/CPU_contract_checks_v1.json',T/'outputs/native_onpolicy_full_ops_v2/CPU_owner_profile_v3_actual_exit.json']\n  for file in[files[0],files[1],files[4],files[5]]:\n   x=read(file);verify(x)\n   if 'exit_code'in x:assert x['exit_code']==0 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']\n   else:assert x['passed']\n  v['input_sha256'].update({str(p):sha(p)for p in files})\n  for folder in[T/'native_written_server_policy_v1',T/'native_owner_profile_v2']:\n   v['source_sha256'].update({str(p):sha(p)for p in folder.glob('*')if p.is_file()})\n"
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text

def transform(name,text):
 text=rename(previous_transform(name,text))
 if name=='runtime.py':
  text=text.replace("on_progress=lambda value:replace_progress(root/'render_overlap_progress.json',value)","on_progress=lambda value:(replace_progress(root/'render_overlap_progress.json',value),replace_progress(root/'owner_profile_progress.json',owner_profile.snapshot()))")
  a=' c.native=render_overlap\n';assert text.count(a)==1;text=text.replace(a,a+OPEN)
  a='  robot.CuroboPlanner=original_planner;c.rng.uninstall()\n';assert text.count(a)==1;text=text.replace(a,a+CLOSE)
 if name=='host.py':
  text=text.replace('from native_governor_margin_v2 import margin as guard','from native_process_missing_guard_v2 import guard as guard').replace('from native_governor_margin_v2.audit import audit as audit_governor_margin','from native_written_server_policy_v1.audit import audit as audit_governor_margin').replace('CPU_governor_margin_audit.json','CPU_written_policy_audit.json')
 if name=='collection_plan.py':
  text=text.replace('ARM67_actual_resource_result_v1.json','ARM70_actual_resource_result_v1.json').replace("nv['cumulative_stage_GPU_seconds']==5069.813","nv['cumulative_stage_GPU_seconds']==5719.868")
  a='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(a)==1;text=text.replace(a,PLAN+a)
 return text.replace('outputs/native_onpolicy_arm_collection_v71/CPU_contract_checks_v1.json','outputs/native_onpolicy_arm_collection_v71/CPU_contract_checks_v2.json').replace('outputs/native_onpolicy_arm_collection_v71/CPU_actual_host_exit_v1.json','outputs/native_onpolicy_arm_collection_v71/CPU_actual_host_exit_v2.json')

def restore_runtime(text):
 assert text.count(OPEN)==1 and text.count(CLOSE)==1
 text=text.replace("on_progress=lambda value:(replace_progress(root/'render_overlap_progress.json',value),replace_progress(root/'owner_profile_progress.json',owner_profile.snapshot()))","on_progress=lambda value:replace_progress(root/'render_overlap_progress.json',value)")
 return previous_restore(text.replace(OPEN,'').replace(CLOSE,''))
