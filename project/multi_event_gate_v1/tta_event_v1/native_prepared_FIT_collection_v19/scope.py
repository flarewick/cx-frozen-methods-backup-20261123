from native_prepared_FIT_collection_v18.scope import transform as previous_transform,restore_runtime as previous_restore
M={'native_onpolicy_arm_collection_v63': 'native_onpolicy_arm_collection_v64', 'native_prepared_FIT_collection_v18': 'native_prepared_FIT_collection_v19', 'native_onpolicy_calibration_data_v46': 'native_onpolicy_calibration_data_v47'}
PRE="  from native_measured_policy_control_v1.runtime import load as load_measured_control\n  value['measured_control_native']=load_measured_control()\n"
INSTALL="     from native_measured_policy_control_v1.runtime import install as install_measured_control\n     measured_usage=install_measured_control(task,c.scene_gate,prepared['measured_control_native'])if branch=='M_event_full'else None\n"
REPORT="     if measured_usage is not None:write(branchfolder/'measured_control_usage.json',measured_usage())\n"
HOST="        from native_measured_policy_control_v1.audit import audit as audit_measured_control\n        write(root/'CPU_measured_control_audit.json',audit_measured_control(root,spec))\n"
PLAN="  files=[T/'outputs/native_measured_policy_control_v1/CPU_contract_checks_v5.json',T/'outputs/native_measured_policy_control_v1/CPU_actual_exit_v5.json',T/'outputs/native_measured_policy_control_v1/CPU_build_v1.json',T/'outputs/native_measured_policy_control_v1/CPU_build_actual_exit_v1.json',T/'outputs/native_measured_policy_control_v1/CPU_full_model_checks_v1.json',T/'outputs/native_measured_policy_control_v1/CPU_full_actual_exit_v1.json']\n  for file in files:\n   x=read(file);verify(x)\n   if 'exit_code'in x:assert x['exit_code']==0 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']\n   else:assert x['passed']\n  v['original_measured_physics_clock_trace_callbacks_in_native_policy_control']=True\n  v['input_sha256'].update({str(p):sha(p)for p in files})\n  v['source_sha256'].update({str(p):sha(p)for p in(T/'native_measured_policy_control_v1').glob('*')if p.is_file()})\n  b=read(T/'outputs/native_measured_policy_control_v1/CPU_build_v1.json');file=mapped(b['binary']);assert sha(file)==b['binary_sha256'];v['input_sha256'][str(file)]=sha(file)\n"
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text

def transform(name,text):
 text=rename(previous_transform(name,text))
 if name=='runtime.py':
  anchor="     runner=episode if deferred_semantics else episode._q0_original\n";assert text.count(anchor)==1;text=text.replace(anchor,INSTALL+anchor)
  anchor='     if cached_revalidated:\n';assert text.count(anchor)==1;text=text.replace(anchor,REPORT+anchor)
 if name=='preflight.py':
  anchor='  return value\n';assert text.count(anchor)==1;text=text.replace(anchor,PRE+anchor)
 if name=='collection_plan.py':
  text=text.replace('ARM62_actual_resource_result_v1.json','ARM63_actual_resource_result_v1.json').replace("nv['cumulative_stage_GPU_seconds']==3935.162","nv['cumulative_stage_GPU_seconds']==4218.938")
  anchor='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
 if name=='host.py':
  anchor="        write(root / 'atomic_full_recovery_qualification.json',";assert text.count(anchor)==1;text=text.replace(anchor,HOST+anchor)
  anchor="for p in (root/'CPU_same_device_driver_cache_seed.json',";assert text.count(anchor)==1;text=text.replace(anchor,"for p in (root/'CPU_measured_control_audit.json',root/'CPU_same_device_driver_cache_seed.json',")
 return text.replace('outputs/native_onpolicy_arm_collection_v64/CPU_contract_checks_v2.json','outputs/native_onpolicy_arm_collection_v64/CPU_contract_checks_v2.json').replace('outputs/native_onpolicy_arm_collection_v64/CPU_actual_host_exit_v2.json','outputs/native_onpolicy_arm_collection_v64/CPU_actual_host_exit_v2.json')

def restore_runtime(text):
 for value in(INSTALL,REPORT):assert text.count(value)==1;text=text.replace(value,'')
 return previous_restore(text)
