from native_prepared_FIT_collection_v19.scope import transform as previous_transform,restore_runtime as previous_restore
M={'native_onpolicy_arm_collection_v64': 'native_onpolicy_arm_collection_v65', 'native_prepared_FIT_collection_v19': 'native_prepared_FIT_collection_v20', 'native_onpolicy_calibration_data_v47': 'native_onpolicy_calibration_data_v48'}
PLAN="  files=[T/'outputs/native_measured_policy_control_v2/CPU_contract_checks_v3.json',T/'outputs/native_measured_policy_control_v2/CPU_actual_exit_v3.json',T/'outputs/native_onpolicy_full_ops_v2/ARM64_CPU_only_cancel_v1.json']\n  for file in files:\n   x=read(file);verify(x)\n   if 'exit_code'in x:assert x['exit_code']==0 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']\n   else:assert x['passed']\n  v['original_randomized_light_and_unsupported_control_fallbacks_preserved']=True\n  v['input_sha256'].update({str(p):sha(p)for p in files})\n  v['source_sha256'].update({str(p):sha(p)for p in(T/'native_measured_policy_control_v2').glob('*')if p.is_file()})\n"
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text

def transform(name,text):
 text=rename(previous_transform(name,text))
 if name in('runtime.py','preflight.py','host.py'):
  text=text.replace('from native_measured_policy_control_v1.runtime','from native_measured_policy_control_v2.runtime').replace('from native_measured_policy_control_v1.audit','from native_measured_policy_control_v2.audit')
 if name=='collection_plan.py':
  anchor='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
 return text.replace('outputs/native_onpolicy_arm_collection_v65/CPU_contract_checks_v2.json','outputs/native_onpolicy_arm_collection_v65/CPU_contract_checks_v2.json').replace('outputs/native_onpolicy_arm_collection_v65/CPU_actual_host_exit_v2.json','outputs/native_onpolicy_arm_collection_v65/CPU_actual_host_exit_v2.json')

def restore_runtime(text):return previous_restore(text.replace('from native_measured_policy_control_v2.runtime','from native_measured_policy_control_v1.runtime'))
