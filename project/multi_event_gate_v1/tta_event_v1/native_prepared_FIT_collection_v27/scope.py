from native_prepared_FIT_collection_v26.scope import transform as previous_transform,restore_runtime as previous_restore
M={'native_onpolicy_arm_collection_v71': 'native_onpolicy_arm_collection_v72', 'native_prepared_FIT_collection_v26': 'native_prepared_FIT_collection_v27', 'native_onpolicy_calibration_data_v54': 'native_onpolicy_calibration_data_v55'}
PLAN="  for file in[T/'outputs/native_FIT0_budget10800_v1/CPU_contract_checks_v1.json',T/'outputs/native_onpolicy_full_ops_v2/CPU_FIT0_10800_v1_actual_exit.json']:\n   x=read(file);verify(x)\n   if 'exit_code'in x:assert x['exit_code']==0 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']\n   else:assert x['passed']\n   v['input_sha256'][str(file)]=sha(file)\n  from native_FIT0_budget10800_v1.budget import apply as apply_explicit_budget\n  apply_explicit_budget(v)\n"
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text

def transform(name,text):
 text=rename(previous_transform(name,text))
 if name=='collection_plan.py':
  a='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(a)==1;text=text.replace(a,PLAN+a)
 return text.replace('outputs/native_onpolicy_arm_collection_v72/CPU_contract_checks_v2.json','outputs/native_onpolicy_arm_collection_v72/CPU_contract_checks_v2.json').replace('outputs/native_onpolicy_arm_collection_v72/CPU_actual_host_exit_v2.json','outputs/native_onpolicy_arm_collection_v72/CPU_actual_host_exit_v2.json')

def restore_runtime(text):return previous_restore(text)
