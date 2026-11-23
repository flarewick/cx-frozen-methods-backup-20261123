from native_prepared_FIT_collection_v23.scope import transform as previous_transform,restore_runtime as previous_restore
M={'native_onpolicy_arm_collection_v68': 'native_onpolicy_arm_collection_v69', 'native_prepared_FIT_collection_v23': 'native_prepared_FIT_collection_v24', 'native_onpolicy_calibration_data_v51': 'native_onpolicy_calibration_data_v52'}
PLAN="  files=[T/'outputs/native_released_context_cache_v2/CPU_contract_checks_v1.json',T/'outputs/native_released_context_cache_v2/CPU_actual_exit_v1.json',T/'outputs/native_released_context_cache_v2/catalog_v1.json',T/'outputs/native_onpolicy_arm_collection_v68/CPU_full_model_checks_v2.json',T/'outputs/native_onpolicy_arm_collection_v68/CPU_full_actual_exit_v2.json']\n  for file in files:\n   x=read(file);verify(x)\n   if 'exit_code'in x:assert x['exit_code']==0 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']\n   else:assert x['passed']\n  v['actual_same_device_context_catalog_supports_GPU1_and_GPU5']=True\n  v['CPU_only_ARM68_wait_cancelled_without_GPU_or_case_work']=True\n  v['input_sha256'].update({str(p):sha(p)for p in files})\n  for row in read(files[2])['devices'].values():\n   p=mapped(row['manifest']);value=read(p);verify(value);v['input_sha256'][str(p)]=sha(p);v['input_sha256'].update(value['input_sha256'])\n  p=T/'outputs/native_onpolicy_arm_collection_v68/fit_000/actual_host_wait_exit.json';x=read(p);assert x['exit_code']==-2 and x['actual_Popen_wait_return']and x['proc_absent_after_wait'];v['input_sha256'][str(p)]=sha(p)\n  v['source_sha256'].update({str(p):sha(p)for p in(T/'native_released_context_cache_v2').glob('*')if p.is_file()})\n"
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text

def transform(name,text):
 text=rename(previous_transform(name,text))
 if name=='host.py':text=text.replace('from native_released_context_cache_v1.cache','from native_released_context_cache_v2.cache')
 if name=='collection_plan.py':
  text=text.replace('outputs/native_onpolicy_arm_collection_v69/CPU_full_model_checks_v2.json','outputs/native_onpolicy_arm_collection_v68/CPU_full_model_checks_v2.json').replace('outputs/native_onpolicy_arm_collection_v69/CPU_full_actual_exit_v2.json','outputs/native_onpolicy_arm_collection_v68/CPU_full_actual_exit_v2.json')
  anchor='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
 return text.replace('outputs/native_onpolicy_arm_collection_v69/CPU_contract_checks_v2.json','outputs/native_onpolicy_arm_collection_v69/CPU_contract_checks_v1.json').replace('outputs/native_onpolicy_arm_collection_v69/CPU_actual_host_exit_v2.json','outputs/native_onpolicy_arm_collection_v69/CPU_actual_host_exit_v1.json')

def restore_runtime(text):return previous_restore(text)
