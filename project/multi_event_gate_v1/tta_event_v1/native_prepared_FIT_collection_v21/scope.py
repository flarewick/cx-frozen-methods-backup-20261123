from native_prepared_FIT_collection_v20.scope import transform as previous_transform,restore_runtime as previous_restore
M={'native_onpolicy_arm_collection_v65': 'native_onpolicy_arm_collection_v66', 'native_prepared_FIT_collection_v20': 'native_prepared_FIT_collection_v21', 'native_onpolicy_calibration_data_v48': 'native_onpolicy_calibration_data_v49'}
PLAN="  files=[T/'outputs/native_released_context_cache_v1/CPU_contract_checks_v1.json',T/'outputs/native_released_context_cache_v1/CPU_actual_exit_v1.json',T/'outputs/native_released_context_cache_v1/catalog_v1.json',T/'outputs/native_released_context_cache_v1/CPU_restore_contract_v1.json',T/'outputs/native_released_context_cache_v1/CPU_restore_actual_exit_v1.json']\n  for file in files:\n   x=read(file);verify(x)\n   if 'exit_code'in x:assert x['exit_code']==0 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']\n   else:assert x['passed']\n  v['same_workload_actual_released_context_cache_only_no_NN_import']=True\n  v['input_sha256'].update({str(p):sha(p)for p in files})\n  catalog=read(T/'outputs/native_released_context_cache_v1/catalog_v1.json')\n  for row in catalog['devices'].values():\n   file=mapped(row['manifest']);value=read(file);verify(value);v['input_sha256'][str(file)]=sha(file);v['input_sha256'].update(value['input_sha256'])\n  v['source_sha256'].update({str(p):sha(p)for p in(T/'native_released_context_cache_v1').glob('*')if p.is_file()})\n"
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text

def transform(name,text):
 text=rename(previous_transform(name,text))
 if name=='host.py':
  text=text.replace('from native_same_device_driver_cache_v1.cache import candidates as cache_candidates','from native_released_context_cache_v1.cache import available as cache_candidates').replace('from native_same_device_driver_cache_v1.cache import restore as restore_same_device_cache','from native_released_context_cache_v1.cache import restore as restore_same_device_cache')
 if name=='collection_plan.py':
  text=text.replace('ARM63_actual_resource_result_v1.json','ARM65_actual_resource_result_v1.json').replace("nv['cumulative_stage_GPU_seconds']==4218.938","nv['cumulative_stage_GPU_seconds']==4501.184")
  anchor='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
 return text.replace('outputs/native_onpolicy_arm_collection_v66/CPU_contract_checks_v2.json','outputs/native_onpolicy_arm_collection_v66/CPU_contract_checks_v2.json').replace('outputs/native_onpolicy_arm_collection_v66/CPU_actual_host_exit_v2.json','outputs/native_onpolicy_arm_collection_v66/CPU_actual_host_exit_v2.json')

def restore_runtime(text):return previous_restore(text)
