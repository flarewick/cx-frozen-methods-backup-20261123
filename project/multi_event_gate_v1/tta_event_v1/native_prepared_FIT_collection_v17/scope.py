from native_prepared_FIT_collection_v16.scope import transform as previous_transform,restore_runtime as previous_restore
M={'native_onpolicy_arm_collection_v61': 'native_onpolicy_arm_collection_v62', 'native_prepared_FIT_collection_v16': 'native_prepared_FIT_collection_v17', 'native_onpolicy_calibration_data_v44': 'native_onpolicy_calibration_data_v45'}
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text

def transform(name,text):
 text=rename(previous_transform(name,text))
 if name=='python_scheduling.py':
  old='sys.setswitchinterval(INTERVAL_SECONDS);assert sys.getswitchinterval()==INTERVAL_SECONDS'
  new='sys.setswitchinterval(INTERVAL_SECONDS);actual=sys.getswitchinterval();assert abs(actual-INTERVAL_SECONDS)<=1e-12'
  assert text.count(old)==1;text=text.replace(old,new).replace('actual_switch_interval_seconds=INTERVAL_SECONDS','requested_switch_interval_seconds=INTERVAL_SECONDS,actual_switch_interval_seconds=actual')
 if name=='collection_plan.py':
  text=text.replace('ARM59_actual_resource_result_v1.json','ARM61_actual_resource_result_v1.json').replace("nv['cumulative_stage_GPU_seconds']==3641.891","nv['cumulative_stage_GPU_seconds']==3650.615")
 return text.replace('outputs/native_onpolicy_arm_collection_v62/CPU_contract_checks_v2.json','outputs/native_onpolicy_arm_collection_v62/CPU_contract_checks_v3.json').replace('outputs/native_onpolicy_arm_collection_v62/CPU_actual_host_exit_v2.json','outputs/native_onpolicy_arm_collection_v62/CPU_actual_host_exit_v3.json')

def restore_runtime(text):return previous_restore(text)
