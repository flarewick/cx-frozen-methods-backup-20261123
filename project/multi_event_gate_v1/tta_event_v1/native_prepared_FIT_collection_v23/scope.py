from native_prepared_FIT_collection_v21.scope import transform as previous_transform,restore_runtime as previous_restore
M={'native_onpolicy_arm_collection_v66': 'native_onpolicy_arm_collection_v68', 'native_prepared_FIT_collection_v21': 'native_prepared_FIT_collection_v23', 'native_onpolicy_calibration_data_v49': 'native_onpolicy_calibration_data_v51'}
PLAN="  v['eight_lane_with_qualified_original_native_readback']=True\n  v['simulation_lanes']=8;v['native_batched_readback_12_lanes']=False\n  v['same_fixed25_and_case_seeds_and_original_full_references']=True\n  files=[T/'outputs/native_onpolicy_arm_collection_v68/CPU_full_model_checks_v2.json',T/'outputs/native_onpolicy_arm_collection_v68/CPU_full_actual_exit_v2.json']\n  for file in files:\n   x=read(file);verify(x)\n   if 'exit_code'in x:assert x['exit_code']==0 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']\n   else:assert x['passed']and x['actual_CPU_heads']==8\n  v['input_sha256'].update({str(p):sha(p)for p in files})\n"
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text

def transform(name,text):
 text=rename(previous_transform(name,text))
 if name=='collection_plan.py':
  text=text.replace('ARM65_actual_resource_result_v1.json','ARM67_actual_resource_result_v1.json').replace("nv['cumulative_stage_GPU_seconds']==4501.184","nv['cumulative_stage_GPU_seconds']==5069.813")
  anchor='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
 if name=='dispatch_checks.py':
  text=text.replace('range(12)','range(8)').replace('done[0]<12','done[0]<8').replace('native_lanes_fixture=12','native_lanes_fixture=8')
 return text.replace('outputs/native_onpolicy_arm_collection_v68/CPU_contract_checks_v2.json','outputs/native_onpolicy_arm_collection_v68/CPU_contract_checks_v2.json').replace('outputs/native_onpolicy_arm_collection_v68/CPU_actual_host_exit_v2.json','outputs/native_onpolicy_arm_collection_v68/CPU_actual_host_exit_v2.json')

def restore_runtime(text):return previous_restore(text)
