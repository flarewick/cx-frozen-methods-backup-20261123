from native_prepared_FIT_collection_v8.scope import transform as previous_transform,restore_runtime as previous_restore
PRIORITY="    from native_policy_query_priority_v1.priority import configure as configure_raw_priority\n    configure_raw_priority(raw_observe,branch=='M_event_full')\n"
PLAN="  latest=T/'outputs/native_onpolicy_full_ops_v2/ARM53_actual_resource_result_v1.json';lp=read(latest);verify(lp);assert lp['release']['own_gpu_released']and lp['cumulative_stage_GPU_seconds']==1938.239\n  if split=='fit'and number==0:v['inherited_stage_GPU_seconds']=lp['cumulative_stage_GPU_seconds']\n  priority_CPU=T/'outputs/native_policy_query_priority_v1/CPU_contract_checks_v1.json';pc=read(priority_CPU);verify(pc);assert pc['passed']and pc['four_cases60_original_calls_all_arrays_and_final_RNG_byte_exact']and pc['reference_original_priority_retained']\n  priority_exit=T/'outputs/native_policy_query_priority_v1/CPU_actual_host_exit_v1.json';pe=read(priority_exit);assert pe['exit_code']==0 and pe['actual_Popen_wait_return']and pe['proc_absent_after_wait']\n  v['primary_raw_observation_priority2_original_model_feedback_priority0']=True\n  v['input_sha256'].update({str(p):sha(p)for p in[latest,priority_CPU,priority_exit]})\n  v['source_sha256'].update({str(p):sha(p)for folder in['native_policy_query_priority_v1','native_prepared_FIT_collection_v8','native_prepared_FIT_collection_v9']for p in(T/folder).glob('*')if p.is_file()})\n"
def transform(name,text):
 text=previous_transform(name,text).replace('native_onpolicy_arm_collection_v53','native_onpolicy_arm_collection_v54').replace('native_onpolicy_calibration_data_v36','native_onpolicy_calibration_data_v37').replace('native_prepared_FIT_collection_v8','native_prepared_FIT_collection_v9')
 if name=='runtime.py':
  anchor='    raw=lambda:c.call(lane,raw_observe)\n';assert text.count(anchor)==1;text=text.replace(anchor,PRIORITY+anchor)
 if name=='collection_plan.py':
  anchor='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
 return text
def restore_runtime(text):
 assert text.count(PRIORITY)==1
 return previous_restore(text.replace(PRIORITY,''))
