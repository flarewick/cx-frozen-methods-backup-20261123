PLAN="""  failure33=T/'outputs/native_onpolicy_full_ops_v2/ARM33_actual_first280_failure_v1.json';prior=read(failure33);verify(prior)
  assert prior['release']['own_gpu_released']and not prior['release']['remaining_own_gpu_pids']
  if split=='val':v['inherited_stage_GPU_seconds']=prior['cumulative_VAL_GPU_seconds']
  v['task_interleaved_original_cached_prefix_order']=True
  v['input_sha256'][str(failure33)]=sha(failure33)
  v['source_sha256'].update({str(p):sha(p)for p in(T/'native_balanced_task_schedule_v1').glob('*')if p.is_file()})
"""
OLD="self.order=sorted(range(count),key=lambda index:(index>=self.prefix,-SCHEDULING_CAPS.get(tasks[index%len(tasks)],0),index//len(tasks),index%len(tasks)))"
NEW="self.order=sorted(range(count),key=lambda index:(index>=self.prefix,index))"
def transformed(name,text):
 text=text.replace('native_onpolicy_arm_collection_v31','native_onpolicy_arm_collection_v34')
 if name=='work.py':assert text.count(OLD)==1;text=text.replace(OLD,NEW)
 if name=='collection_plan.py':
  anchor="  v['object_state_storage_CPU_whole_original_raw_controls']=True\n";assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
  text=text.replace('outputs/native_onpolicy_arm_collection_v34/CPU_contract_checks_v1.json','outputs/native_onpolicy_arm_collection_v34/CPU_contract_checks_v2.json').replace('outputs/native_onpolicy_arm_collection_v34/CPU_actual_host_exit_v1.json','outputs/native_onpolicy_arm_collection_v34/CPU_actual_host_exit_v2.json')
 return text
