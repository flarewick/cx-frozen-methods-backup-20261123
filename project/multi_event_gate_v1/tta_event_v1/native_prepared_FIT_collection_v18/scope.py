from native_prepared_FIT_collection_v17.scope import transform as previous_transform,restore_runtime as previous_restore
M={'native_onpolicy_arm_collection_v62': 'native_onpolicy_arm_collection_v63', 'native_prepared_FIT_collection_v17': 'native_prepared_FIT_collection_v18', 'native_onpolicy_calibration_data_v45': 'native_onpolicy_calibration_data_v46'}
PLAN="  files=[T/'outputs/native_same_device_driver_cache_v1/CPU_contract_checks_v2.json',T/'outputs/native_same_device_driver_cache_v1/CPU_actual_host_exit_v2.json']\n  for file in files:\n   x=read(file);verify(x)\n   if 'exit_code'in x:assert x['exit_code']==0 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']\n   else:assert x['passed']and x['actual_original_seed_and_exact_file_readback']\n  v['original_same_physical_device_context_cache_seed_before_CUDA']=True\n  v['parent_compiled_sampler_NOT_imported']=True\n  v['input_sha256'].update({str(p):sha(p)for p in files})\n  v['source_sha256'].update({str(p):sha(p)for p in(T/'native_same_device_driver_cache_v1').glob('*')if p.is_file()})\n"
INSERT="  from native_same_device_driver_cache_v1.cache import restore as restore_same_device_cache\n  restore_same_device_cache(root,info['uuid'])\n"
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text

def transform(name,text):
 text=rename(previous_transform(name,text))
 if name=='collection_plan.py':
  text=text.replace("v['preferred_GPU_indices']=[1,6,5,4,2,3]","v['preferred_GPU_indices']=[5,1,4,2,3,6]")
  text=text.replace('ARM61_actual_resource_result_v1.json','ARM62_actual_resource_result_v1.json').replace("nv['cumulative_stage_GPU_seconds']==3650.615","nv['cumulative_stage_GPU_seconds']==3935.162")
  anchor='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
 if name=='host.py':
  text=text.replace('for gpu in (1,6,5,4,2,3):','for gpu in (5,1,4,2,3,6):')
  point="    admission.require_admission(fleet,info['uuid'])\n"
  extra="    from native_same_device_driver_cache_v1.cache import candidates as cache_candidates\n    try:cache_candidates(container.HERE,info['uuid'],gpu_doctor.query(['-i',str(gpu),'--query-gpu=driver_version']).strip())\n    except AssertionError:continue\n"
  assert text.count(point)==1;text=text.replace(point,extra+point)
  anchor="  lockdir=container.WORK/'gpu_locks';lockdir.mkdir(exist_ok=True);waiting=time.monotonic()\n";assert text.count(anchor)==1;text=text.replace(anchor,INSERT+anchor)
  anchor="for p in (root/'CPU_guide_reuse_audit.json',";assert text.count(anchor)==1;text=text.replace(anchor,"for p in (root/'CPU_same_device_driver_cache_seed.json',root/'CPU_guide_reuse_audit.json',")
 return text.replace('outputs/native_onpolicy_arm_collection_v63/CPU_contract_checks_v3.json','outputs/native_onpolicy_arm_collection_v63/CPU_contract_checks_v2.json').replace('outputs/native_onpolicy_arm_collection_v63/CPU_actual_host_exit_v3.json','outputs/native_onpolicy_arm_collection_v63/CPU_actual_host_exit_v2.json')

def restore_runtime(text):return previous_restore(text)
