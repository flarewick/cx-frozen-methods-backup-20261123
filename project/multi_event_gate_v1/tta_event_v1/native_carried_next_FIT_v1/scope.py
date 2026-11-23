RUNTIME="   from native_carried_next_FIT_v1.carry import install as install_carried_next\n   install_carried_next(private,spec)\n"
HOST="        from native_carried_next_FIT_v1.carry import audit as audit_carried_next\n        audit_carried_next(root,spec)\n"
PLAN="""  failure34=T/'outputs/native_onpolicy_full_ops_v2/ARM34_actual_resource_result_v1.json';prior=read(failure34);verify(prior)
  assert prior['release']['own_gpu_released']and not prior['release']['remaining_own_gpu_pids']
  if split=='val':v['inherited_stage_GPU_seconds']=prior['cumulative_VAL_GPU_seconds']
  if split=='val':v['cumulative_budget_amendment']['all_failed_seconds_retained']=v['inherited_stage_GPU_seconds']
  v['future_preparation_each_task']=1 if split=='val' or number==0 else 0
  v['carried_next_FIT0_scope']=True
  v['carried_FIT0_input_proof']=str(T/'outputs/native_fixed_FIT_inputs_v1/verified_old_FIT0_prefix_v1.json')if split=='val'else None
  if split=='val':
   from native_carried_next_FIT_v1.carry import proof as carried_proof
   carried_file,carried=carried_proof(v);v['input_sha256'][str(carried_file)]=sha(carried_file);v['input_sha256'].update(carried['input_sha256'])
  v['input_sha256'][str(failure34)]=sha(failure34)
  v['source_sha256'].update({str(p):sha(p)for p in(T/'native_carried_next_FIT_v1').glob('*')if p.is_file()})
"""
def restore_runtime(text):
 assert text.count(RUNTIME)==1;return text.replace(RUNTIME,'')
def transformed(name,text):
 text=text.replace('native_onpolicy_arm_collection_v34','native_onpolicy_arm_collection_v35')
 if name=='runtime.py':
  anchor="   prepared['fixed_next_schedule']=private\n";assert text.count(anchor)==1
  # Keep the indentation of the original future-worker's try block.
  text=text.replace(anchor,anchor+RUNTIME)
 if name=='host.py':
  anchor='        audit_storage(root,spec)\n';assert text.count(anchor)==1;text=text.replace(anchor,anchor+HOST)
  # FIT has no carried prefix. Its ordinary original future audit remains mandatory.
  text=text.replace("root/'CPU_storage_whole25_audit.json')","root/'CPU_storage_whole25_audit.json',*([root/'CPU_carried_FIT0_inputs_audit.json']if spec.get('carried_FIT0_input_proof')else[]))")
 if name=='collection_plan.py':
  anchor='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
  text=text.replace('native_onpolicy_calibration_data_v26.reader','native_onpolicy_calibration_data_v28.reader')
  # The current proof is CPU2, matching the inherited number after source mapping.
 return text
