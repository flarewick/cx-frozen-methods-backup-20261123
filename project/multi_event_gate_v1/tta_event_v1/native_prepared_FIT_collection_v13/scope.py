from native_prepared_FIT_collection_v11.scope import transform as previous_transform,restore_runtime as previous_restore
M={'native_onpolicy_arm_collection_v56': 'native_onpolicy_arm_collection_v58', 'native_onpolicy_calibration_data_v39': 'native_onpolicy_calibration_data_v41', 'native_prepared_FIT_collection_v11': 'native_prepared_FIT_collection_v13', 'native_render_overlap_v3': 'native_render_overlap_v4', 'outputs/native_render_overlap_v4/CPU_contract_checks_v2.json': 'outputs/native_render_overlap_v4/CPU_contract_checks_v1.json', 'outputs/native_render_overlap_v4/CPU_actual_host_exit_v2.json': 'outputs/native_render_overlap_v4/CPU_queue_actual_host_exit_v1.json'}
PLAN="  newest=T/'outputs/native_onpolicy_full_ops_v2/ARM57_actual_resource_result_v1.json';nv=read(newest);verify(nv);assert nv['release']['own_gpu_released']and nv['cumulative_stage_GPU_seconds']==3072.234\n  if split=='fit'and number==0:v['inherited_stage_GPU_seconds']=nv['cumulative_stage_GPU_seconds']\n  proofs=[T/'outputs/native_bounded_object_state_v1/CPU_contract_checks_v1.json',T/'outputs/native_bounded_object_state_v1/CPU_actual_host_exit_v1.json',T/'outputs/native_render_overlap_v4/CPU_capture_wait_checks_v1.json',T/'outputs/native_render_overlap_v4/CPU_capture_actual_host_exit_v1.json']\n  from native_direct_frames_v2.verify import verify_proof\n  for file in proofs:\n   x=read(file);assert verify_proof(x)\n   if 'exit_code'in x:assert x['exit_code']==0 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']\n   else:assert x['passed']\n  v['bounded_original_object_state_spatial_storage_only']=True\n  v['active_object_state_storage']='original_GPU_recent7_and_all_conditional_CPU_older_spatial'\n  v['legacy_CPU_storage_flag_denotes_inherited_whole25_raw_contract_not_active_all_CPU_storage']=True\n  v['partial_render_progress_never_qualification']=True\n  v['input_sha256'].update({str(p):sha(p)for p in[newest,*proofs]})\n  v['source_sha256'].update({str(p):sha(p)for folder in['native_bounded_object_state_v1','native_render_overlap_v4','native_prepared_FIT_collection_v13']for p in(T/folder).glob('*')if p.is_file()})\n"
OLD=' render_overlap=NativeOverlap(c,torch,c.native,lambda:denoise_graph.entry is not None,control_codes=original_controls(execute,check_precision.__code__))\n'
NEW=" from .common import replace as replace_progress\n render_overlap=NativeOverlap(c,torch,c.native,lambda:denoise_graph.entry is not None,control_codes=original_controls(execute,check_precision.__code__),on_progress=lambda value:replace_progress(root/'render_overlap_progress.json',value))\n"
OLD_SOURCE='       from native_object_state_storage_v2.core import CPUStateSource\n       selected_source=CPUStateSource\n'
NEW_SOURCE='       from native_bounded_object_state_v1.core import BoundedSource\n       selected_source=BoundedSource\n'
OLD_TRACK='      selected_source=tracked(selected_source,branchfolder)\n'
NEW_TRACK="      if branch=='M_event_full'and seed_report['feedback_interface']=='native_object_association_v1':\n       from native_bounded_object_state_v1.audit import tracked as tracked_bounded\n       selected_source=tracked_bounded(selected_source,branchfolder)\n      selected_source=tracked(selected_source,branchfolder)\n"
HOST_ANCHOR="        write(root / 'atomic_full_recovery_qualification.json',"
HOST_EXTRA="        from native_bounded_object_state_v1.audit import audit as audit_bounded_state\n        write(root/'CPU_bounded_state_audit.json',audit_bounded_state(root,spec))\n"
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text

def transform(name,text):
 text=rename(previous_transform(name,text))
 if name=='runtime.py':
  for old,new in [(OLD,NEW),(OLD_SOURCE,NEW_SOURCE),(OLD_TRACK,NEW_TRACK)]:
   assert text.count(old)==1;text=text.replace(old,new)
 if name=='collection_plan.py':
  anchor='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n'
  assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
 if name=='host.py':
  assert text.count(HOST_ANCHOR)==1;text=text.replace(HOST_ANCHOR,HOST_EXTRA+HOST_ANCHOR)
  text=text.replace("for p in (root/'CPU_native_image_readback_audit.json',", "for p in (root/'CPU_bounded_state_audit.json',root/'CPU_native_image_readback_audit.json',")
 return text

def restore_runtime(text):
 for old,new in [(OLD,NEW),(OLD_SOURCE,NEW_SOURCE),(OLD_TRACK,NEW_TRACK)]:
  assert text.count(new)==1;text=text.replace(new,old)
 return previous_restore(text.replace('native_render_overlap_v4','native_render_overlap_v3'))
