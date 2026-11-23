from native_prepared_FIT_collection_v7.scope import transform as original_transform,restore_runtime as original_restore,RAW
INIT=" from native_render_overlap_v1.scheduler import NativeOverlap\n render_overlap=NativeOverlap(c,torch,c.native,lambda:denoise_graph.entry is not None)\n c.native=render_overlap\n"
OLD_OBSERVE="     return observe(task)\n"
NEW_OBSERVE="     with render_overlap.active(branch=='M_event_full'):return observe(task)\n"
CLOSE="  render_overlap.close()\n"
REPORT=" write(root/'render_overlap_runtime.json',render_overlap.report())\n"
AUDIT="        from native_render_overlap_v1.audit import audit as audit_render_overlap\n        write(root/'CPU_render_overlap_audit.json',audit_render_overlap(root,spec))\n"
PLAN="  previous=T/'outputs/native_onpolicy_full_ops_v2/ARM52_actual_resource_result_v1.json';pv=read(previous);verify(pv);assert pv['release']['own_gpu_released']and pv['cumulative_stage_GPU_seconds']==1653.674\n  if split=='fit'and number==0:v['inherited_stage_GPU_seconds']=pv['cumulative_stage_GPU_seconds']\n  render_CPU=T/'outputs/native_render_overlap_v1/CPU_contract_checks_v3.json';rp=read(render_CPU);verify(rp);assert rp['passed']and rp['all_lane_arrays_and_final_RNG_byte_exact']and rp['actual_report_positive_and_wrong_owner_rejected']\n  render_exit=T/'outputs/native_render_overlap_v1/CPU_actual_host_exit_v3.json';re=read(render_exit);assert re['exit_code']==0 and re['actual_Popen_wait_return']and re['proc_absent_after_wait']\n  v['native_original_render_overlap_safe_owner_requests']=True\n  v['input_sha256'].update({str(p):sha(p)for p in[previous,render_CPU,render_exit]})\n  v['source_sha256'].update({str(p):sha(p)for folder in['native_render_overlap_v1','native_prepared_FIT_collection_v7','native_prepared_FIT_collection_v8']for p in(T/folder).glob('*')if p.is_file()})\n"

def transform(name,text):
 text=original_transform(name,text).replace('native_onpolicy_arm_collection_v52','native_onpolicy_arm_collection_v53').replace('native_onpolicy_calibration_data_v35','native_onpolicy_calibration_data_v36').replace('native_prepared_FIT_collection_v7','native_prepared_FIT_collection_v8')
 text=text.replace('outputs/native_onpolicy_arm_collection_v53/CPU_contract_checks_v2.json','outputs/native_onpolicy_arm_collection_v53/CPU_contract_checks_v1.json').replace('outputs/native_onpolicy_arm_collection_v53/CPU_actual_host_exit_v2.json','outputs/native_onpolicy_arm_collection_v53/CPU_actual_host_exit_v1.json')
 if name=='runtime.py':
  anchor=" install_setup_fairness(c,root/'setup_queue_admission.jsonl')\n";assert text.count(anchor)==1;text=text.replace(anchor,anchor+INIT)
  assert text.count(OLD_OBSERVE)==1;text=text.replace(OLD_OBSERVE,NEW_OBSERVE)
  anchor='  telemetry.capture=original_capture;cache.uninstall()\n';assert text.count(anchor)==1;text=text.replace(anchor,CLOSE+anchor)
  anchor=' # Avoid unwinding large CUDA/physics local ownership after a complete batch.\n';assert text.count(anchor)==1;text=text.replace(anchor,REPORT+anchor)
 if name=='host.py':
  anchor="        write(root / 'atomic_full_recovery_qualification.json',";assert text.count(anchor)==1;text=text.replace(anchor,AUDIT+anchor)
  anchor="for p in (root / 'terminal_audit.json', root / 'CPU_recovery_stream_audit.json'";assert text.count(anchor)==1;text=text.replace(anchor,"for p in (root/'CPU_render_overlap_audit.json', root / 'terminal_audit.json', root / 'CPU_recovery_stream_audit.json'")
 if name=='collection_plan.py':
  anchor='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
 return text

def restore_runtime(text):
 for fragment in(INIT,CLOSE,REPORT):assert text.count(fragment)==1;text=text.replace(fragment,'')
 assert text.count(NEW_OBSERVE)==1;text=text.replace(NEW_OBSERVE,OLD_OBSERVE)
 return original_restore(text)
