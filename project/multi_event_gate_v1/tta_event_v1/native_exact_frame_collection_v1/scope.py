INSTALL=" from native_exact_frame_cv2_v1.frames import install as install_exact_frames,active as exact_frames_active\n install_exact_frames()\n"
OLD="       with sam2_profile():return fn(*args,**kwargs)"
NEW="       with exact_frames_active(branch=='M_event_full'),sam2_profile():return fn(*args,**kwargs)"
PLAN="""  probe_file=T/'outputs/native_exact_frame_gpu_v1/actual_report.json';probe=read(probe_file);verify(probe)
  assert probe['passed']and probe['own_GPU_released']and probe['actual_host_exit']==0
  prior_fit=T/'outputs/native_onpolicy_full_ops_v2/ARM36_actual_resource_result_v1.json';fit=read(prior_fit);verify(fit)
  assert fit['stage']=='fit_000'and fit['actual_host_exit']['exit_code']==1 and fit['release']['own_gpu_released']
  if split=='val':
   v['inherited_stage_GPU_seconds']=probe['cumulative_VAL_seconds'];v['cumulative_budget_amendment']['all_failed_seconds_retained']=v['inherited_stage_GPU_seconds']
  elif number==0:v['inherited_stage_GPU_seconds']=fit['cumulative_stage_GPU_seconds']
  v['exact_CPU_cv2_frame_normalization']=True
  v['input_sha256'].update({str(p):sha(p)for p in [probe_file,prior_fit]})
  v['source_sha256'].update({str(p):sha(p)for folder in ['native_exact_frame_cv2_v1','native_exact_frame_collection_v1']for p in(T/folder).glob('*')if p.is_file()})
"""
def transformed(name,text):
 text=text.replace('native_onpolicy_arm_collection_v36','native_onpolicy_arm_collection_v37').replace('native_onpolicy_calibration_data_v28','native_onpolicy_calibration_data_v29')
 if name=='runtime.py':
  anchor=' assert check_precision()\n';assert text.count(anchor)==1;text=text.replace(anchor,anchor+INSTALL)
  assert text.count(OLD)==1;text=text.replace(OLD,NEW)
 if name=='collection_plan.py':
  anchor='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
 return text

def restore(text):
 assert text.count(INSTALL)==1 and text.count(NEW)==1
 return text.replace(INSTALL,'').replace(NEW,OLD)
