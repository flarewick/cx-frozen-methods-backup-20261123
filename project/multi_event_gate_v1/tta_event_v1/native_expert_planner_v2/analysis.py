"""Independent raw state/clock comparisons. Mismatches are outcomes, not discarded cases."""
from pathlib import Path
import argparse,numpy as np
from native_expert_planner_v2.paths import *
def load(f):
 with np.load(f,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def compare(a,b):
 a=np.asarray(a);b=np.asarray(b)
 if a.shape!=b.shape:return dict(shape_equal=False,actual_shape=list(a.shape),reference_shape=list(b.shape),exact=False,max_abs=None)
 return dict(shape_equal=True,exact=bool(np.array_equal(a,b)),max_abs=float(np.max(abs(a.astype(float)-b.astype(float))))if a.size else 0.)
def object_compare(a,b):
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape==(7,)
 return dict(position_max_abs_m=float(abs(a[:3]-b[:3]).max()),quaternion_sign_invariant_max_abs=float(min(abs(a[3:]-b[3:]).max(),abs(a[3:]+b[3:]).max())))
def audit(spec,root):
 verify(spec);report=read(root/'GPU_frames_complete.json');assert report['complete']and report['protocol_identity']==spec['identity']and len(report['rows'])==len(spec['scenes'])==4
 rows=[];checks=0
 for item,row in zip(spec['scenes'],report['rows']):
  assert item['case_id']==row['case_id'];checks+=1
  if not row['available']:
   assert row['reason']=='native_unstable_initialization_retained';rows.append(dict(case_id=item['case_id'],available=False));continue
  assert [v['eval_mode']for v in row['initializations']]==[True,False]
  modes={v['eval_mode']:load(v['file'])for v in row['initializations']}
  for v in row['initializations']:assert sha(v['file'])==v['sha256']and v['native_action_count']==0;checks+=1
  oldq0=load(item['previous_eval_true_q0']['file']);reference=load(item['reference_file']);actual=load(row['file']);assert sha(row['file'])==row['sha256'];checks+=1
  q0_same=all(np.array_equal(oldq0[k],modes[True][k])for k in('cam0','cam1','cam2','command_state'))
  pairs={name:object_compare(modes[True]['object_'+name],modes[False]['object_'+name])for name in item['object_names']}
  init_ref={name:object_compare(modes[False]['object_'+name],reference['source_object_'+name][0])for name in item['object_names']}
  assert row['frames']==row['original_picture_calls']==len(actual['frame_ticks'])and np.all(np.diff(actual['frame_ticks'])>=0)
  assert row['policy_take_action_calls']==0 and row['actual_recorded_motion_steps']==row['last_physics_tick']-row['first_motion_tick']
  n=row['actual_recorded_motion_steps'];assert np.array_equal(actual['actual_tick_before'],np.arange(row['first_motion_tick'],row['last_physics_tick']))
  for k,shape in(('actual_arm_position_target',(n,12)),('actual_arm_velocity_target',(n,12)),('actual_gripper_logical_command',(n,2))):assert actual[k].shape==shape and np.isfinite(actual[k]).all();checks+=1
  assert np.all(abs(actual['actual_dt_s']-.004)<1e-8)
  commands=np.column_stack((actual['actual_arm_position_target'][:,:6],actual['actual_gripper_logical_command'][:,0],actual['actual_arm_position_target'][:,6:],actual['actual_gripper_logical_command'][:,1])).astype(np.float32)
  offset=row['first_motion_tick']-int(actual['dense_ticks'][0]);assert np.array_equal(commands,actual['dense_commands'][offset:offset+n]);checks+=1
  comparisons={k:compare(actual[k],reference['source_'+k])for k in('dense_commands','dense_ticks','dense_dt','frame_ticks','frame_commands','arm_qpos','gripper_qpos')}
  # Actual poses are aligned to the original frame index only when all frame ticks agree.
  if comparisons['frame_ticks']['exact']:
   comparisons['objects']={name:compare(actual['object_'+name],reference['source_object_'+name])for name in item['object_names']}
  else:comparisons['objects']=dict(unavailable_reason='original_frame_ticks_do_not_match')
  rows.append(dict(case_id=item['case_id'],available=True,native_task_success=row['native_task_success'],native_planner_success=row['native_planner_success'],
   original_eval_true_q0_RGB_and_state_exact=q0_same,actual_initial_mode_object_differences=pairs,
   original_mode_q0_vs_earliest_saved_frame=init_ref,original_mode_comparison_has_one_step_clock_difference=True,
   original_record_comparisons=comparisons,actual_motion_steps=n,actual_motion_duration_s=float(actual['actual_dt_s'].sum()),
   frames=row['frames'],actual_velocity_nonzero_elements=int(np.count_nonzero(actual['actual_arm_velocity_target'])),
   cached_path_counts=dict(left_used=row['left_paths_used'],left_available=row['left_paths_available'],right_used=row['right_paths_used'],right_available=row['right_paths_available']),
   privilege_used_only_for_offline_identity=True))
 value=dict(passed=True,complete=True,checks=checks,requested=4,rows=rows,successes=sum(r.get('native_task_success',False)for r in rows),
  CPU_only=True,interface_audit_does_not_require_success_or_old_trajectory_equality=True,all_mismatches_and_negative_cases_retained=True,
  native32_policy_training_cache_ready=False,direction_training_steps=0,online_updates=0,TTA_utility_proven=False)
 return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();s=read(O/'protocol.json');assert read(a.dataset/'result.json')['complete']
 v=audit(s,a.dataset);write(a.dataset/'interface_qualification.json',v);print('ORIGINAL_PLANNER_CPU_AUDIT',json.dumps(v),flush=True)
