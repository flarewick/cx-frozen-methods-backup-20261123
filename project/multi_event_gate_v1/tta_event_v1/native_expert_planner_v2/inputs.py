"""Prepare readonly original physical/clock references and cached planner arrays for four FIT cases."""
from pathlib import Path
import argparse,io,json
import h5py,numpy as np
from native_expert_planner_v2.paths import *
from native_expert_replay_v1.trajectory_inventory import ArrayReader

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
 planfile=T/'outputs/native_expert_replay_v1/input_plan.json';plan=read(planfile)
 invfile=T/'outputs/native_expert_replay_v1/planner_trajectory_inventory.json';inv=read(invfile);bycase={r['case_id']:r for r in inv['rows']}
 prior=T/'outputs/native_expert_smoke_v1/collections/9ca0b742c3bbcf1f08759713';completed=read(prior/'GPU_frames_complete.json');oldrows={r['case_id']:r for r in completed['rows']}
 assert read(prior/'result.json')['complete']and read(prior/'interface_qualification.json')['passed']
 rows=[]
 for index in plan['interface_smoke_record_indices']:
  r=plan['records'][index];hfile=a.archive/Path(r['original_source']).relative_to(P);assert sha(hfile)==r['original_source_sha256']
  traj=bycase[r['case_id']];pf=a.archive/Path(traj['file']).relative_to('/mnt');assert sha(pf)==traj['sha256']
  v=ArrayReader(io.BytesIO(pf.read_bytes())).load();arrays={};segments={}
  for arm,parts in v.items():
   segments[arm]=[]
   for j,part in enumerate(parts):
    assert part['status']=='Success'and set(part)=={'status','position','velocity'},(r['case_id'],set(part))
    for key in('position','velocity'):arrays[arm+'_'+str(j)+'_'+key]=part[key]
    segments[arm].append(dict(status=part['status'],position_key=arm+'_'+str(j)+'_position',velocity_key=arm+'_'+str(j)+'_velocity'))
  with h5py.File(hfile,'r')as h:
   names=list(h['cx_privileged']);arrays.update(source_frame_ticks=h['cx_measurement/physics_tick'][:],source_frame_commands=h['joint_action/vector'][:],source_arm_qpos=h['cx_measurement/arm_qpos'][:],source_gripper_qpos=h['cx_measurement/gripper_joint_qpos'][:],source_dt=h['cx_measurement/physics_dt_s'][:],source_dense_commands=h['cx_dense/command'][:],source_dense_ticks=h['cx_dense/tick_before'][:],source_dense_dt=h['cx_dense/dt_s'][:])
   for name in names:arrays['source_object_'+name]=h['cx_privileged/'+name][:]
   for cam in('head_camera','right_camera','left_camera'):
    arrays['source_initial_'+cam+'_jpeg']=h['observation/'+cam+'/rgb'][0]
    arrays['source_initial_'+cam+'_extrinsic']=h['observation/'+cam+'/extrinsic_cv'][0]
  file=O/'inputs'/('case%03d.npz'%index);file.parent.mkdir(parents=True,exist_ok=True)
  with file.open('xb')as f:np.savez_compressed(f,**arrays)
  q0=oldrows[r['case_id']]['observations'][0]
  rows.append(dict(r,original_planner_file=str(pf),original_planner_sha256=traj['sha256'],reference_file=str(file),reference_sha256=sha(file),segments=segments,object_names=names,previous_eval_true_q0=q0,
   metadata_role='offline expert identity and teacher execution only; never online features'))
 value=dict(complete=True,records=rows,original_selected_FIT_indices=plan['interface_smoke_record_indices'],new_independent_data=False,original_data_unchanged=True,
  input_sha256={str(planfile):sha(planfile),str(invfile):sha(invfile),str(prior/'GPU_frames_complete.json'):sha(prior/'GPU_frames_complete.json')},source_sha256={str(Path(__file__).resolve()):sha(__file__)})
 write(O/'inputs.json',value);print('PLANNER_REFERENCE_INPUTS',len(rows),flush=True)
if __name__=='__main__':main()
