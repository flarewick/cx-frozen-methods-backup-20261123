"""Actual command-trace values through real CPU PhysX getters; no task dynamics claim."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,time,resource
import numpy as np,torch,sapien
from .capture import load,capture
from actuation_telemetry import capture as original
T=Path(__file__).resolve().parents[1];C=Path(__file__).resolve().parent;O=T/'outputs/native_telemetry_batch_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();native=load()
 source=T/'outputs/native_expert_full_failure_v5/report.json';report=read(source);assert report['passed']and report['actual_host_exit_code']==1
 protocol=T/'outputs/native_expert_full_cache_v5/protocol.json';spec=read(protocol);reused={r['index']for r in spec['recovered_cases']}
 new=[r for r in report['rows']if r['index']not in reused];assert len(new)==6
 files=[];inputs={str(source):sha(source),str(protocol):sha(protocol)};total=0
 for record in new:
  row=read(record['capture_file']);assert row['available']
  for chunk in row['chunks']:
   file=Path(chunk['file']);assert sha(file)==chunk['sha256'];inputs[str(file)]=chunk['sha256']
   with np.load(file,allow_pickle=False)as z:trace={k:z[k].copy()for k in('arm_position_target','arm_velocity_target','gripper_position_target','gripper_velocity_target','gripper_logical_command','gripper_joint_counts')}
   n=len(trace['arm_position_target']);files.append(dict(case_id=row['case_id'],query=chunk['query'],start=total,end=total+n,arrays=trace));total+=n
 assert total>=2048;chosen=np.linspace(0,total-1,2048,dtype=np.int64);assert len(set(chosen.tolist()))==2048
 scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);builder=scene.create_articulation_builder();root=builder.create_link_builder();root.set_name('root')
 finger_counts=files[0]['arrays']['gripper_joint_counts'][0].tolist();n_joints=12+sum(finger_counts)
 for i in range(n_joints):
  link=builder.create_link_builder(root);link.set_name('link'+str(i));link.set_joint_name('joint'+str(i));link.set_joint_properties('revolute',limits=np.array([[-10.,10.]]),pose_in_parent=sapien.Pose(),pose_in_child=sapien.Pose());link.set_mass_and_inertia(1.,sapien.Pose(),np.ones(3))
 articulation=builder.build(fix_root_link=True);joints=articulation.get_active_joints();assert len(joints)==n_joints
 logical=np.zeros(2,np.float64)
 robot=SimpleNamespace(left_arm_joints=joints[:6],right_arm_joints=joints[6:12],left_gripper=[(j,)for j in joints[12:12+finger_counts[0]]],right_gripper=[(j,)for j in joints[12+finger_counts[0]:]],get_left_gripper_val=lambda:float(logical[0]),get_right_gripper_val=lambda:float(logical[1]))
 before_qpos=articulation.get_qpos().copy();before_qvel=articulation.get_qvel().copy();before_rng=torch.random.get_rng_state().clone()
 timings=dict(original=[],batched=[]);checks=0;records=[];cursor=0
 for counter,absolute in enumerate(chosen):
  while absolute>=files[cursor]['end']:cursor+=1
  entry=files[cursor];offset=int(absolute-entry['start']);expected={k:v[offset]for k,v in entry['arrays'].items()};assert expected['gripper_joint_counts'].tolist()==finger_counts
  position=np.r_[expected['arm_position_target'],expected['gripper_position_target']];velocity=np.r_[expected['arm_velocity_target'],expected['gripper_velocity_target']]
  for j,p,v in zip(joints,position,velocity,strict=True):j.set_drive_target([float(p)]);j.set_drive_velocity_target([float(v)])
  logical[:]=expected['gripper_logical_command'];values={}
  order=('original','batched')if counter%2==0 else('batched','original')
  for mode in order:
   began=time.perf_counter_ns();value=original(robot)if mode=='original'else capture(robot,native);elapsed=time.perf_counter_ns()-began;timings[mode].append(elapsed);values[mode]=value
  for mode,value in values.items():
   assert set(value)==set(expected)
   for k,wanted in expected.items():assert value[k].dtype==wanted.dtype and value[k].shape==wanted.shape and np.array_equal(value[k],wanted),(mode,entry['case_id'],entry['query'],offset,k);checks+=1
  records.append(dict(case_id=entry['case_id'],query=entry['query'],tick_offset=offset))
 assert np.array_equal(before_qpos,articulation.get_qpos())and np.array_equal(before_qvel,articulation.get_qvel())and torch.equal(before_rng,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 failures=[]
 def rejects(fn):
  try:fn()
  except (ValueError,RuntimeError):return True
  return False
 logical[0]=np.nan;assert rejects(lambda:original(robot))and rejects(lambda:capture(robot,native));failures.append('nonfinite_logical_command');logical[0]=0.
 saved=robot.left_arm_joints;robot.left_arm_joints=saved[:5];assert rejects(lambda:original(robot))and rejects(lambda:capture(robot,native));failures.append('arm_joint_count');robot.left_arm_joints=saved
 saved=robot.left_gripper;robot.left_gripper=[(None,)];assert rejects(lambda:original(robot))and rejects(lambda:capture(robot,native));failures.append('missing_finger');robot.left_gripper=saved
 for p,h in inputs.items():assert sha(p)==h
 mean={k:float(np.mean(v)/1000)for k,v in timings.items()};median={k:float(np.median(v)/1000)for k,v in timings.items()}
 result=dict(passed=True,CPU_only=True,real_SAPIEN_CPU_PhysX_getters=True,real_saved_native_command_states=2048,checks=checks,source_new_native_cases=6,
  negative_contracts=failures,mean_microseconds=mean,median_microseconds=median,mean_speedup=mean['original']/mean['batched'],
  every_field_dtype_shape_value_byte_exact=True,no_scene_steps=True,qpos_qvel_unchanged=True,Torch_RNG_unchanged=True,CUDA_initialized=False,
  task_dynamics_equivalence_claimed=False,native_same_world_GPU_bridge_still_required=True,approved_for_collector=False,
  source_sha256={str(p):sha(p)for p in C.glob('*')if p.suffix in('.py','.cpp')},input_sha256=inputs,sampled_actual_tick_identities=records,
  maximum_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,training_steps=0,online_updates=0,TTA_utility_proven=False)
 file=O/'CPU_native_getter_checks.json';assert not file.exists()
 with file.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('COMMAND_TELEMETRY_REAL_SDK_AND_ACTUAL_TRACES_PASS',checks,json.dumps(dict(mean_us=mean,median_us=median,speedup=result['mean_speedup'])),flush=True)
if __name__=='__main__':main()
