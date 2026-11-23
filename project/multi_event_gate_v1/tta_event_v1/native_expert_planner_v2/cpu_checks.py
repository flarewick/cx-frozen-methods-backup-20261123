"""CPU contracts for the exact step rewrite, picture sink and reference alignment."""
from pathlib import Path
from types import SimpleNamespace
import ast,numpy as np
from native_expert_planner_v2.paths import *
from native_expert_planner_v2.recording import instrument,Sink
from native_expert_planner_v2.analysis import compare,object_compare

def main():
 old=read(T/'outputs/native_feedback_confirm_v5/protocol.json');file=Path(old['runtime_source'])/'envs/_base_task.py';source=file.read_text();changed=instrument(source)
 restored=changed.split('\n',1)[1].replace('_cx_record_step(self)','self.scene.step()');assert restored==source
 tree=ast.parse(source);picture=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='_take_picture')
 assert sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='save_pkl'for n in ast.walk(picture))==1
 # Other module functions may save trajectories; only _take_picture receives the private sink namespace.
 sink=Sink(SimpleNamespace(),['object'])
 obs=dict(cx_measurement=dict(physics_tick=2801,physics_dt_s=.004,arm_qpos=np.ones((2,6)),gripper_joint_qpos=np.zeros(2)),cx_privileged=dict(object=np.array([0,0,1,1,0,0,0.])),joint_action=dict(vector=np.zeros(14)),observation={k:dict(rgb=np.zeros((240,320,3),np.uint8),extrinsic_cv=np.eye(3,4))for k in('head_camera','right_camera','left_camera')})
 sink('0.pkl',obs);sink('1.pkl',obs);arrays=sink.arrays();assert len(arrays['frame_ticks'])==2 and np.array_equal(arrays['frame_ticks'],[2801,2801])
 assert not compare(np.zeros((3,14)),np.zeros((2,14)))['shape_equal']
 assert not compare(np.zeros(2),np.array([1.,0]))['exact']
 assert object_compare(np.array([0,0,1,1,0,0,0.]),np.array([0,0,1,-1,0,0,0.]))['quaternion_sign_invariant_max_abs']==0
 from native_expert_planner_v2.observe import unrecorded
 original=lambda:'raw';recorded=lambda:'recorded';task=SimpleNamespace(get_obs=recorded)
 with unrecorded(task,original):assert task.get_obs()=='raw'
 assert task.get_obs is recorded
 try:
  with unrecorded(task,original):raise RuntimeError('fixture')
 except RuntimeError:pass
 assert task.get_obs is recorded
 references=read(O/'inputs.json');assert len(references['records'])==4
 for r in references['records']:
  assert sha(r['reference_file'])==r['reference_sha256']
  with np.load(r['reference_file'],allow_pickle=False)as z:
   assert np.all(np.diff(z['source_dense_ticks'])==1)and np.all(np.diff(z['source_frame_ticks'])>=0)
   for parts in r['segments'].values():
    for v in parts:assert z[v['position_key']].shape==z[v['velocity_key']].shape
 write(O/'cpu_checks.json',dict(passed=True,CPU_fixture_only=True,original_runtime_AST_reversible_exact=True,original_picture_callback_and_duplicate_ticks_preserved=True,shape_and_numeric_mismatches_retained=True,quaternion_sign_invariance=True,all_four_reference_arrays_checked=True,GPU_started=False))
 print('EXPERT_PLANNER_CPU_CONTRACTS_PASS',flush=True)
if __name__=='__main__':main()
