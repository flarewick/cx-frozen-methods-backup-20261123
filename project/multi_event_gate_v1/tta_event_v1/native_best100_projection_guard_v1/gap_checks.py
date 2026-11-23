"""Real saved RGB/trace windows, injected projection boundary then recovery."""
from pathlib import Path
import copy,json,hashlib,numpy as np,torch
from types import SimpleNamespace
from .source import SourceStream
from .local import PROJECTION_ERROR
from native_delayed_query_v1.local_feedback import LocalFeedbackStream as Original
from native_expert_full_feedback_v1.tracking_checks import SavedOwner
from native_delayed_query_v1.primitives import configuration,old
from cx_tta.robot_kinematics import RobotFK
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def main():
 torch.set_num_threads(2)
 spec=read(T/'outputs/native_expert_feedback_v1/protocol.json');record=read(next((T/'outputs/native_expert_feedback_v1/collections').glob('*/GPU_masks_complete.json')))['rows'][0];job=spec['jobs'][0]
 cfg,camera,robots=configuration();history=[]
 for q in job['prefix']:
  z=load(q['observation']['file']);history.append(dict(query=q['query'],tick=q['tick'],episode_elapsed_s=float(q['tick']*np.float32(.004)),rgb=z['cam0'],command_state=z['command_state']))
 source=SourceStream(None,load(job['mask_file'])['masks'],history,job['instruction'],robots=robots,cfg=cfg,camera=camera,call=lambda f,*a,**k:f(*a,**k),owner_factory=lambda m,s:SavedOwner(record))
 segment=job['segments'][0];chunk=segment['chunk'];trace=load(chunk['file']);dense=load(chunk['dense_file']);callback=source.capture_callback(segment['kind'],chunk['global_start_tick'],segment['query'])
 recorder=SimpleNamespace(rows=[{k:trace[k][i]for k in ('arm_position_target','arm_velocity_target','gripper_logical_command')}for i in range(len(trace['dt_s']))],dt=trace['dt_s'])
 class Boundary:
  def project(self,*a,**k):raise ValueError(PROJECTION_ERROR)
 source.local.camera=Boundary();before_rows=source.local.rows
 i=0;callback(int(dense['ticks'][i]),float(dense['elapsed_s'][i]),dict(rgb=(dense['head_rgb'][i],),command_state=dense['command_state'][i]),recorder)
 assert source.rows==[] and len(source.missing)==1 and source.missing[0]['reason']=='commanded_TCP_projection_unsupported'
 assert source.queue.eligible(source.ticket)==()
 source.local.camera=camera
 expected=Original.__new__(Original);expected.__dict__.update(source.local.__dict__);expected.prior_labels=dict(source.local.prior_labels)
 prev=source.previous_tick;i=1;tick=int(dense['ticks'][i]);elapsed=float(dense['elapsed_s'][i]);first=prev-chunk['global_start_tick']
 window={k:trace[k][first:tick]for k in('arm_position_target','arm_velocity_target','gripper_logical_command','dt_s')}
 # SavedOwner returns the actual masks at the next frame without recomputation.
 index=source.owner.actual_tracking_calls;m=source.owner.data
 expected_rows,expected_missing=expected.arrive(dense['head_rgb'][i],m['masks'][index],m['presence'][index],window,dense['command_state'][i],tick=chunk['global_start_tick']+tick,elapsed_s=elapsed,segment=segment['kind'],segment_start_tick=chunk['global_start_tick'])
 callback(tick,elapsed,dict(rgb=(dense['head_rgb'][i],),command_state=dense['command_state'][i]),recorder)
 assert len(source.rows)==len(expected_rows)>0
 for a,b in zip(source.rows,expected_rows,strict=True):
  for key in('context','command','target','start_points','end_points','point_alive','relative_target','relative_available'):
   x,y=np.asarray(a[key]),np.asarray(b[key]);assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),key
  assert np.array_equal(a['context'][34:44],np.zeros(10))
 assert len(source.missing)==1+len(expected_missing)
 # Real failure q12 must be described honestly; the unsaved in-flight dense
 # frame cannot be recovered from a failed capture.
 failed=T/'outputs/native_best100_resume_v2/batches/003/runs/8adc1c06899d4f5a88f04b1f362fdf5e/cases/stack_bowls_two:100023/M_event_full/observation012.npz'
 q=load(failed)['command_state'];outside=False
 try:old.tcp(q[None],robots,cfg,camera)
 except ValueError as error:assert str(error)==PROJECTION_ERROR;outside=True
 result=dict(passed=True,actual_saved_window_fixture=True,injected_projection_boundary=True,no_invalid_local_rows=True,old_queue_did_not_receive_invalid_window=True,next_supported_window_byte_identical=True,gap_history_unavailable_as_original=True,saved_failed_query12_outside_frustum=outside,failed_observation_sha256=hashlib.sha256(failed.read_bytes()).hexdigest(),unsaved_failure_frame_not_reconstructed=True,GPU_initialized=torch.cuda.is_initialized(),training_steps=0)
 assert not result['GPU_initialized']
 out=T/'outputs/native_best100_projection_guard_v1/CPU_gap_and_projection_checks.json'
 with out.open('x')as f:json.dump(result,f,indent=2)
 print('PROJECTION_GAP_RECOVERY_CPU_PASS',outside,flush=True)
if __name__=='__main__':main()
