"""Actual saved masks/receipts replay verifies live callbacks against all12 old streams."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,copy,numpy as np,torch
from .local import PROJECTION_ERROR
from native_delayed_query_v1.local_feedback import LocalFeedbackStream as OriginalLocal
from native_delayed_query_v1.primitives import old
from .source import SourceStream
from native_expert_full_feedback_v1.tracking_checks import SavedOwner
from native_expert_packets_v1.build import configuration,build_head
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def main():
 torch.set_num_threads(2);before=torch.random.get_rng_state().clone()
 spec=read(T/'outputs/native_expert_feedback_v1/protocol.json');gpu=read(next((T/'outputs/native_expert_feedback_v1/collections').glob('*/GPU_masks_complete.json')));packets=read(T/'outputs/native_expert_packets_v1/manifest.json')
 with torch.random.fork_rng(devices=[]):head,robots=build_head(read(T/'outputs/native_expert_policy_v1/protocol.json')['training'])
 cfg,camera,_=configuration();reports=[];total=0;boundary_checks=[]
 for job,record,archived in zip(spec['jobs'],gpu['rows'],packets['streams'],strict=True):
  history=[]
  for q in job['prefix']:
   z=load(q['observation']['file']);history.append(dict(query=q['query'],tick=q['tick'],episode_elapsed_s=float(q['tick']*np.float32(.004)),rgb=z['cam0'],command_state=z['command_state']))
  source=SourceStream(None,load(job['mask_file'])['masks'],history,job['instruction'],robots=robots,cfg=cfg,camera=camera,call=lambda fn,*a,**k:fn(*a,**k),owner_factory=lambda m,s:SavedOwner(record))
  releases=[]
  for segment in job['segments']:
   chunk=segment['chunk'];trace=load(chunk['file']);dense=load(chunk['dense_file'])
   recorder=SimpleNamespace(rows=[{k:trace[k][i]for k in ('arm_position_target','arm_velocity_target','gripper_logical_command')}for i in range(len(trace['dt_s']))],dt=trace['dt_s'])
   callback=source.capture_callback(segment['kind'],chunk['global_start_tick'],segment['query'])
   for i,tick in enumerate(dense['ticks']):
    previous_tick=source.previous_tick
    local_before=copy.copy(source.local);local_before.__dict__=dict(source.local.__dict__)
    callback(int(tick),float(dense['elapsed_s'][i]),dict(rgb=(dense['head_rgb'][i],),command_state=dense['command_state'][i]),recorder)
    if not boundary_checks and source.local.rows>local_before.rows:
     first=previous_tick-chunk['global_start_tick'];last=int(tick)
     window={k:np.stack([r[k]for r in recorder.rows[first:last]])for k in ('arm_position_target','arm_velocity_target','gripper_logical_command')};window['dt_s']=np.asarray(recorder.dt[first:last],np.float64)
     kwargs=dict(tick=source.previous_tick,elapsed_s=source.last_s,segment=segment['kind'],segment_start_tick=chunk['global_start_tick'])
     args=(dense['head_rgb'][i],source.latest['masks'],source.latest['presence'],window,dense['command_state'][i])
     class BrokenCamera:
      def __init__(self,message):self.message=message
      def project(self,*a,**k):raise ValueError(self.message)
     def clone(cls,camera):
      value=cls.__new__(cls);value.__dict__.update(local_before.__dict__);value.camera=camera;value.prior_labels=dict(local_before.prior_labels);return value
     raw=clone(OriginalLocal,BrokenCamera(PROJECTION_ERROR))
     try:raw.arrive(*args,**kwargs)
     except ValueError as error:assert str(error)==PROJECTION_ERROR
     else:raise AssertionError('Fixture did not exercise original projection failure')
     guarded=clone(type(source.local),BrokenCamera(PROJECTION_ERROR));rows,missing=guarded.arrive(*args,**kwargs)
     assert rows==[]and len(missing)==1 and missing[0]['reason']=='commanded_TCP_projection_unsupported'
     assert guarded.rows==local_before.rows and guarded.missing==local_before.missing+1
     assert guarded.previous['tick']==source.previous_tick and np.array_equal(guarded.command,source.local.command)
     assert guarded.prior_labels==local_before.prior_labels
     for message in ('unrelated camera validation error',):
      bad=clone(type(source.local),BrokenCamera(message))
      try:bad.arrive(*args,**kwargs)
      except ValueError as error:assert str(error)==message
      else:raise AssertionError('Unrelated error swallowed')
     bad=clone(type(source.local),BrokenCamera(PROJECTION_ERROR));broken=dict(window,dt_s=window['dt_s']*2)
     try:bad.arrive(args[0],args[1],args[2],broken,args[4],**kwargs)
     except ValueError as error:assert str(error)=='Only the exact arrived own-command prefix'
     else:raise AssertionError('Invalid trace swallowed')
     # On the next supported window, old contiguous-history checks see the
     # gap naturally. Compare original and repaired streams from that state.
     guarded.camera=camera
     boundary_checks.append(dict(saved_real_window=True,injected_projection_error_fixture=True,original_error_reproduced=True,no_target_or_packet_generated=True,physical_clock_and_exact_command_advanced=True,unrelated_error_and_bad_trace_propagated=True,old_prior_labels_retained=True))
   if segment['kind']=='origin':source.close_origin(trace,chunk)
   b=segment['boundary'];rgb=load(b['observation']['file'])['cam0'];source.boundary(dict(query=b['query'],tick=b['tick'],rgb=rgb))
   lease=source.queue.at_query(query=b['query'],tick=b['tick'],query_proof_sha256='a'*64)
   if lease is not None:releases.append(dict(query=lease.query,row_ids=list(lease.rows)))
  old_local=load(archived['local_file'])
  for key in ('context','command','target','start_points','end_points','point_alive','relative_target','relative_available'):
   actual=np.stack([r[key]for r in source.rows]);assert actual.dtype==old_local[key].dtype and actual.shape==old_local[key].shape and actual.tobytes()==old_local[key].tobytes(),(job['job_id'],key)
  assert [r['command_query']for r in source.rows]==old_local['command_query'].tolist()
  expected=[dict(query=p['query'],row_ids=p['row_ids'])for p in packets['packets']if p['case_id']==job['case_id']and p['source_query']==job['source_query']]
  assert releases==expected
  result=source.finish();assert result['missing']==archived['missing']and result['queue']==archived['queue']and result['local']==archived['local']
  total+=len(source.rows);reports.append(dict(job_id=job['job_id'],rows=len(source.rows),local_arrays_and_queue_release_exact=True))
 assert len(reports)==12 and torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 failed=T/'outputs/native_best100_resume_v2/batches/003/runs/8adc1c06899d4f5a88f04b1f362fdf5e/cases/stack_bowls_two:100023/M_event_full'
 q=load(failed/'observation012.npz')['command_state']
 try:old.tcp(q[None],robots,cfg,camera)
 except ValueError as error:
  assert str(error)==PROJECTION_ERROR
  actual_failure=dict(saved_query12_command_outside_frustum=True,observation_sha256=sha(failed/'observation012.npz'))
 else:actual_failure=dict(saved_query12_command_outside_frustum=False,unsaved_midchunk_command_not_reconstructed=True,observation_sha256=sha(failed/'observation012.npz'))
 assert len(boundary_checks)==1
 out=T/'outputs/native_best100_projection_guard_v1';out.mkdir(exist_ok=True)
 report=dict(passed=True,boundary_checks=boundary_checks,actual_failure_diagnostic=actual_failure,actual_saved_streams=12,local_rows=total,records=reports,saved_owner_fixture=True,actual_SAM2_forwards=0,new_native_rollouts=0,training_steps=0,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')})
 with(out/'CPU_live_source_callback_checks.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print('LIVE_SOURCE_CALLBACK_AND_PHASE_QUEUE_COMPATIBILITY_PASS',12,total,flush=True)
if __name__=='__main__':main()
