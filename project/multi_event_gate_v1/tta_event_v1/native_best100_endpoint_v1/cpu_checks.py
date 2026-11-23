"""Saved real RGB/trace replay and strict endpoint control-flow checks; no GPU."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json,numpy as np,torch
from .readouts import CurrentReadouts,rgb_sha
from cx_tta_feedback_confirm5.capture import CurrentReadouts as Original
from native_phase_feedback_v1.feedback_queue import FeedbackQueue
from native_best100_projection_guard_v1.source import SourceStream
from native_expert_full_feedback_v1.tracking_checks import SavedOwner
from native_expert_packets_v1.build import configuration,build_head
T=Path(__file__).resolve().parents[1];O=T/'outputs/native_best100_endpoint_v1'
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def observation(rgb,command,meta):
 return dict(rgb=(rgb,),command_state=command,physical_sha256=meta['physics_sha256'],rng_sha256=meta['rng_sha256'],images={'0':hashlib.sha256(rgb.tobytes()).hexdigest()})
def boundary_checks():
 root=T/'outputs/native_best100_resume_v4/batches/004/runs/333cf82c219c4771a1ad51ac12814c79/cases/stack_blocks_two:100019/TTA'
 dense=load(root/'execution007_arrivals.npz');trace=load(root/'execution007.npz');meta=read(root/'execution007.json');query=load(root/'observation008.npz');lookup={int(t):i for i,t in enumerate(dense['ticks'])};tick=len(trace['dt_s']);assert tick==1312 and tick%32==0
 assert not np.array_equal(dense['head_rgb'][-1],query['cam0'])and np.array_equal(dense['command_state'][-1],query['command_state'])
 prefix=0.
 for q in range(7):prefix+=float(np.sum(load(root/('execution%03d.npz'%q))['dt_s']))
 def run(cls,*,censored=False,bad=None):
  rec=SimpleNamespace(rows=[],dt=[],failed=False);calls={};delivered=[];queue=FeedbackQueue()
  def reader(t,elapsed):
   index=lookup[t];calls[t]=calls.get(t,0)+1;rgb=dense['head_rgb'][index];q=dense['command_state'][index];m=meta['readouts'][index]
   if t==tick and calls[t]>1:
    rgb=query['cam0'];q=query['command_state'].copy()
    if bad=='command':q[0]+=.1
    if bad=='physics':m=dict(m,physics_sha256='CPU_fixture_physics_mismatch')
   return observation(rgb,q,m)
  def consumer(t,e,obs,recorder):
   delivered.append(dict(tick=t,recorder_length=len(recorder.rows),RGB_sha256=rgb_sha(obs)));queue.arrive(meta['global_start_tick']+t,prefix+e,rgb_sha(obs))
  hook=cls(rec,reader,on_arrival=consumer)
  for i in range(tick):
   rec.rows.append({k:trace[k][i]for k in('arm_position_target','arm_velocity_target','gripper_logical_command')});rec.dt.append(float(trace['dt_s'][i]));hook.after(i)
  if cls is Original:hook.read(tick)
  else:hook.terminal_read(tick,complete_native_chunk=not censored)
  assert len(delivered)==len(lookup)and len({v['tick']for v in delivered})==len(delivered)
  if cls is not Original:
   assert all(v['recorder_length']==v['tick']+1 for v in delivered[:-1]);assert delivered[-1]['recorder_length']==tick
  return hook,queue,calls,delivered
 old,queue,_,_=run(Original)
 try:queue.arrive(meta['global_start_tick']+tick,prefix+float(sum(trace['dt_s'])),hashlib.sha256(query['cam0'].tobytes()).hexdigest())
 except ValueError as error:assert str(error)=='Same tick has conflicting RGB/physical clock'
 else:raise AssertionError('Actual failed endpoint did not reproduce original error')
 fixed,queue,calls,delivered=run(CurrentReadouts);queue.arrive(meta['global_start_tick']+tick,prefix+float(sum(trace['dt_s'])),hashlib.sha256(query['cam0'].tobytes()).hexdigest());assert fixed.endpoint_readback['RGB_changed']and calls[tick]==2
 try:queue.arrive(meta['global_start_tick']+tick,prefix+float(sum(trace['dt_s'])), 'b'*64)
 except ValueError:pass
 else:raise AssertionError('Strict RGB duplicate check was relaxed')
 for bad in('command','physics'):
  try:run(CurrentReadouts,bad=bad)
  except ValueError:pass
  else:raise AssertionError('Non-photometric change swallowed')
 partial,_,calls,_=run(CurrentReadouts,censored=True);assert calls[tick]==1 and partial.endpoint_readback is None
 return dict(actual_failed_RGB_and_commands_replayed=True,original_error_reproduced=True,strict_original_queue_now_accepts_actual_query_RGB=True,unchanged_clock_and_command=True,only_one_feedback_frame_per_tick=True,no_future_command_prefix_used=True,censored_endpoint_unchanged=True,changed_command_physics_and_conflicting_RGB_rejected=True,saved_boundary_GT_not_available=True,CPU_fixture_reuses_same_tick_diagnostic_physics_only=True,actual_failed_observation_sha256=sha(root/'observation008.npz'))
def compatible_streams():
 spec=read(T/'outputs/native_expert_feedback_v1/protocol.json');gpu=read(next((T/'outputs/native_expert_feedback_v1/collections').glob('*/GPU_masks_complete.json')));packets=read(T/'outputs/native_expert_packets_v1/manifest.json')
 with torch.random.fork_rng(devices=[]):head,robots=build_head(read(T/'outputs/native_expert_policy_v1/protocol.json')['training'])
 cfg,camera,_=configuration();reports=[];total=0
 for job,record,archived in zip(spec['jobs'],gpu['rows'],packets['streams'],strict=True):
  history=[]
  for q in job['prefix']:
   z=load(q['observation']['file']);history.append(dict(query=q['query'],tick=q['tick'],episode_elapsed_s=float(q['tick']*np.float32(.004)),rgb=z['cam0'],command_state=z['command_state']))
  source=SourceStream(None,load(job['mask_file'])['masks'],history,job['instruction'],robots=robots,cfg=cfg,camera=camera,call=lambda fn,*a,**k:fn(*a,**k),owner_factory=lambda m,s:SavedOwner(record));releases=[]
  for segment in job['segments']:
   chunk=segment['chunk'];trace=load(chunk['file']);dense=load(chunk['dense_file']);rows=[{k:trace[k][i]for k in('arm_position_target','arm_velocity_target','gripper_logical_command')}for i in range(len(trace['dt_s']))];rec=SimpleNamespace(rows=[],dt=[],failed=False);positions={int(t):i for i,t in enumerate(dense['ticks'])};boundary=load(segment['boundary']['observation']['file']);seen={}
   def reader(t,e):
    i=positions[t];seen[t]=seen.get(t,0)+1;rgb=dense['head_rgb'][i];cmd=dense['command_state'][i]
    if t==len(trace['dt_s'])and seen[t]>1:rgb=boundary['cam0'];cmd=boundary['command_state'];assert np.array_equal(rgb,dense['head_rgb'][i])and np.array_equal(cmd,dense['command_state'][i])
    return observation(rgb,cmd,chunk['readouts'][i])
   callback=source.capture_callback(segment['kind'],chunk['global_start_tick'],segment['query']);hook=CurrentReadouts(rec,reader,on_arrival=callback)
   # Replay actual sampled points, with the original issuing prefix available.
   # Dispatch delay does not change any captured tick/image/window.
   for t in dense['ticks']:
    t=int(t);rec.rows=rows[:t];rec.dt=trace['dt_s'][:t].tolist();hook.flush();hook.read(t)
   hook.terminal_read(len(trace['dt_s']),complete_native_chunk=chunk['complete_source_chunk'])
   if segment['kind']=='origin':source.close_origin(trace,chunk)
   b=segment['boundary'];source.boundary(dict(query=b['query'],tick=b['tick'],rgb=boundary['cam0']))
   lease=source.queue.at_query(query=b['query'],tick=b['tick'],query_proof_sha256='a'*64)
   if lease is not None:releases.append(dict(query=lease.query,row_ids=list(lease.rows)))
  old_local=load(archived['local_file'])
  for key in('context','command','target','start_points','end_points','point_alive','relative_target','relative_available'):
   actual=np.stack([r[key]for r in source.rows]);assert actual.dtype==old_local[key].dtype and actual.shape==old_local[key].shape and actual.tobytes()==old_local[key].tobytes(),(job['job_id'],key)
  assert[r['command_query']for r in source.rows]==old_local['command_query'].tolist()
  expected=[dict(query=p['query'],row_ids=p['row_ids'])for p in packets['packets']if p['case_id']==job['case_id']and p['source_query']==job['source_query']];assert releases==expected
  result=source.finish();assert result['missing']==archived['missing']and result['queue']==archived['queue']and result['local']==archived['local'];total+=len(source.rows);reports.append(dict(job_id=job['job_id'],rows=len(source.rows),arrays_queue_phase_release_exact=True))
 assert len(reports)==12 and total==1594
 return reports,total
def main():
 from .capture import capture
 from .episode import execute
 assert callable(capture)and callable(execute)
 torch.set_num_threads(2);before=torch.random.get_rng_state().clone();bound=boundary_checks();records,total=compatible_streams();assert torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 old=ast.parse((T/'native_visible_episode_v1/episode.py').read_text());new=ast.parse((Path(__file__).parent/'episode.py').read_text());a=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='execute');b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='execute');assert ast.dump(a)==ast.dump(b)
 report=dict(passed=True,actual_capture_and_episode_import_passed=True,boundary=bound,actual_saved_streams=12,local_rows=total,records=records,original_episode_body_AST_exact=True,strict_queue_source_unchanged=True,actual_SAM_forwards=0,GPU_initialized=False,training_steps=0,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')})
 O.mkdir(exist_ok=True)
 with(O/'CPU_endpoint_checks_v2.json').open('x')as f:json.dump(report,f,indent=2)
 print('ENDPOINT_NATIVE_CADENCE_CPU_COMPATIBILITY_PASS',12,total,flush=True)
if __name__=='__main__':main()
