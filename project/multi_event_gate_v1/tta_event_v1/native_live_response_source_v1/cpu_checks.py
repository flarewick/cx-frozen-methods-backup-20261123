"""Actual saved masks/receipts replay verifies live callbacks against all12 old streams."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,numpy as np,torch
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
 cfg,camera,_=configuration();reports=[];total=0
 for job,record,old in zip(spec['jobs'],gpu['rows'],packets['streams'],strict=True):
  history=[]
  for q in job['prefix']:
   z=load(q['observation']['file']);history.append(dict(query=q['query'],tick=q['tick'],episode_elapsed_s=float(q['tick']*np.float32(.004)),rgb=z['cam0'],command_state=z['command_state']))
  source=SourceStream(None,load(job['mask_file'])['masks'],history,job['instruction'],robots=robots,cfg=cfg,camera=camera,call=lambda fn,*a,**k:fn(*a,**k),owner_factory=lambda m,s:SavedOwner(record))
  releases=[]
  for segment in job['segments']:
   chunk=segment['chunk'];trace=load(chunk['file']);dense=load(chunk['dense_file'])
   recorder=SimpleNamespace(rows=[{k:trace[k][i]for k in ('arm_position_target','arm_velocity_target','gripper_logical_command')}for i in range(len(trace['dt_s']))],dt=trace['dt_s'])
   callback=source.capture_callback(segment['kind'],chunk['global_start_tick'],segment['query'])
   for i,tick in enumerate(dense['ticks']):callback(int(tick),float(dense['elapsed_s'][i]),dict(rgb=(dense['head_rgb'][i],),command_state=dense['command_state'][i]),recorder)
   if segment['kind']=='origin':source.close_origin(trace,chunk)
   b=segment['boundary'];rgb=load(b['observation']['file'])['cam0'];source.boundary(dict(query=b['query'],tick=b['tick'],rgb=rgb))
   lease=source.queue.at_query(query=b['query'],tick=b['tick'],query_proof_sha256='a'*64)
   if lease is not None:releases.append(dict(query=lease.query,row_ids=list(lease.rows)))
  old_local=load(old['local_file'])
  for key in ('context','command','target','start_points','end_points','point_alive','relative_target','relative_available'):
   actual=np.stack([r[key]for r in source.rows]);assert actual.dtype==old_local[key].dtype and actual.shape==old_local[key].shape and actual.tobytes()==old_local[key].tobytes(),(job['job_id'],key)
  assert [r['command_query']for r in source.rows]==old_local['command_query'].tolist()
  expected=[dict(query=p['query'],row_ids=p['row_ids'])for p in packets['packets']if p['case_id']==job['case_id']and p['source_query']==job['source_query']]
  assert releases==expected
  result=source.finish();assert result['missing']==old['missing']and result['queue']==old['queue']and result['local']==old['local']
  total+=len(source.rows);reports.append(dict(job_id=job['job_id'],rows=len(source.rows),local_arrays_and_queue_release_exact=True))
 assert len(reports)==12 and torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 out=T/'outputs/native_live_response_source_v1';out.mkdir(exist_ok=True)
 report=dict(passed=True,actual_saved_streams=12,local_rows=total,records=reports,saved_owner_fixture=True,actual_SAM2_forwards=0,new_native_rollouts=0,training_steps=0,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')})
 with(out/'CPU_live_source_callback_checks.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print('LIVE_SOURCE_CALLBACK_AND_PHASE_QUEUE_COMPATIBILITY_PASS',12,total,flush=True)
if __name__=='__main__':main()
