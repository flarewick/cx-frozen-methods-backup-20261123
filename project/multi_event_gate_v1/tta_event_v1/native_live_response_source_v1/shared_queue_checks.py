"""Real late-q2 source with episode-global shared queue, not independent local time."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,numpy as np,torch
from .source import SourceStream
from native_phase_feedback_v1.feedback_queue import FeedbackQueue
from native_expert_full_feedback_v1.tracking_checks import SavedOwner
from native_expert_packets_v1.build import configuration,build_head
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def rgb_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def main():
 torch.set_num_threads(2);before=torch.random.get_rng_state().clone()
 spec=read(T/'outputs/native_expert_feedback_v1/protocol.json');packets=read(T/'outputs/native_expert_packets_v1/manifest.json')['packets'];p=max((p for p in packets if p['query']==p['source_query']+2),key=lambda p:p['source_query'])
 i=next(i for i,j in enumerate(spec['jobs'])if j['case_id']==p['case_id']and j['source_query']==p['source_query']);job=spec['jobs'][i];record=read(next((T/'outputs/native_expert_feedback_v1/collections').glob('*/GPU_masks_complete.json')))['rows'][i]
 with torch.random.fork_rng(devices=[]):head,robots=build_head(read(T/'outputs/native_expert_policy_v1/protocol.json')['training'])
 cfg,camera,_=configuration();history=[];queue=FeedbackQueue()
 for ref in job['prefix']:
  z=load(ref['observation']['file']);h=dict(query=ref['query'],tick=ref['tick'],episode_elapsed_s=float(ref['tick']*np.float32(.004)),rgb=z['cam0'],command_state=z['command_state']);history.append(h)
  queue.arrive(h['tick'],h['episode_elapsed_s'],rgb_sha(h['rgb']));assert queue.at_query(query=h['query'],tick=h['tick'],query_proof_sha256='a'*64)is None
 ticket=queue.register(job['source_query'])
 source=SourceStream(None,load(job['mask_file'])['masks'],history,job['instruction'],robots=robots,cfg=cfg,camera=camera,queue=queue,ticket=ticket,call=lambda fn,*a,**k:fn(*a,**k),owner_factory=lambda m,s:SavedOwner(record))
 assert source.queue_start_s==history[-1]['episode_elapsed_s']>0
 releases=[]
 for segment in job['segments']:
  chunk=segment['chunk'];trace=load(chunk['file']);dense=load(chunk['dense_file']);callback=source.capture_callback(segment['kind'],chunk['global_start_tick'],segment['query'])
  rec=SimpleNamespace(rows=[{k:trace[k][j]for k in ('arm_position_target','arm_velocity_target','gripper_logical_command')}for j in range(len(trace['dt_s']))],dt=trace['dt_s'])
  for j,tick in enumerate(dense['ticks']):callback(int(tick),float(dense['elapsed_s'][j]),dict(rgb=(dense['head_rgb'][j],),command_state=dense['command_state'][j]),rec)
  if segment['kind']=='origin':source.close_origin(trace,chunk)
  b=segment['boundary'];source.boundary(dict(query=b['query'],tick=b['tick'],rgb=load(b['observation']['file'])['cam0']))
  packet=queue.at_query(query=b['query'],tick=b['tick'],query_proof_sha256='b'*64)
  if packet is not None:releases.append(packet)
 assert len(releases)==1 and releases[0].query==p['query']and list(releases[0].rows)==p['row_ids']
 result=source.finish();assert result['shared_queue_ownership_preserved']and result['queue']is None and not queue.closed
 summary=queue.finish();assert summary['released_feedback_packets']==1
 assert torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 out=T/'outputs/native_live_response_source_v1';report=dict(passed=True,actual_job=job['job_id'],source_query=job['source_query'],actual_q2_release=p['query'],episode_global_queue_start_s=source.queue_start_s,source_relative_local_windows_preserved=True,shared_queue_not_closed_by_source=True,real_SAM2_forwards=0,new_native_rollouts=0,saved_owner_replay_fixture=True,source_sha256={str(f):sha(f)for f in [Path(__file__),Path(__file__).with_name('source.py')]})
 with(out/'CPU_shared_absolute_phase_queue_checks.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print('SHARED_EPISODE_PHASE_QUEUE_Q2_BINDING_PASS',job['job_id'],p['query'],flush=True)
if __name__=='__main__':main()
