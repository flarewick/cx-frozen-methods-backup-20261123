"""Actual four old cases verify independent prefixes, shared phases and early retirement."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,numpy as np,torch
from .source import SourceStream
from .pool import SourcePool
from native_expert_full_feedback_v1.tracking_checks import SavedOwner
from native_expert_packets_v1.build import configuration,build_head
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def main():
 torch.set_num_threads(2);before=torch.random.get_rng_state().clone()
 index=read(T/'outputs/native_expert_full_feedback_v1/index.json');spec=read(T/'outputs/native_expert_feedback_v1/protocol.json');old=read(T/'outputs/native_expert_packets_v1/manifest.json');gpu=read(next((T/'outputs/native_expert_feedback_v1/collections').glob('*/GPU_masks_complete.json')))
 jobs={j['job_id']:(j,r)for j,r in zip(spec['jobs'],gpu['rows'],strict=True)}
 with torch.random.fork_rng(devices=[]):head,robots=build_head(read(T/'outputs/native_expert_policy_v1/protocol.json')['training'])
 cfg,camera,_=configuration();checks=[];packets=0
 for case in [c for c in index['cases']if c['old_pilot_reused']]:
  case_jobs={j['source_query']:j for j in spec['jobs']if j['case_id']==case['case_id']};expected={p['query']:p for p in old['packets']if p['case_id']==case['case_id']}
  def factory(history,queue,ticket):
   job,record=jobs[case['case_id']+'/source'+str(history[-1]['query'])]
   return SourceStream(None,load(job['mask_file'])['masks'],history,case['instruction'],robots=robots,cfg=cfg,camera=camera,queue=queue,ticket=ticket,call=lambda fn,*a,**k:fn(*a,**k),owner_factory=lambda m,s:SavedOwner(record))
  pool=SourcePool(factory);max_live=0
  for query in case['queries']:
   q=query['query'];obs=load(query['observation']['file']);cur=dict(query=q,tick=query['tick'],episode_elapsed_s=float(query['tick']*np.float32(.004)),rgb=obs['cam0'],command_state=obs['command_state'])
   pool.arrive_query(cur);lease=pool.queue.at_query(query=q,tick=query['tick'],query_proof_sha256='a'*64)
   if lease is not None:
    p=expected[q];assert lease.source_query==p['source_query']and list(lease.rows)==p['row_ids'];arrays=load(p['local_file']);rows=pool.rows()[lease.source_query]
    for key in ('context','command','target','start_points','end_points','point_alive','relative_target','relative_available'):
     got=np.stack([rows[i][key]for i in lease.rows]);wanted=arrays[key][list(lease.rows)]
     if key=='relative_target':
      # Old full-stream stacking may promote this optional diagnostic only when
      # a LATER row contains a float64 NaN. Use a fixed lossless diagnostic dtype.
      got=got.astype(np.float64);wanted=wanted.astype(np.float64)
     assert got.dtype==wanted.dtype and got.shape==wanted.shape and got.tobytes()==wanted.tobytes(),key
    packets+=1
   else:assert q not in expected
   pool.retire_after_query(q)
   if q in case_jobs:pool.admit()
   max_live=max(max_live,len(pool.live));chunk=query['trace'];plan=pool.capture_plan(q,query['tick'])
   if plan['collect_dense']:
    trace=load(chunk['file']);dense=load(chunk['dense_file']);rec=SimpleNamespace(rows=[{k:trace[k][i]for k in ('arm_position_target','arm_velocity_target','gripper_logical_command')}for i in range(len(trace['dt_s']))],dt=trace['dt_s'])
    for i,tick in enumerate(dense['ticks']):plan['on_arrival'](int(tick),float(dense['elapsed_s'][i]),dict(rgb=(dense['head_rgb'][i],),command_state=dense['command_state'][i]),rec)
    pool.close_origin(q,trace,chunk)
  result=pool.finish();summary=pool.queue.finish()
  assert result['remaining_live_sources']==0 and len(result['completed_sources'])==3 and summary['released_feedback_packets']==3 and max_live==1
  assert all(r['feedback_consumed']for r in result['completed_sources'])
  checks.append(dict(case_id=case['case_id'],source_local_prefixes=True,all_three_packets_exact=True,maximum_live_source_owners=max_live,immediate_completed_owner_release_verified=True))
 assert packets==12 and torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 out=T/'outputs/native_live_response_source_v1';report=dict(passed=True,actual_old_cases=checks,actual_old_packets=packets,all_primary_update_features_byte_exact=True,relative_diagnostic_lossless_float64_exact=True,saved_mask_owner_fixture=True,early_retirement_after_consumption=True,SAM2_forward_calls=0,new_native_rollouts=0,source_sha256={str(p):sha(p)for p in [Path(__file__),Path(__file__).with_name('pool.py'),Path(__file__).with_name('source.py')]})
 with(out/'CPU_episode_source_pool_checks.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print('MULTI_EVENT_SOURCE_PREFIX_QUEUE_AND_RETIREMENT_PASS',4,12,flush=True)
if __name__=='__main__':main()
