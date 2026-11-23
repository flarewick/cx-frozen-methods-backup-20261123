"""Read-only independent numerical/source/input/clock/RNG audit of real GPU proof."""
import os
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import hashlib,json
import numpy as np
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1]
O=T/'outputs/native_nextchunk_v1/sampler_gpu_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rawpath(raw):
 for prefix in ('/data/qiufw/cx_migration/cx_new_direction_guide_TTA/','/home/qiufw/cx_migration/cx_new_direction_guide_TTA/'):
  if raw.startswith(prefix):return P/raw[len(prefix):]
 return Path(raw)
def load(p):
 with np.load(p,allow_pickle=False) as z:return{k:z[k].copy() for k in z.files}
def imagehash(a):
 a=np.ascontiguousarray(a)
 h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def run():
 spec=read(O/'protocol.json');latest=read(O/'latest_complete.json');root=rawpath(latest['root'])
 report=read(root/'report.json');checks=0
 assert latest['host_exit_code']==0 and report['complete'] and report['actual_full_VLA_weights_used'];checks+=3
 assert report['protocol_identity']==latest['protocol_identity']==spec['identity']
 for group in ('source_sha256','input_sha256'):
  for raw,h in spec[group].items():assert sha(rawpath(raw))==h;checks+=1
 total=0
 for j,job in enumerate(spec['jobs']):
  case=root/('case%02d'%j);ref=load(case/'unmodified_reference.npz');obs=load(case/'witnessed_same_path.npz')
  a=read(case/'unmodified_reference.json');b=read(case/'witnessed_same_path.json')
  assert set(ref)==set(obs);checks+=1
  for k in ref:
   assert ref[k].dtype==obs[k].dtype and ref[k].shape==obs[k].shape
   assert ref[k].tobytes()==obs[k].tobytes(),(j,k)
   assert np.isfinite(ref[k]).all();checks+=3
  assert sha(case/'unmodified_reference.npz')==a['file_sha256'] and sha(case/'witnessed_same_path.npz')==b['file_sha256'];checks+=2
  n=job['queries'];assert ref['base'].shape==(n,32,14) and ref['raw_delta'].shape==(n,32,14)
  assert not ref['raw_delta'][:,:,[6,13]].any()
  assert float(np.max(abs(ref['raw_delta'])))<=.100001
  assert ref['legal_base'].tobytes()==ref['base'].tobytes();checks+=4
  history=load(job['history']);clock=read(job['clock'])['query_observation_ticks']
  assert len(a['trace'])==len(b['trace'])==len(clock)==n
  for q,(left,right) in enumerate(zip(a['trace'],b['trace'])):
   assert left['query']==right['query']==q and left['tick']==right['tick']==clock[q]
   assert left['before']==right['before'] and left['after']==right['after'];checks+=3
   imgs={str(c):imagehash(history['cam'+str(c)][q]) for c in range(3)}
   assert imgs==left['actor_images']==right['actor_images'];checks+=1
   witness=right['noise_witness']
   assert witness['shape']==[1,32,32] and witness['device_type']=='cuda'
   assert witness['before']!=witness['after']
   assert witness['environment_before']==witness['environment_after']
   assert witness['base_sha256']==hashlib.sha256(ref['base'][q].tobytes()).hexdigest()
   assert witness['rng_before']==left['before'] and witness['rng_after']==left['after']
   assert right['cached_context_not_executed']is True;checks+=6
  total+=n
 assert total==149 and report['base_infer_calls']==298 and len(spec['jobs'])==29;checks+=3
 exitproof=read(root/'barrier/normal_child_exit.json')
 assert exitproof['normal_exit'] and exitproof['exit_code']==exitproof['wait_status']==0
 assert exitproof['proc_absent_after_wait'] and not exitproof['cpu_only_fixture'];checks+=2
 telemetry=[json.loads(line) for line in (root/'resources.jsonl').read_text().splitlines()]
 samples=[x for x in telemetry if 'seconds'in x]
 assert not any('stopped'in x for x in telemetry)
 assert telemetry[-1]['own_gpu_released'] and not telemetry[-1]['remaining_own_gpu_pids'];checks+=2
 assert max(x['account_gpu_count'] for x in samples)<=3
 assert max(x['memory_mib'] for x in samples)<=22528
 assert max(max(x['rss_bytes'].values(),default=0) for x in samples)<128*2**30;checks+=3
 long=[x for x in samples if x['seconds']>=280]
 assert not long
 accepted=[x for x in samples if x.get('normal_child_exit_proof',{}).get('accepted')]
 result=dict(passed=True,checks=checks,old_cached_cases=29,cached_queries=149,
   actual_base_infer_calls=298,current_RGB_hashes_verified=True,all_base_delta_legal_arrays_exact=True,
   all_reference_witness_RNG_hashes_exact=True,noise_shape=[1,32,32],internal_noise_draws_per_witnessed_query=1,
   account_peak_cards=max(x['account_gpu_count'] for x in samples),
   peak_VRAM_MiB=max(x['memory_mib'] for x in samples),
   peak_GPU_worker_RSS_GiB=max(max(x['rss_bytes'].values(),default=0) for x in samples)/2**30,
   GPU_release_seconds=telemetry[-1]['cleanup_seconds_after_start'],
   naturally_completed_below5min=True,long_running_resource_qualification=False,
   normal_child_certificate_emitted=True,new_exit_exception_actually_triggered_samples=len(accepted),
   source_sha256=sha(Path(__file__)),online_updates=0,TTA_utility_proven=False,
   scope='Full actual VLA currentRGB sampler interface only, alreadycached observations never generated by the tested newbases. No simulator, newnative trajectory, success benefit or independent confirmation. Run naturally ended150s; perPID mean~27%, so no long40 qualification claim. Normalchild marker/release proven, added retirement exception not exercised if accepted samplecount0; CPU race tests are separate.')
 out=O/'independent_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
 print('REAL_VLA_SAMPLER_INDEPENDENT_AUDIT',checks,'149currentRGB_NO_TTA_CLAIM',flush=True)
if __name__=='__main__':run()
