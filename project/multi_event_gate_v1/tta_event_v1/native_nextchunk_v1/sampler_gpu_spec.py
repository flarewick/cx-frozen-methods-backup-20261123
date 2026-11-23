"""Freeze a whole-cache real-VLA sampler test, never simulator/task evaluation."""
from pathlib import Path
import hashlib,json
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1]
O=T/'outputs/native_nextchunk_v1/sampler_gpu_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(x,indent=2,allow_nan=False)+'\n'
 if p.exists():assert p.read_text()==text,p
 else:p.write_text(text)
def run():
 audit=read(T/'outputs/native_nextchunk_v1/actor_input_audit.json')
 assert audit['source_cases']==29 and audit['differing_later_queries']==120
 R=T/'outputs/native_stage_v1/collections/5d5369942959181f58a604c4';jobs=[];inputs={}
 old=read(T/'outputs/native_stage_v1/protocol.json')
 for f in sorted((R/'sessions').rglob('case.json')):
  case=read(f)
  if case.get('role')!='development' or (f.parent/'unavailable.json').exists():continue
  meta=read(f.parent/'discovery.json');scene=next(s for s in old['scenes'][6:] if s['case_id']==case['case_id'])
  history=f.parent/'observed_history.npz';clock=f.parent/'prefix_clock.json'
  jobs.append(dict(case_id=case['case_id'],history=str(history),clock=str(clock),instruction=scene['instruction'],queries=meta['sampler_calls']))
  for p in (history,clock):inputs[str(p)]=sha(p)
 assert len(jobs)==29 and sum(j['queries'] for j in jobs)==149
 proofs=[T/'outputs/native_nextchunk_v1'/n for n in ('sampler_cpu_checks.json','current_sampler_cpu_checks.json','current_query_cpu_checks.json','exit_cpu_checks.json','guard_cpu_checks.json','guard_source_audit.json','actor_input_audit.json')]
 for p in proofs:
  evidence=read(p);assert evidence.get('passed',evidence.get('complete',False));inputs[str(p)]=sha(p)
  for raw,digest in evidence.get('source_sha256',{}).items():
   fixed=raw
   for prefix in ('/data/qiufw/cx_migration/cx_new_direction_guide_TTA/','/home/qiufw/cx_migration/cx_new_direction_guide_TTA/'):
    if raw.startswith(prefix):fixed=str(P/raw[len(prefix):]);break
   assert sha(fixed)==digest,('CPU proof source changed',p,raw)
 guardproof=read(T/'outputs/native_nextchunk_v1/guard_source_audit.json')
 assert guardproof['patched_sha256']==sha(C/'strict_guard.py')
 assert guardproof['parent_sha256']==sha(T/'resource_guard_v1/guard.py')
 for p in (Path(old['projection']),Path(old['training'])/'adapter_seed0.pt'):inputs[str(p)]=sha(p)
 sources=[C/n for n in ('sampler_gpu_spec.py','sampler_gpu_entry.py','sampler_gpu_work.py','sampler_gpu_host.py','sampler_gpu.sh','sampler_fork.py','query_ledger.py','query_current.py','child_exit.py','strict_guard.py')]
 sources += [T/'native_stage_v1/cx_tta_stage'/n for n in ('query.py','views.py')]
 sources += [P/'deployment'/n for n in ('adapter_cache_cpu.py','startup_loading.py','concurrent_rng.py','model_service.py')]
 spec=dict(version=1,jobs=jobs,cases=29,cached_queries=149,passes=('unmodified_reference','witnessed_same_path'),seed=20261027,projection=old['projection'],training=old['training'],
  source_sha256={str(p):sha(p) for p in sources},input_sha256=inputs,
  scope='All29 oldcached histories. One real VLA infer(noise=None) per cached input per controlled pass; currentRGB/frozenadapter/FK and noise witness. RGB features encoded once and reused exactly. No simulator execution, no claim new bases generated cached laterRGB. No full-policy/nativecontinuation/actorupdate/tasksuccess, no newscene/100evaluation.',
  policy_training_allowed=False,simulator_started=False,online_updates=0,TTA_utility_proven=False)
 spec['identity']=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
 write(O/'protocol.json',spec);print('SAMPLER_REAL_GPU_SPEC',spec['identity'],29,149,flush=True);return spec
def verify(spec):
 for group in ('source_sha256','input_sha256'):
  for raw,h in spec[group].items():assert sha(raw)==h,raw
if __name__=='__main__':run()
