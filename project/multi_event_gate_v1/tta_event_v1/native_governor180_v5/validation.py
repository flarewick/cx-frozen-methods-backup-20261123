"""Pure CPU proof of account averaging order and unchanged raw/per-PID data."""
import ast,json,copy,os,hashlib
from pathlib import Path
from .observer import C,T,verify_reference
from .statistics import aggregate
from native_governor180_v4.statistics import aggregate as old_aggregate

def main():
 verify_reference()
 from .checks import checks
 assert len(checks)==39
 assert (C/'policy.py').read_text()==(T/'native_rule8_startup_v1/policy.py').read_text()
 source=(C/'statistics.py').read_text()
 for before,after in reversed(json.loads((C/'statistics_edits.json').read_text())):
  if after:assert source.count(after)==1;source=source.replace(after,before)
  else:
   anchor="        device = [dict(monotonic=h['monotonic']"
   assert source.count(anchor)==1;source=source.replace(anchor,before+anchor)
 assert source==(T/'native_governor180_v4/statistics.py').read_text()
 def functions(p):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
 a,b=functions(T.parent/'libero/resource_guard_v15/observer.py'),functions(C/'sampler.py')
 assert all(a[n]==b[n]for n in('proc','sample'))
 file=T/'outputs/native_governor180_v5/CPU_reference_raw_window_v1.json';rows=json.loads(file.read_text());assert len(rows)>=50
 raw=rows[-1];new=aggregate(copy.deepcopy(raw),copy.deepcopy(rows));old=old_aggregate(copy.deepcopy(raw),copy.deepcopy(rows));proof=[]
 for c,d in zip(new['cards'],old['cards'],strict=True):
  assert c['processes']==d['processes']and c['device_rolling180']==d['device_rolling180']and c['device_recent60']==d['device_recent60']
  x,y=c['own_rolling180'],d['own_rolling180']
  assert {k:v for k,v in x.items()if k not in('sm','bandwidth','util')}=={k:v for k,v in y.items()if k not in('sm','bandwidth','util')}
  for component in('sm','bandwidth'):
   own_mean=sum(sum((p.get(component)or 0)for p in r['cards'][c['index']]['processes']if p['uid']==os.getuid())for r in rows)/len(rows)
   assert x[component]==min(100.,own_mean,c['device_rolling180'][component])
   assert 0<=x[component]<=own_mean and x[component]<=c['device_rolling180'][component]
  assert x['util']==max(x['sm'],x['bandwidth'])
  proof.append(dict(GPU=c['index'],old_pointwise_minimum=y['util'],corrected_window_components=x['util'],device_mean=c['device_rolling180']['util']))
 # Deliberately out-of-phase component observations: window means must come first.
 synthetic=[]
 for i in range(60):
  p=dict(pid=1,birth=10,uid=os.getuid(),age=400+i*3,rss=1024,cpu=100,sm=0 if i%2==0 else 100,bandwidth=0,memory_mib=4096,metrics_available=True)
  card=dict(index=0,total_mib=24564,sm=100 if i%2==0 else 0,bandwidth=0,memory_mib=4096,processes=[p],own_pids=[1],available=False)
  synthetic.append(dict(monotonic=i*3,wall_time=0,cards=[card]))
 s=aggregate(copy.deepcopy(synthetic[-1]),synthetic);assert s['cards'][0]['own_rolling180']['util']==50
 for r in synthetic:r['cards'][0]['processes'][0]['uid']=os.getuid()+1;r['cards'][0]['own_pids']=[]
 assert aggregate(copy.deepcopy(synthetic[-1]),synthetic)['cards'][0]['own_rolling180']['util']==0
 for r in synthetic:r['cards'][0]['processes'][0].update(uid=os.getuid(),sm=None,bandwidth=None,metrics_available=False);r['cards'][0]['own_pids']=[1]
 z=aggregate(copy.deepcopy(synthetic[-1]),synthetic)['cards'][0]['own_rolling180'];assert z['util']==0 and z['metric_gap_seconds']==999
 from .sampler import replace,record_collection_error
 from .observer import read_current
 fixture=T/'outputs'/C.name/'CPU_error_fixture';fixture.mkdir(exist_ok=False)
 initial=dict(wall_time=10.,schema='CPU_valid_original_timestamp');replace(fixture/'current.json',initial);before=(fixture/'current.json').read_bytes();stamp=(fixture/'current.json').stat().st_mtime_ns
 record_collection_error(TimeoutError('injected'),root=fixture);assert(fixture/'current.json').read_bytes()==before and(fixture/'current.json').stat().st_mtime_ns==stamp
 assert read_current(fixture/'current.json',now=17.99)==initial
 try:read_current(fixture/'current.json',now=18.001)
 except RuntimeError:pass
 else:raise AssertionError('Stale accepted')
 value=dict(passed=True,no_GPU_work=True,raw_proc_sample_AST_original_exact=True,per_PID_and_device_statistics_byte_equal=True,account_component_window_mean_then_cap_then_MAX=True,no_foreign_or_missing_load_credit=True,original39_policy_checks=checks,Rule8_creation_grace_policy_exact=True,raw_window_comparison=proof,input_sha256={str(file):hashlib.sha256(file.read_bytes()).hexdigest()},source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in C.glob('*')if p.is_file()})
 (T/'outputs'/C.name/'CPU_checks_v1.json').write_text(json.dumps(value,indent=2)+'\n');print('ACCOUNT_WINDOW_MEANS_CPU_PASS',json.dumps(proof),flush=True)
if __name__=='__main__':main()
