"""Clock-only readiness fixtures and live read-only NVML; no CUDA allocations."""
import ast,copy,json,time,hashlib
from pathlib import Path
from .worker import clock_pair
from . import metrics
from native_process_sampler_v2.cpu_checks import fixture
T=Path(__file__).resolve().parents[1];C=Path(__file__).resolve().parent;O=T/'outputs/native_process_sampler_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=T/'native_process_sampler_v2'
 def fn(p,name):return next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
 assert ast.dump(fn(C/'metrics.py','select'))==ast.dump(fn(old/'metrics.py','select'))
 assert ast.dump(ast.parse((C/'strict_guard.py').read_text().replace('native_process_sampler_v3','native_process_sampler_v2')))==ast.dump(ast.parse((old/'strict_guard.py').read_text()))
 def pair(extra):
  a,b=fixture()['frames'];early=copy.deepcopy(b);early['driver_timestamp_us']=a['driver_timestamp_us']+799999
  # First endpoint is invalid by ONE microsecond; extra clock-ready frame has identical strict acceptance bounds.
  early['raw_process_records'][0].update(SM=99,bandwidth=99)
  b['raw_process_records'][0].update(SM=extra,bandwidth=extra)
  b['host_timestamp_us']+=25000
  values=iter([a,early,b]);slept=[];frames,observed=clock_pair(lambda _:next(values),sleep=lambda v:slept.append(v))
  data=dict(fixture(),frames=frames);assert metrics.select(data,3)[0]=={123:extra};assert len(observed)==3 and slept==[1.05,.025]
  return frames
 pair(0);pair(100) # Endpoint chosen by clock identically for low/high utilization.
 a,b=fixture()['frames'];values=iter([a,b]);frames,observed=clock_pair(lambda _:next(values),sleep=lambda _:None);assert len(observed)==2
 # Existing selector rejects upper-bound violations, stale/future/missing/invalid component exactly as before.
 for mode in ('too_far','stale','future','bad_component'):
  d=fixture();last=d['frames'][1]
  if mode=='too_far':last['driver_timestamp_us']=11600001
  elif mode=='stale':last['host_timestamp_us']=11750001
  elif mode=='future':last['driver_timestamp_us']=11100000
  else:last['raw_process_records'][0]['SM']=None
  try:metrics.select(d,3)
  except ValueError:pass
  else:raise AssertionError(mode+' accepted')
 d=fixture();d['frames'][1]['raw_process_records']=[];d['frames'][1]['API_status']='no_process_sample_NOT_FOUND';assert metrics.select(d,3)[0]=={}
 d=fixture();d['frames'][1]['raw_process_records'][0]['timestamp_us']=10000000;assert metrics.select(d,3)[0]=={}
 a,b=fixture()['frames'];b['driver_timestamp_us']=10799999;b['host_timestamp_us']=11800000
 values=iter([a,b])
 try:clock_pair(lambda _:next(values),sleep=lambda _:None)
 except ValueError:pass
 else:raise AssertionError('Clock readiness deadline waived')
 actual=[]
 for i in range(20):
  values=metrics.process_utilization(5);d=metrics.diagnostics();assert d['collection_seconds']<3 and not d['missing_records_inferred_zero'];actual.append(d)
  assert {x['pid']:max(x['SM'],x['bandwidth'])for x in d['components']}==values
  print('READ_ONLY_NVML_CLOCK_PASS',i,flush=True)
 O.mkdir(exist_ok=True)
 report=dict(passed=True,original_selector_AST_exact=True,entire_guard_AST_exact_except_metric_import=True,one_microsecond_lower_edge_reproduced=True,first_clock_ready_endpoint_independent_of_utilization=True,all_original_bounds_missing_timeout_and_errors_retained=True,no_GPU_work_started=True,actual_read_only_NVML_collections=actual,source_sha256={str(p):sha(p)for p in C.glob('*')if p.suffix in('.py','.json')},parent_failure=str(T/'outputs/native_cross_task100_v15/batches/008/runs/c863551890ce4f3dbd54b3671a9c2b1b/failure.json'),parent_failure_raw_frames_not_available=True,actual_parent_failure_clock_delta_not_inferred=True)
 with(O/'CPU_clock_checks.json').open('x')as f:json.dump(report,f,indent=2)
 print('BOUNDED_CLOCK_READINESS_CPU_PASS',flush=True)
if __name__=='__main__':main()
