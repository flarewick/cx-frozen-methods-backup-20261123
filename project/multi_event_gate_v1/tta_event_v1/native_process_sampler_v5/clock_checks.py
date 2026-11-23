"""Replay actual overlong-window evidence; short-poll only clock readiness."""
import ast,copy,json,hashlib,math
from pathlib import Path
from .worker import clock_pair
from . import metrics
from native_process_sampler_v3.worker import clock_pair as fixed_wait_pair
from native_process_sampler_v2.cpu_checks import fixture
T=Path(__file__).resolve().parents[1];C=Path(__file__).resolve().parent;O=T/'outputs/native_process_sampler_v5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=T/'native_process_sampler_v2'
 def fn(p,name):return next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
 assert ast.dump(fn(C/'metrics.py','select'))==ast.dump(fn(old/'metrics.py','select'))
 assert ast.dump(ast.parse((C/'strict_guard.py').read_text().replace('native_process_sampler_v5','native_process_sampler_v2')))==ast.dump(ast.parse((old/'strict_guard.py').read_text()))
 failure=T/'outputs/native_cross_task100_v16/batches/008/runs/8ff4543be51147699137877209bd56f5/failure.json'
 raw=json.loads(json.loads(failure.read_text())['traceback'].rsplit('; raw_NVML=',1)[1].strip());a,b=raw['frames']
 assert b['driver_timestamp_us']-a['driver_timestamp_us']==1799205 and b['host_timestamp_us']-a['host_timestamp_us']==1801294
 try:metrics.select(raw,5)
 except ValueError:pass
 else:raise AssertionError('Actual overlong window wrongly accepted')
 def simulated(method,value=0,read_pause=0):
  wall=[10050000];calls=[0];sleeps=[]
  def sleep(seconds):
   sleeps.append(seconds);wall[0]+=int(round(seconds*1e6))+(750000 if seconds==1.05 else 0)
  def frame(_):
   calls[0]+=1
   if calls[0]==2:wall[0]+=read_pause
   stamp=(wall[0]//200000)*200000
   return dict(driver_timestamp_us=stamp,host_timestamp_us=wall[0],API_status='success',raw_process_records=[dict(pid=123,timestamp_us=stamp,SM=value,bandwidth=0)])
  frames,observed=method(frame,sleep=sleep,monotonic=lambda:wall[0]/1e6)
  return dict(fixture(),frames=frames),observed,sleeps
 bad,_,_=simulated(fixed_wait_pair)
 try:metrics.select(bad,3)
 except ValueError:pass
 else:raise AssertionError('Fixed-wait overrun not reproduced')
 for value in [0,100]:
  d,observed,sleeps=simulated(clock_pair,value);assert metrics.select(d,3)[0]=={123:value};assert all(s==.05 for s in sleeps)and len(observed)>2
  d,_,_=simulated(clock_pair,value,750000);assert metrics.select(d,3)[0]=={123:value}
 # A pause so long that one actual read exceeds the original limits MUST still fail.
 d,_,_=simulated(clock_pair,50,1800000)
 try:metrics.select(d,3)
 except ValueError:pass
 else:raise AssertionError('Long-read upper limit waived')
 for mode in ('too_far','stale','future','bad_component'):
  d=fixture();last=d['frames'][1]
  if mode=='too_far':last['driver_timestamp_us']=11600001
  elif mode=='stale':last['host_timestamp_us']=11750001
  elif mode=='future':last['driver_timestamp_us']=11100000
  else:last['raw_process_records'][0]['SM']=None
  try:metrics.select(d,3)
  except ValueError:pass
  else:raise AssertionError(mode+' accepted')
 d=fixture();d['frames'][1].update(raw_process_records=[],API_status='no_process_sample_NOT_FOUND');assert metrics.select(d,3)[0]=={}
 d=fixture();d['frames'][1]['raw_process_records'][0]['timestamp_us']=10000000;assert metrics.select(d,3)[0]=={}
 actual=[]
 for i in range(20):
  values=metrics.process_utilization(5);d=metrics.diagnostics();assert d['collection_seconds']<3 and not d['missing_records_inferred_zero'];actual.append(d)
  assert {x['pid']:max(x['SM'],x['bandwidth'])for x in d['components']}==values
  print('READ_ONLY_SHORT_CLOCK_PASS',i,flush=True)
 O.mkdir(exist_ok=True);report=dict(passed=True,original_selector_AST_exact=True,entire_guard_AST_exact_except_metric_import=True,actual_failed_raw_frames_replayed=True,actual_failed_driver_delta_us=1799205,actual_failed_host_delta_us=1801294,fixed_sleep_overrun_reproduced=True,first_endpoint_independent_of_high_low_utilization=True,read_pause_750ms_accepted_only_with_original_valid_interval=True,overlong_1800ms_read_still_rejected=True,all_original_bounds_missing_timeout_errors_retained=True,no_GPU_work_started=True,actual_read_only_NVML_collections=actual,source_sha256={str(p):sha(p)for p in C.glob('*')if p.suffix in('.py','.json')},actual_failure_sha256={str(failure):sha(failure)})
 with(O/'CPU_clock_checks.json').open('x')as f:json.dump(report,f,indent=2)
 print('SHORT_CLOCK_READINESS_CPU_PASS',flush=True)
if __name__=='__main__':main()
