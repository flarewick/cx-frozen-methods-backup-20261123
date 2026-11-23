"""Original select/guard exact; real full endpoints only, no value-based selection."""
import ast,copy,json,hashlib,time
from pathlib import Path
from .worker import clock_pair,verify_trace,choose_pair,collect
from .metrics import select
T=Path(__file__).resolve().parents[1];D=Path(__file__).parent;O=T/'outputs/native_process_sampler_v13'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 old=T/'native_process_sampler_v7'
 for name in('strict_guard.py','device_audit.py','nvml_query.py','nvml_query_worker.py'):
  a=ast.parse((old/name).read_text());b=ast.parse((D/name).read_text().replace('native_process_sampler_v13','native_process_sampler_v7'));assert ast.dump(a)==ast.dump(b),name
 current=(D/'guard.py').read_text().replace('native_process_sampler_v13','native_process_sampler_v7')
 current=current.replace("    from .metrics import sampling_session\n    with sampling_session(kwargs['gpu'],Path(report).with_name('NVML_stream_actual_exit.json')):\n        return implementation.guarded_command(\n            *args, min_cumulative_utilization=MIN_GPU_UTILIZATION, max_memory_mib=MAX_GPU_MEMORY_MIB, **kwargs)","    return implementation.guarded_command(\n        *args, min_cumulative_utilization=MIN_GPU_UTILIZATION, max_memory_mib=MAX_GPU_MEMORY_MIB, **kwargs)")
 assert ast.dump(ast.parse(current))==ast.dump(ast.parse((old/'guard.py').read_text()))
 def function(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
 assert ast.dump(function(old/'metrics.py','select'))==ast.dump(function(D/'metrics.py','select'))
 node=function(D/'worker.py','frame');node.body=node.body[2:6]+node.body[:2]+node.body[6:];assert ast.dump(function(old/'worker.py','frame'))==ast.dump(node),'Only bandwidth read and validation moved before critical process+SM reads'
 epoch=1700000000000000
 def fixture(value):
  stamps=[0,.2,.4,.6,1.8]
  return [dict(driver_timestamp_us=epoch+round(x*1e6),host_timestamp_us=epoch+round((x+.1)*1e6),API_status='success',raw_process_records=[dict(pid=123,timestamp_us=epoch+round(x*1e6),SM=value,bandwidth=0)],device_samples={k:[dict(timestamp_us=epoch+round(x*1e6),value=value)]for k in('SM','bandwidth')})for x in stamps]
 choices=[]
 for value in(0,99):
  data=fixture(value);calls=[]
  def read(last_seen):calls.append(last_seen);return data[len(calls)-1]
  pair,observed,clocks=clock_pair(read,sleep=lambda seconds:None,monotonic=lambda:0)
  assert pair==[data[1],data[4]]and len(calls)==5 and observed==data and not clocks
  packet=dict(gpu=4,gpu_uuid='GPU-fixture',API='nvmlDeviceGetProcessesUtilizationInfo_v1',frames=pair,clock_readiness_frames=observed,selected_frame_indices=[1,4]);assert verify_trace(packet)
  assert select(packet,4)[0]=={123:value};choices.append(packet['selected_frame_indices'])
 assert choices[0]==choices[1]
 # A valid first pair with absent PID is returned immediately, not retried for a value.
 fs=fixture(0);fs=[fs[0],dict(fs[-1],driver_timestamp_us=epoch+1000000,host_timestamp_us=epoch+1100000,raw_process_records=[],API_status='no_process_sample_NOT_FOUND',device_samples={k:[dict(timestamp_us=epoch+1000000,value=0)]for k in('SM','bandwidth')})];calls=[]
 def missing(last_seen):calls.append(last_seen);return fs[len(calls)-1]
 pair,obs,_=clock_pair(missing,sleep=lambda x:None,monotonic=lambda:0);assert len(calls)==2 and select(dict(gpu=4,gpu_uuid='GPU-fixture',API='nvmlDeviceGetProcessesUtilizationInfo_v1',frames=pair),4)[0]=={}
 # Single blocking read without a real intermediate full endpoint remains a failure.
 fs=fixture(0);fs[-1]=dict(fs[-1],driver_timestamp_us=epoch+3000000,host_timestamp_us=epoch+3100000);calls=[]
 def long_read(last_seen):calls.append(last_seen);return fs[0]if len(calls)==1 else fs[-1]
 try:clock_pair(long_read,sleep=lambda x:None,monotonic=lambda:0)
 except ValueError:pass
 else:raise AssertionError('Overlong unverified acquisition accepted')
 # Actual old failure with only its two original full endpoints stays rejected.
 source=T/'outputs/native_frozen100_v19/mixed/batches/027/runs/fa07aa6bcfd84ad193d901f1ef5818c7/failure.json';text=json.loads(source.read_text())['traceback'];raw=json.JSONDecoder().raw_decode(text[text.rfind('raw_NVML=')+9:])[0]
 try:select(raw,4)
 except ValueError:pass
 else:raise AssertionError('Old failure reclassified')
 from types import SimpleNamespace
 from . import worker,metrics
 import native_process_sampler_v7.worker as original_worker
 class FakeNVML:
  NVML_GPU_UTILIZATION_SAMPLES=1;NVML_MEMORY_UTILIZATION_SAMPLES=2;NVML_VALUE_TYPE_UNSIGNED_INT=3;NVMLError_NotFound=type('NotFound',(Exception,),{})
  def __init__(self):self.now=epoch
  def nvmlDeviceGetProcessesUtilizationInfo(self,h,last):return[SimpleNamespace(pid=123,timeStamp=self.now,smUtil=49,memUtil=3)]
  def nvmlDeviceGetSamples(self,h,kind,last):
   if kind==2:self.now+=850000
   return 3,[SimpleNamespace(timeStamp=self.now,sampleValue=SimpleNamespace(uiVal=0))]
 outcomes=[]
 for module in(original_worker,worker):
  nvml,clock=module.n,module.time;fake=FakeNVML();module.n=fake;module.time=SimpleNamespace(time_ns=lambda:fake.now*1000)
  try:outcomes.append(module.frame(object(),0))
  finally:module.n=nvml;module.time=clock
 assert outcomes[0]['host_timestamp_us']-outcomes[0]['driver_timestamp_us']>750000
 assert outcomes[1]['host_timestamp_us']-outcomes[1]['driver_timestamp_us']==0 and worker.device_time_fresh(outcomes[1])
 assert outcomes[0]['raw_process_records'][0]['SM']==outcomes[1]['raw_process_records'][0]['SM']==49
 # Missing stderr was the blocker: actual nonzero/timeout evidence must survive.
 from unittest.mock import patch
 import subprocess
 with patch.object(metrics.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='actual_full_frame_evidence')):
  try:metrics.process_utilization(4)
  except RuntimeError as exc:assert 'actual_full_frame_evidence'in str(exc)
  else:raise AssertionError('Worker failure hidden')
 with patch.object(metrics.subprocess,'run',side_effect=subprocess.TimeoutExpired('fixture',3,stderr=b'actual_frame_before_timeout')):
  try:metrics.process_utilization(4)
  except RuntimeError as exc:assert 'actual_frame_before_timeout'in str(exc)
  else:raise AssertionError('Original3second timeout hidden')
 records=[]
 from .metrics import sampling_session,diagnostics
 with sampling_session(4,O/'CPU_real_metadata_worker_exit.json')as client:
  for i in range(20):
   began=time.monotonic();values=metrics.process_utilization(4);record=diagnostics();data=record['raw_NVML'];assert verify_trace(data);assert values==select(data,4)[0];elapsed=time.monotonic()-began;assert elapsed<3
   hi=max({x['timestamp_us']for x in data['frames'][-1]['device_samples']['SM']}&{x['timestamp_us']for x in data['frames'][-1]['device_samples']['bandwidth']})
   age=record['unix_ns']//1000-hi
   if not 0<=age<=750000:raise AssertionError('Actual response stale at parent: '+str(age))
   records.append(dict(seconds=elapsed,frame_count=len(data['clock_readiness_frames']),selected_frame_indices=data['selected_frame_indices'],parent_native_sample_age_us=age,transport=record['transport'],raw=data))
 assert len({r['transport']['worker_pid']for r in records})==1
 assert [r['transport']['request_sequence']for r in records]==list(range(1,21))
 dump(O/'CPU_readonly_NVML_trials.json',dict(passed=True,trials=records,no_CUDA_or_GPU_work_created=True))
 proof=dict(passed=True,original_select_strict_guard_and_replay_AST_exact7=True,DeviceWindow_accounting_changed_explicitly=True,frame_only_bandwidth_read_order_changed_with_original_operations_AST_verified=True,bandwidth_delay_fixture_preserves_critical_clock_freshness=True,nonzero_and3second_timeout_worker_stderr_evidence_retained=True,only_actual_full_frames_eligible=True,first_temporally_eligible_pair_and_earliest_prior_endpoint=True,selection_independent_of_zero_or_high_utilization=True,unknown_PID_returned_without_searching_for_numeric=True,single_overlong_read_without_intermediate_still_rejected=True,old_actual_failure_still_rejected=True,all_intermediate_full_frames_and_lifetime_accounting_retained=True,all_original_time_and_resource_bounds_unchanged=True,actual20_metadata_only_NVML_trials_passed=True,original_failure_sha256=digest(source),input_sha256={str(source):digest(source)},source_sha256={str(p):digest(p)for p in D.glob('*.py')})
 dump(O/'CPU_acquisition_checks.json',proof);dump(O/'CPU_check_internal_completion.json',dict(passed=True,exit_code=0,actual20_NVML_trials_completed=True,metadata_only_no_GPU_work=True));print('SAMPLER8_ACTUAL20_METADATA_TRIALS_AND_STRICT_CPU_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
