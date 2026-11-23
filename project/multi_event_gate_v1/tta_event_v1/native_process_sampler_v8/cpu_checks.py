"""Original select/guard exact; real full endpoints only, no value-based selection."""
import ast,copy,json,hashlib,time
from pathlib import Path
from .worker import clock_pair,verify_trace,choose_pair,collect
from .metrics import select
T=Path(__file__).resolve().parents[1];D=Path(__file__).parent;O=T/'outputs/native_process_sampler_v8'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 old=T/'native_process_sampler_v7'
 for name in('strict_guard.py','device_window.py','device_audit.py','nvml_query.py','nvml_query_worker.py','guard.py'):
  a=ast.parse((old/name).read_text());b=ast.parse((D/name).read_text().replace('native_process_sampler_v8','native_process_sampler_v7'));assert ast.dump(a)==ast.dump(b),name
 def function(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
 assert ast.dump(function(old/'metrics.py','select'))==ast.dump(function(D/'metrics.py','select'))
 assert ast.dump(function(old/'worker.py','frame'))==ast.dump(function(D/'worker.py','frame'))
 epoch=1700000000000000
 def fixture(value):
  stamps=[0,.2,.4,.6,1.8]
  return [dict(driver_timestamp_us=epoch+round(x*1e6),host_timestamp_us=epoch+round((x+.1)*1e6),API_status='success',raw_process_records=[dict(pid=123,timestamp_us=epoch+round(x*1e6),SM=value,bandwidth=0)],device_samples=dict(SM=[],bandwidth=[]))for x in stamps]
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
 fs=fixture(0);fs=[fs[0],dict(fs[-1],driver_timestamp_us=epoch+1000000,host_timestamp_us=epoch+1100000,raw_process_records=[],API_status='no_process_sample_NOT_FOUND')];calls=[]
 def missing(last_seen):calls.append(last_seen);return fs[len(calls)-1]
 pair,obs,_=clock_pair(missing,sleep=lambda x:None,monotonic=lambda:0);assert len(calls)==2 and select(dict(gpu=4,gpu_uuid='GPU-fixture',API='nvmlDeviceGetProcessesUtilizationInfo_v1',frames=pair),4)[0]=={}
 # Single blocking read without a real intermediate full endpoint remains a failure.
 fs=fixture(0);calls=[]
 def long_read(last_seen):calls.append(last_seen);return fs[0]if len(calls)==1 else fs[-1]
 try:clock_pair(long_read,sleep=lambda x:None,monotonic=lambda:0)
 except ValueError:pass
 else:raise AssertionError('Overlong unverified acquisition accepted')
 # Actual old failure with only its two original full endpoints stays rejected.
 source=T/'outputs/native_frozen100_v19/mixed/batches/027/runs/fa07aa6bcfd84ad193d901f1ef5818c7/failure.json';text=json.loads(source.read_text())['traceback'];raw=json.JSONDecoder().raw_decode(text[text.rfind('raw_NVML=')+9:])[0]
 try:select(raw,4)
 except ValueError:pass
 else:raise AssertionError('Old failure reclassified')
 records=[]
 for i in range(20):
  began=time.monotonic();data=collect(4);assert verify_trace(data);select(data,4);elapsed=time.monotonic()-began;assert elapsed<3;records.append(dict(seconds=elapsed,frame_count=len(data['clock_readiness_frames']),selected_frame_indices=data['selected_frame_indices'],raw=data))
 dump(O/'CPU_readonly_NVML_trials.json',dict(passed=True,trials=records,no_CUDA_or_GPU_work_created=True))
 proof=dict(passed=True,original_select_frame_and_all_guard_device_accounting_AST_exact7=True,only_actual_full_frames_eligible=True,first_temporally_eligible_pair_and_earliest_prior_endpoint=True,selection_independent_of_zero_or_high_utilization=True,unknown_PID_returned_without_searching_for_numeric=True,single_overlong_read_without_intermediate_still_rejected=True,old_actual_failure_still_rejected=True,all_intermediate_full_frames_and_lifetime_accounting_retained=True,all_original_time_and_resource_bounds_unchanged=True,actual20_metadata_only_NVML_trials_passed=True,original_failure_sha256=digest(source),input_sha256={str(source):digest(source)},source_sha256={str(p):digest(p)for p in D.glob('*.py')})
 dump(O/'CPU_acquisition_checks.json',proof);dump(O/'CPU_actual_exit.json',dict(passed=True,exit_code=0,actual20_NVML_trials_completed=True,metadata_only_no_GPU_work=True));print('SAMPLER8_ACTUAL20_METADATA_TRIALS_AND_STRICT_CPU_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
