"""No GPU allocations; real driver acquisition and fail-closed negative checks."""
import ast,copy,json,hashlib,subprocess
from pathlib import Path
from unittest.mock import patch
from . import metrics
T=Path(__file__).resolve().parents[1];C=Path(__file__).resolve().parent;O=T/'outputs/native_process_sampler_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(sm=0,bw=0):
 return dict(gpu=3,gpu_uuid='GPU-test',API='nvmlDeviceGetProcessesUtilizationInfo_v1',frames=[dict(driver_timestamp_us=10000000,host_timestamp_us=10050000,API_status='success',raw_process_records=[dict(pid=123,timestamp_us=10000000,SM=99,bandwidth=99)]),dict(driver_timestamp_us=11000000,host_timestamp_us=11050000,API_status='success',raw_process_records=[dict(pid=123,timestamp_us=11000000,SM=sm,bandwidth=bw)])])
def main():
 old=ast.parse((T/'native_process_sampler_v1/strict_guard.py').read_text());new=ast.parse((C/'strict_guard.py').read_text())
 for node in ast.walk(new):
  if isinstance(node,ast.ImportFrom)and node.module=='native_process_sampler_v3.metrics':node.module='native_process_sampler_v1.metrics'
 assert ast.dump(old)==ast.dump(new)
 assert metrics.select(fixture(),3)[0]=={123:0}
 assert metrics.select(fixture(2,1),3)[0]=={123:2}
 f=fixture();f['frames'][1].update(raw_process_records=[],API_status='no_process_sample_NOT_FOUND');assert metrics.select(f,3)[0]=={},'NotFound MUST NOT imply zero'
 negative=[]
 for field,value in [('SM',None),('SM',-1),('bandwidth',101),('SM',4294967295),('timestamp_us',11060000),('pid',0)]:
  f=fixture();f['frames'][1]['raw_process_records'][0][field]=value;negative.append(f)
 f=fixture();f['frames'][1]['raw_process_records'][0]['timestamp_us']=10000000;assert metrics.select(f,3)[0]=={},'Stale record MUST remain unavailable'
 f=fixture();f['frames'][1]['driver_timestamp_us']=10000000;negative.append(f)
 f=fixture();f['frames'][1]['host_timestamp_us']=12000000;negative.append(f)
 f=fixture();f['frames'][1]['raw_process_records']*=2;negative.append(f)
 f=fixture();f['gpu']=5;negative.append(f)
 f=fixture();f['frames'][1]['API_status']='error';negative.append(f)
 for f in negative:
  try:metrics.select(f,3)
  except ValueError:pass
  else:raise AssertionError('Invalid sample accepted')
 for error in(subprocess.TimeoutExpired('nvml',3),subprocess.CalledProcessError(1,'nvml')):
  with patch.object(metrics.subprocess,'run',side_effect=error)as mocked:
   try:metrics.process_utilization(0)
   except type(error):pass
   else:raise AssertionError('NVML failure suppressed')
   assert mocked.call_count==1 and mocked.call_args.kwargs['timeout']==3
 actual=[]
 for i in range(8):
  gpu=(0,3,5,6)[i%4];values=metrics.process_utilization(gpu);d=metrics.diagnostics();assert d['collection_seconds']<3 and not d['missing_records_inferred_zero'];actual.append(d)
  assert {r['pid']:max(r['SM'],r['bandwidth'])for r in d['components']}==values
 probe=json.loads((O/'raw_NVML_v1_pmon_probe.json').read_text())
 witness=json.loads((O/'explicit_zero_driver_witness.json').read_text());assert witness['SM']==witness['bandwidth']==0
 zeros=[witness]+[r for d in actual for r in d['components']if r['SM']==r['bandwidth']==0]+[r for d in probe for r in d['NVML_v1']if r.get('SM')==r.get('bandwidth')==0]
 assert zeros,'Need actual explicit numeric zero records before scientific resume'
 report=dict(passed=True,all_strict_guard_AST_unchanged_except_metric_import=True,same_three_second_timeout=True,final_fresh_driver_timestamp_only=True,explicit_numeric_zeros_observed=zeros,missing_NotFound_stays_unavailable=True,unknown_stale_error_timeout_stops=True,negative_checks=len(negative),actual_read_only_NVML_collections=actual,original_guard_SHA256=sha(T/'native_process_sampler_v1/strict_guard.py'),source_sha256={str(p):sha(p)for p in C.rglob('*')if p.is_file()and p.suffix in('.py','.json')},no_GPU_work_started=True)
 with(O/'CPU_direct_NVML_checks.json').open('x')as f:json.dump(report,f,indent=2)
 print('DIRECT_NVML_V1_FINAL_INTERVAL_GUARD_PASS',len(actual),len(zeros),flush=True)
if __name__=='__main__':main()
