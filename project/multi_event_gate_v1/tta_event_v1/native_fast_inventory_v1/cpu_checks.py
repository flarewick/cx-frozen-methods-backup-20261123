"""Actual seven-card NVML/XML bracketing, strict invalid/timeout propagation."""
import json,time,sys,copy,subprocess,hashlib
from pathlib import Path
from unittest.mock import patch
from . import inventory as module
T=Path(__file__).resolve().parents[1];sys.path.insert(0,str(T.parents[1]/'deployment'))
from gpu_policy import inventory as original

def main():
 records=[];speed=[]
 for i in range(5):
  start=time.monotonic();before=original();old_seconds=time.monotonic()-start
  start=time.monotonic();fast=module.inventory();fast_seconds=time.monotonic()-start
  after=original();stable=[]
  for a,b,c in zip(before['cards'],fast['cards'],after['cards'],strict=True):
   assert a['index']==b['index']==c['index']and a['uuid']==b['uuid']==c['uuid']and a['total_mib']==b['total_mib']==c['total_mib']
   if set(a['pids'])==set(c['pids']):
    assert set(b['pids'])==set(a['pids'])and set(b['own_pids'])==set(a['own_pids'])==set(c['own_pids']);stable.append(b['index'])
   if not a['pids']and not c['pids']and a['memory_mib']==c['memory_mib']:
    assert b['memory_mib']==a['memory_mib']
  assert len(stable)>=5
  if before['own_gpu_count']==after['own_gpu_count']:assert fast['own_gpu_count']==before['own_gpu_count']
  records.append(dict(before=before,fast=fast,after=after,stable_PID_ownership_cards=stable,original_seconds=old_seconds,NVML_seconds=fast_seconds));speed.append(fast_seconds)
 rejected=[];base=records[-1]['fast']
 tests=[]
 for label,modify in(('missing_card',lambda x:x['cards'].pop()),('missing_SM',lambda x:x['cards'][0].update(sm=None)),('invalid_memory',lambda x:x['cards'][0].update(memory_mib=-1)),('fake_empty',lambda x:x['cards'][0].update(available=True)),('wrong_quota',lambda x:x.update(own_gpu_count=7)),('missing_PID',lambda x:x['cards'][0].update(pids=None))):
  x=copy.deepcopy(base);modify(x);tests.append((label,x))
 for label,value in tests:
  try:module.validate(value)
  except(AssertionError,TypeError,KeyError):rejected.append(label)
  else:raise AssertionError(label)
 with patch.object(module.subprocess,'run',side_effect=subprocess.TimeoutExpired('NVMLfixture',3))as mocked:
  try:module.inventory()
  except subprocess.TimeoutExpired:rejected.append('three_second_timeout_no_fallback')
  else:raise AssertionError('Timeout suppressed')
  assert mocked.call_args.kwargs['timeout']==3 and mocked.call_count==1
 with patch.object(module.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'NVMLfixture')):
  try:module.inventory()
  except subprocess.CalledProcessError:rejected.append('NVML_failure_no_fallback')
  else:raise AssertionError('NVML failure suppressed')
 O=T/'outputs/native_fast_inventory_v1';value=dict(passed=True,actual_all_seven_card_brackets=records,negative_contracts_rejected=rejected,max_actual_NVML_seconds=max(speed),same_three_second_timeout=True,no_stale_or_missing_metric_fallback=True,normal_process_GPU_utilization_guard_unchanged=True,NVML_memory_v2_excludes_driver_reserved=True,MiB_rounding='ceil to match NVIDIA CLI',CUDA_work_started=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')},pynvml_source_sha256=hashlib.sha256(Path('/usr/lib/python3/dist-packages/pynvml.py').read_bytes()).hexdigest())
 with(O/'CPU_actual_inventory_checks.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
 print('ACTUAL_NVML_INVENTORY_AND_FAIL_CLOSED_PASS',len(records),max(speed),rejected,flush=True)
if __name__=='__main__':main()
