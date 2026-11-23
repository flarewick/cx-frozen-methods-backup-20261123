"""Actual archived diagnosis and fail-closed negative cases; CPU only."""
import json,copy
from .sampling_gap import REFERENCE,diagnose
from .common import O,write

def main():
 rows=[json.loads(s)for s in(REFERENCE/'attempt/resources.jsonl').read_text().splitlines()];sample=[r for r in rows if r.get('gpu_pids')][-1];released=rows[-1].get('own_gpu_released')and not rows[-1]['remaining_own_gpu_pids'];proof=diagnose(sample,released);pid=str(sample['gpu_pids'][0]);negative=[]
 for name,edit in [('actual_numeric_low_utilization',lambda x:x['process_utilization'].__setitem__(pid,0)),('device_low',lambda x:x.__setitem__('mean_utilization',33)),('foreign_GPU_PID',lambda x:x.__setitem__('raw_foreign_pids',[999])),('VRAM',lambda x:x.__setitem__('memory_mib',24564)),('CPU',lambda x:x['cpu'].__setitem__(pid,1601))]:
  changed=copy.deepcopy(sample);edit(changed)
  try:diagnose(changed,True)
  except AssertionError:negative.append(name)
  else:raise AssertionError('Non-sensor failure was incorrectly accepted')
 try:diagnose(sample,False)
 except AssertionError:negative.append('GPU_not_released')
 else:raise AssertionError('Release proof absent')
 proof.update(negative_cases_rejected=negative,no_GPU_started=True);write(O/'CPU_sampling_gap_diagnosis.json',proof)
 print('ARCHIVED_NATIVE_PID_GAP_CPU_DIAGNOSIS_PASS',flush=True)
if __name__=='__main__':main()
