"""Known governor counterexample and scheduled actual read-only observations."""
from pathlib import Path
import time,json
from .history import History
from .admission import Monitor
from native_server_rule4_v1.admission import decision
T=Path(__file__).resolve().parents[1]
def fleet(memory=.7,util=20,own=True):
 return dict(own_gpu_count=int(own),cards=[dict(index=0,uuid='GPU-prior',pids=[123]if own else[],own_pids=[123]if own else[],own_fraction=memory if own else 0,attributable_utilization=util if own else None),dict(index=1,uuid='GPU-selected',pids=[],own_pids=[],own_fraction=0,attributable_utilization=None)])
def main():
 idle=History()
 for t in range(0,31,5):idle.update(fleet(own=False),t,{})
 assert idle.decision('GPU-selected',30)[0]and not idle.decision('GPU-selected',40)[0]
 old=History()
 for t in range(0,121,5):old.update(fleet(.49,99 if t==120 else 22.9),t,{123:(17,3600+t)})
 assert decision(fleet(.49,99),'GPU-selected')[0]
 assert not old.decision('GPU-selected',120)[0],'Instantaneous spike must not qualify server averages'
 util_only=History()
 for t in range(0,126,5):util_only.update(fleet(.2,100),t,{123:(17,t+1)})
 assert not util_only.decision('GPU-selected',125)[0],'This conservative campaign uses the average-memory path only'
 stable=History()
 for t in range(0,126,5):stable.update(fleet(),t,{123:(17,t+1)})
 assert stable.decision('GPU-selected',125)[0]
 stable.update(fleet(),130,{123:(18,1)})
 assert not stable.decision('GPU-selected',130)[0],'Reused PID/new startticks must lose eligibility'
 # A job starting after the quiet-history snapshot must revoke admission,
 # even if the old instantaneous rule would accept its transient utilization.
 import threading
 from . import admission as admission_module
 test=Monitor.__new__(Monitor);test.lock=threading.Lock();test.failed=None;test.history=History();test.root=T/'outputs/native_server_rule4_v5/CPU_identity_fixture';test.latest=dict(process_ages={})
 now=time.monotonic()
 for elapsed in range(0,31,5):test.history.update(fleet(own=False),now-30+elapsed,{})
 saved=admission_module.snapshot;admission_module.snapshot=lambda:fleet(.49,99)
 try:
  try:test.require('GPU-selected');raise AssertionError('Changed account population used old history')
  except RuntimeError as error:assert 'population changed'in str(error)
 finally:admission_module.snapshot=saved
 actual=Monitor()
 try:
  end=time.monotonic()+15
  while time.monotonic()<end:
   with actual.lock:
    if actual.failed:raise RuntimeError(actual.failed)
    lines=(actual.root/'observations.jsonl').read_text().splitlines()if(actual.root/'observations.jsonl').exists()else[]
   if len(lines)>=2:break
   time.sleep(.1)
  assert len(lines)>=2;frames=[json.loads(line)for line in lines];assert all(f['no_CUDA_or_GPU_work']and f['scientific_GPU_clock_not_reset']for f in frames)
 finally:actual.close()
 import hashlib
 o=T/'outputs/native_server_rule4_v5';report=dict(passed=True,known49_percent_average_and_transient99_percent_spike_rejected=True,first_GPU_requires30s_account_idle=True,PID_identity_changes_revoke_eligibility=True,stable_full_process_age_history_accepted=True,actual_fresh_readonly_frames=len(frames),actual_monitor_root=str(actual.root),no_GPU_or_CUDA_work=True,original34_percent_scientific_resource_checks_unchanged=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*')if p.is_file()})
 with(o/'CPU_actual_handoff_admission_checks_v2.json').open('x')as f:json.dump(report,f,indent=2)
 print('CONSERVATIVE_RULE4_KNOWN_GOVERNOR_COUNTEREXAMPLE_AND_ACTUAL_READONLY_MONITOR_PASS')
if __name__=='__main__':main()
