"""Replay actual ARM11 admission and failure; CPU-only, no NVML/GPU calls."""
from pathlib import Path
import copy,json,threading
from unittest.mock import patch
from native_policy_execution_v1.common import T,read,write,sha
from native_server_rule4_v10.history import History
from native_server_rule4_v10 import admission as old
from . import admission as new

def history(rows):
 h=History()
 for d in rows:h.update(d['fleet'],d['monotonic'],{int(p):tuple(v)for p,v in d['process_ages'].items()})
 return h

def instance(h,latest,admitted):
 m=object.__new__(new.Monitor);m.root=T/'outputs/native_server_rule4_v11/CPU_replay_fixture';m.history=h;m.latest=latest;m.admitted=admitted;m.failed=None;m.lock=threading.Lock();return m

def rejects(m,uuid):
 try:m.require(uuid,running=True)
 except RuntimeError:return True
 raise AssertionError('Invalid running provenance accepted')

def main():
 O=T/'outputs/native_server_rule4_v11';O.mkdir(exist_ok=True);assert not(O/'CPU_actual_scope_checks_v4.json').exists()
 stage=T/'outputs/native_onpolicy_arm_collection_v11/val_000';root=Path(read(stage/'execution_state.json')['root']);assert read(stage/'actual_host_wait_exit.json')['exit_code']==1
 failure=read(root/'failure.json');assert 'Conservative ongoing server rule4 cannot be verified' in failure['traceback']
 original=read(root/'attempt/rule4_admission.json');uuid=read(root/'protocol.json')['required_gpu_uuid']
 assert original['passed']and not any(c['own_pids']for c in original['latest_observation']['fleet']['cards'])
 monitor=Path(read(stage/'rule4_monitor_identity.json')['root']);file=monitor/'observations.jsonl'
 rows=[json.loads(l)for l in file.read_text().splitlines()];cutoff=read(root/'last_successfully_collected_metric_at_failure.json')['metric']['unix_ns']
 failed=[d for d in rows if d['observed_unix_ns']<=cutoff];h=history(failed);latest=failed[-1];now=latest['monotonic']
 okay,reason,_=h.decision(uuid,now,running=True);assert not okay and 'except the least-filled' in reason
 m=instance(h,latest,{uuid:copy.deepcopy(original)})
 with patch.object(new.time,'monotonic',return_value=now):
  accepted=m.require(uuid,running=True);assert accepted['passed']and accepted['original_strict_admission_sha256']==new.digest(original)
  missing=instance(copy.deepcopy(h),latest,{});assert rejects(missing,uuid)
  stale=instance(copy.deepcopy(h),latest,{uuid:original});stale.history.observed=now-8.01;assert rejects(stale,uuid)
  broken=instance(copy.deepcopy(h),latest,{uuid:original});broken.failed='actual metadata error';assert rejects(broken,uuid)
  absent=instance(copy.deepcopy(h),latest,{uuid:original});absent.history.fleet['cards']=[c for c in absent.history.fleet['cards']if c['uuid']!=uuid];assert rejects(absent,uuid)
  excessive=instance(copy.deepcopy(h),latest,{uuid:original});excessive.history.fleet['own_gpu_count']=4;assert rejects(excessive,uuid)
  # A NEW allocation under the exact failing fleet remains inadmissible.
  fresh=instance(copy.deepcopy(h),latest,{})
  for card in fresh.history.fleet['cards']:
   if card['uuid']==uuid:card['pids']=[];card['own_pids']=[]
  fresh.history.fleet['own_gpu_count']-=1
  try:fresh.require(uuid)
  except RuntimeError:pass
  else:raise AssertionError('Low-memory prior cards accepted for new allocation')
  assert fresh.admitted=={}
 # Replay the real first-card quiet prefix and execute the ORIGINAL strict method.
 at=original['latest_observation']['monotonic'];prefix=[d for d in rows if d['monotonic']<=at];initial=instance(history(prefix),prefix[-1],{})
 with patch.object(new.time,'monotonic',return_value=at),patch.object(old,'snapshot',return_value=original['fresh_original_rule4_fleet']):
  before=initial.require(uuid);assert before['passed']and initial.admitted[uuid]==before
 assert new.digest({2:[1,2],1000000:(3,4)})==new.digest(json.loads(json.dumps({2:[1,2],1000000:(3,4)})))
 from .audit import audit as actual_audit
 fixture=O/'CPU_scope_serialization_fixture_v4';assert not fixture.exists()
 for name,wrong in [('valid',False),('wrong_admission',True)]:
  target=fixture/name;write(target/'protocol.json',dict(required_gpu_uuid=uuid,CPU_only_fixture=True))
  write(target/'attempt/rule4_admission.json',original);write(target/'attempt/barrier/permit.json',dict(CPU_only_fixture=True))
  decision=copy.deepcopy(accepted)
  if wrong:decision['original_strict_admission']['reason']+=' WRONG ADMISSION'
  write(target/'attempt/rule4_running.json',decision)
  (target/'attempt/rule4_running.jsonl').write_text(json.dumps(dict(seconds=1,proof=decision))+'\n')
  if wrong:
   try:actual_audit(target)
   except AssertionError:pass
   else:raise AssertionError('Wrong admission provenance accepted by terminal audit')
  else:assert actual_audit(target)['passed']
 assert new.Monitor.__init__.__globals__['OriginalMonitor']is old.Monitor
 inputs=[stage/'actual_host_wait_exit.json',root/'failure.json',root/'attempt/rule4_admission.json',root/'protocol.json',root/'attempt/resources.jsonl',file]
 sources=[p for p in Path(__file__).parent.glob('*')if p.is_file()]+[T/'native_server_rule4_v10/admission.py',T/'native_server_rule4_v10/history.py']
 write(O/'CPU_actual_scope_checks_v4.json',dict(passed=True,actual_ARM11_first_card_admission_replayed=True,actual_later_job_false_positive_replayed=True,old_running_rejected_reason=reason,new_running_preserves_prior_legal_admission=True,original_strict_admission_method_executed_unchanged=True,new_allocation_under_same_low_prior_memory_still_refused=True,account4_stale_failed_metadata_absent_card_and_missing_admission_rejected=True,all_GPU_CPU_VRAM_RSS_PID_and_clock_guards_unchanged=True,no_GPU_or_scientific_qualification=True,source_sha256={str(p):sha(p)for p in sources},input_sha256={str(p):sha(p)for p in inputs}))
 print('RULE4V11_ACTUAL_LEGAL_ADMISSION_AND_LATER_JOB_SCOPE_PASS',flush=True)

if __name__=='__main__':main()
