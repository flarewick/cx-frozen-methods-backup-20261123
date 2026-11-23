"""Fresh empty-card stability proof; no CUDA, no reservation or guard exemption."""
import time
from native_policy_execution_v1.common import T,read,write,sha
from native_account_stream_v1.transport import inventory
from native_onpolicy_recovery_v3.common import mapped
OLD=T/'outputs/native_calibrated_validation_v21/development_000'

def carry():
 actual=OLD/'actual_host_wait_exit.json';a=read(actual)
 assert a['exit_code']==1 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 root=mapped(read(OLD/'execution_state.json')['root']);file=root/'attempt/resources.jsonl'
 import json
 rows=[json.loads(s)for s in file.read_text().splitlines()]
 assert [r['stopped']for r in rows if 'stopped'in r]==['Per-process cumulative/rolling GPU utilization is unverified or below 34%']
 release=rows[-1];assert release['own_gpu_released']and not release['remaining_own_gpu_pids']and release['cleanup_seconds_after_start']==282.633
 assert not(root/'terminal_audit.json').exists() # Failed partial policy branches remain excluded.
 from native_calibrated_eval_resume_v1.admission import carry as prior_carry
 previous,prior_bindings=prior_carry();assert previous==12.834
 return previous+282.633,{**prior_bindings,**{str(p):sha(p)for p in (actual,file,root/'failure.json',OLD/'execution_state.json')}}

def valid(fleet,gpu,uuid):
 assert fleet['own_gpu_count']<3
 rows=[r for r in fleet['cards']if r['index']==gpu and r['uuid']==uuid];assert len(rows)==1
 row=rows[0];assert row['available']and not row['pids']and row['memory_mib']<=64 and row['sm']<=5
 return row

def certify(root,gpu,uuid):
 assert not(root/'attempt/barrier/permit.json').exists()
 rows=[]
 for i in range(2):
  fleet=inventory();card=valid(fleet,gpu,uuid);rows.append(dict(monotonic=time.monotonic(),fleet=fleet,selected=card))
  if i==0:time.sleep(2)
 write(root/'CPU_stable_empty_before_permit.json',dict(passed=True,observations=rows,CUDA_permit_absent=True,no_GPU_work=True,original_postpermit_foreign_guard_unchanged=True))
