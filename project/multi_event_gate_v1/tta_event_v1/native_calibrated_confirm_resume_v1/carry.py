"""Retain the exact failed cold-confirmation GPU budget; no result promotion."""
import json
from native_policy_execution_v1.common import T,read,sha
from native_onpolicy_recovery_v3.common import mapped

def carry():
 base=T/'outputs/native_calibrated_validation_v23/confirmation_000'
 actual=base/'actual_host_wait_exit.json';a=read(actual);assert a['exit_code']==1 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 root=mapped(read(base/'execution_state.json')['root']);file=root/'attempt/resources.jsonl';rows=[json.loads(s)for s in file.read_text().splitlines()]
 assert [d['stopped']for d in rows if 'stopped'in d]==['Per-process cumulative/rolling GPU utilization is unverified or below 34%']
 release=rows[-1];assert release['own_gpu_released']and not release['remaining_own_gpu_pids']and release['cleanup_seconds_after_start']==282.305
 assert not(root/'terminal_audit.json').exists()
 return release['cleanup_seconds_after_start'],{str(p):sha(p)for p in (actual,base/'execution_state.json',file,root/'failure.json')}
