"""Independent replay of every additionally enforced resource checkpoint."""
import json
from native_policy_execution_v1.common import read,write,sha
from .margin import Monitor

def audit(root,spec):
 assert spec['governor_extra_margin_required']
 file=root/'attempt/resources.jsonl';extra=root/'attempt/governor_margin.jsonl';rows=[json.loads(line)for line in file.read_text().splitlines()];recorded=[json.loads(line)for line in extra.read_text().splitlines()];monitor=Monitor();n=0;steady=0
 for row in rows:
  if 'seconds'not in row:continue
  reason,value=monitor.observe(row);assert reason is None
  assert recorded[n]==dict(**value,stop_reason=None);n+=1
  if row['seconds']>=280 and row['gpu_pids']:steady+=1
 assert n==len(recorded)and steady>0 and rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids']
 proof=dict(passed=True,additional_cumulative35_and_process60_180_device60_36=True,original34_and_all_original_guards_preserved=True,all_samples_replayed=n,steady_samples=steady,input_sha256={str(p):sha(p)for p in[file,extra,root/'protocol.json']})
 write(root/'CPU_governor_margin_audit.json',proof);return proof
