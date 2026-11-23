"""Require fresh full scans and actual normal CPU worker exits."""
import json
from .transport import validate
from native_policy_execution_v1.common import read,write,sha
def audit(root,proofs):
 bindings={};total=0
 assert len(proofs)==3 and len(set(proofs))==3
 for proof in proofs:
  note=read(proof);assert note['exit_code']==0 and note['actual_wait_return']and note['proc_absent_after_wait']and note['no_CUDA_or_GPU_work']and not note['forced']and not note['failed_request']
  log=proof.with_suffix('.requests.jsonl');rows=[json.loads(x)for x in log.read_text().splitlines()]
  assert len(rows)==note['request_count']>0 and [r['request']['sequence']for r in rows]==list(range(1,len(rows)+1))
  for r in rows:
   assert 'error'not in r and r['worker_pid']==note['worker_pid']and r['request']['nonce']==note['nonce']
   assert r['timeout_seconds']==3 and r['elapsed_seconds']<=3 and not r['automatic_retry']and not r['CLI_used']and not r['previous_call_frames_used']
   assert r['parent_requested_unix_ns']<=r['request_started_unix_ns']<=r['response_ready_unix_ns']<=r['parent_received_unix_ns']
   validate(r['data'],r['request']['kind'])
  bindings.update({str(proof):sha(proof),str(log):sha(log)});total+=len(rows)
 value=dict(passed=True,actual_normal_CPU_workers=3,actual_fresh_account_scans=total,original_collection_bodies_and_three_second_deadline_unchanged=True,no_GPU_metrics_or_accounting_clock_changed=True,no_CUDA_or_GPU_work=True,input_sha256=bindings)
 write(root/'CPU_account_transport_terminal.json',value);return value
