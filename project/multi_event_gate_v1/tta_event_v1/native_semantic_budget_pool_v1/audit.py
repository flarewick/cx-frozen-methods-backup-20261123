"""Actual immutable CPU leases and real inference receipts; no GPU metrics."""
from pathlib import Path
from native_policy_execution_v1.common import read,write,sha

def audit(root):
 root=Path(root);result=read(root/'semantic/result.json');assert result['complete']and result['CPU_thread_budget_guarded']and result['maximum_active_CPU_threads']==8
 leases=sorted((root/'semantic/CPU_thread_budget').glob('*.json'));assert len(leases)==result['requests']and all(code==0 for code in result['actual_worker_exit_codes'])
 inputs={};peak=0
 for path in leases:
  lease=read(path);assert lease['threads']in(2,4,8)and lease['maximum_active_CPU_threads']==8
  before=lease['active_before'];assert len(before)==4 and before[lease['worker']]==0 and all(v in(0,2,4,8)for v in before)
  assert sum(before)+lease['threads']==lease['active_after']<=8;peak=max(peak,lease['active_after'])
  answer=root/'semantic/answers'/path.name;value=read(answer);allocation=Path(value['CPU_thread_allocation_file']);assert allocation.read_bytes()==path.read_bytes()and sha(allocation)==value['CPU_thread_allocation_sha256']
  assert value['actual_CPU_threads']==lease['threads']and value['content_key']==lease['content_key']and not value['CUDA_initialized']and value['CPU_RNG_preserved']
  for f in(path,answer,allocation):inputs[str(f)]=sha(f)
 value=dict(passed=True,actual_requests=len(leases),maximum_active_threads=peak,limit8_preserved=True,workers_actually_waited0=True,only_CPU_scheduling_no_new_GPU_or_policy_inputs=True,input_sha256=inputs)
 write(root/'CPU_semantic_thread_budget_audit.json',value);return value
