from pathlib import Path
import threading,ast
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import verify
from .scope import transformed

def main():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 old=T/'native_onpolicy_arm_collection_v31';new=T/'native_onpolicy_arm_collection_v34'
 for p in old.rglob('*'):
  if p.is_file()and p.suffix in('.py','.md','.sh'):
   q=new/p.relative_to(old);assert q.read_text()==transformed(p.name,p.read_text()),str(q)
   if q.suffix=='.py':ast.parse(q.read_text())
 from native_onpolicy_arm_collection_v34.work import FixedWorkQueue
 from native_onpolicy_arm_collection_v34.dispatch_checks import run
 tasks=['hammer','switch','bottle','roller','laptop'];cancel=threading.Event()
 for cached_count in [1,3,5]:
  cached={task:dict(items=list(range(cached_count)))for task in tasks}
  q=FixedWorkQueue(25,tasks,cached,cancel);assert q.order==list(range(25))
  assert len(set(tasks[i%5]for i in q.order[:4]))==4
  for task in tasks:assert [i//5 for i in q.order if tasks[i%5]==task]==list(range(5))
 routing=run();assert routing['passed']and routing['cases']==25
 proof=T/'outputs/native_onpolicy_arm_collection_v31/CPU_contract_checks_v1.json';verify(read(proof));assert read(proof)['passed']
 actual=proof.with_name('CPU_actual_host_exit_v1.json');assert read(actual)['exit_code']==0
 sources=[p for folder in[new,Path(__file__).parent]for p in folder.rglob('*')if p.is_file()and p.suffix in('.py','.md','.sh')]
 write(T/'outputs/native_onpolicy_arm_collection_v34/CPU_contract_checks_v2.json',dict(passed=True,
  original31_runtime_episode_owner_model_teacher_guards_byte_exact=True,only_task_interleaved_cached_prefix_order_changed=True,
  actual_Torch_numpy_python_case_RNG25=routing,all25_and_per_task_seed_order_preserved=True,
  no_GPU_or_native_resource_or_TTA_qualification=True,source_sha256={str(p):sha(p)for p in sources},input_sha256={str(p):sha(p)for p in[proof,actual]}))
 print('INTERLEAVED_ORIGINAL_TASKS_ACTUAL25_RNG_CPU_PASS',flush=True)
if __name__=='__main__':main()
