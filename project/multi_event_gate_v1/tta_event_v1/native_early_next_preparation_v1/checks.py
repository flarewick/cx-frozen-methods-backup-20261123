"""Production scope and bookkeeping checks; fixtures cannot qualify native data."""
from pathlib import Path
import copy,inspect,threading,types
from native_policy_execution_v1.common import T,read,write,sha
from .schedule import ready,isolated,build_prepare,adopt_closed_tasks

def main():
 import torch
 from native_onpolicy_arm_collection_v22 import operational_checks as original_checks
 from native_onpolicy_arm_collection_v22.future_preparation import prepare as original_prepare
 from native_onpolicy_arm_collection_v22.future_preparation import audit
 from native_onpolicy_arm_collection_v22.known_preparation import collect
 from native_onpolicy_arm_collection_v22.common import TASKS
 from native_onpolicy_arm_collection_v22.priority_lock import PriorityLock,FutureTail
 from concurrent_rng import Lane
 from native_q0_task_handoff_v3.handoff import detach
 assert not torch.cuda.is_initialized()
 out=T/'outputs/native_early_next_preparation_v1';out.mkdir(parents=True,exist_ok=True)
 work=types.SimpleNamespace(condition=threading.Condition(),ready=set(range(8)))
 spec=dict(future_preparation_population=dict(split='fit',seed_start=7124000,first_index=0),dataset_split='adapter_validation')
 event=threading.Event();assert not ready(work,spec,event);work.ready.add(8);assert ready(work,spec,event)
 assert not ready(work,{},event);event.set();assert ready(work,{},event)
 # Reproduce the exact stale-entry failure before touching any Torch/GPU state.
 fixture=types.SimpleNamespace(c=types.SimpleNamespace(service=types.SimpleNamespace(owner=threading.get_ident())),q0_pending={})
 try:detach(fixture,'hammer',dict(case_id='required-future:0'),None,None)
 except KeyError as error:assert error.args==('required-future:0',)
 else:raise AssertionError('Missing future q0 consumer not reproduced')
 cancel=threading.Event();lock=PriorityLock(cancel);whole=[25]
 parent=types.SimpleNamespace(state={'a':dict(accepted=5,rows=[])},tasks={'current':object()},
  lanes={'a':object()},locks={'a':threading.Lock()},warmed={'a'},spec={'tasks':['a']},
  q0_pending={'current':object()},on_initial=object(),serial_screen_lock=lock,
  pool={},allow_future=lambda:whole[0]>=4,c=fixture.c)
 before=(copy.deepcopy(parent.state),dict(parent.tasks),parent.on_initial,parent.q0_pending)
 calls=[]
 def original_next(self,name):
  with self.serial_screen_lock:
   calls.append(name);self.state[name]['accepted']+=1;self.tasks[name]=object();return self.state[name]['accepted']
 private=isolated(parent,original_next,Lane)
 assert private.original_next.__func__ is original_next and private.pool is parent.pool and private.serial_screen_lock is lock
 with lock.background():assert private.next('a')==6
 assert parent.state==before[0]and parent.tasks==before[1]and parent.on_initial is before[2]and parent.q0_pending is before[3]
 whole[0]=3
 with lock.background():
  try:private.next('a')
  except FutureTail:pass
  else:raise AssertionError('Original minimum4 dispatch guard bypassed')
 assert calls==['a'];adopt_closed_tasks(parent,private)
 assert 'fixed_next_closed_a'in parent.tasks and not private.tasks
 # Exercise the actual original exporter with only its declared completion-read
 # scope changed. Fail if it attempts to read the current-population completion.
 prepare=build_prepare(original_prepare)
 prepare.__globals__['write']=write
 audit=types.FunctionType(audit.__code__,dict(audit.__globals__,write=write))
 original_read=prepare.__globals__['read']
 def strict_read(path):
  path=Path(path)
  if path.name=='screening_complete.json'and path.parent.name!='next_preparation':raise AssertionError('Early preparation read unfinished current population')
  return original_read(path)
 prepare.__globals__['read']=strict_read
 source=inspect.getsource(original_checks.future_check)
 source=source.replace(' from .future_preparation import prepare,audit',' prepare=patched_prepare; audit=original_audit')
 source=source.replace(" spec=dict(future_preparation_each_task=2,", " spec=dict(early_fixed_next_population_preparation=True,future_preparation_each_task=2,")
 namespace=dict(original_checks.__dict__,patched_prepare=prepare,original_audit=audit,O=out)
 exec(compile(source,__file__,'exec'),namespace)
 cache=collect({t:dict(next_seed=7125000,accepted=0)for t in TASKS},TASKS,5,7125000)
 exports=[namespace['future_check'](cache,n)for n in (0,3,10)]
 assert not torch.cuda.is_initialized()
 files=[Path(__file__),Path(__file__).with_name('schedule.py'),Path(__file__).with_name('AGENTS.md')]
 write(out/'CPU_scope_checks_v1.json',dict(passed=True,original_exporter_AST_exact_after_declared_completion_read_unwind=True,
  actual_old_future_handoff_KeyError_reproduced=True,original_raw_screen_next_called_once=True,
  actual_ready9_and_minimum4_original_lock_preserved=True,parent_state_tasks_callback_pending_unchanged=True,
  shared_single_planner_and_owner_cleanup_preserved=True,CPU_exports=exports,
  fixed_population_and_maximum10_unchanged=True,no_GPU_or_native_qualification=True,
  source_sha256={str(f):sha(f)for f in files},input_sha256={str(T/'outputs/native_onpolicy_arm_collection_v22/CPU_contract_checks_v2.json'):sha(T/'outputs/native_onpolicy_arm_collection_v22/CPU_contract_checks_v2.json')}))
 print('INDEPENDENT_REQUIRED_NEXT_PREPARATION_CPU_PASS',flush=True)

if __name__=='__main__':main()
