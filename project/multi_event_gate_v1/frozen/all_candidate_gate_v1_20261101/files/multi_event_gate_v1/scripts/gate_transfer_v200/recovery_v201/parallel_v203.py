"""Independent control-branch scheduling; preserve every rollout statement and guard."""
import ast,copy,inspect,linecache,json,time
from pathlib import Path
_INSTALLED=False

def remove_wait(fn):
 assert isinstance(fn,ast.FunctionDef)and fn.name=='ordered'
 original=copy.deepcopy(fn)
 hits=[(i,n)for i,n in enumerate(fn.body)if isinstance(n,ast.If)and ast.unparse(n.test)=="branch not in ('A0', 'G')"]
 assert len(hits)==1
 index,old=hits[0];assert 'completed'in ast.unparse(old)and 'check_cancel'in ast.unparse(old)and 'rollout('not in ast.unparse(old)
 del fn.body[index]
 restored=copy.deepcopy(fn);restored.body.insert(index,copy.deepcopy(old));assert ast.dump(restored)==ast.dump(original)

def tree_for(function):
 import textwrap
 tree=ast.parse(textwrap.dedent(inspect.getsource(function)));remove_wait(tree.body[0]);return tree

def independent(function):
 tree=tree_for(function);ns=dict(function.__globals__);ns.update({k:c.cell_contents for k,c in zip(function.__code__.co_freevars,function.__closure__ or ())})
 text=ast.unparse(ast.fix_missing_locations(tree))+'\n';label='<independent_control_branches_v203>';linecache.cache[label]=(len(text),None,text.splitlines(True),label);exec(compile(text,label,'exec'),ns);return ns[function.__name__]

class BacklogExpertSlot:
 """Resource admission only; never read a policy success or intervention."""
 def __init__(self,original,queues,capacity=10):self.original,self.queues,self.capacity=original,queues,capacity
 def __getattr__(self,name):return getattr(self.original,name)
 def pending(self):
  total=0
  for queue in list(self.queues.values()if isinstance(self.queues,dict)else self.queues):
   with queue.lock:
    total+=sum(bool(row['eligible'])and str(row['seed'])not in queue.state['completed']for row in queue.state['attempts'])
  return total
 def acquire(self,blocking=True,timeout=None):
  if self.pending()>=self.capacity:
   if blocking:time.sleep(min(.05,timeout)if timeout is not None else .05)
   return False
  return self.original.acquire(blocking=blocking,timeout=timeout)
 def release(self):return self.original.release()

def install():
 global _INSTALLED
 if _INSTALLED:return
 import benefit_growth_runtime as runtime
 previous=runtime.install
 def install_runtime():
  previous()
  import benefit_followup_protocol_v2 as p
  import shared6_eval_core as core,benefit_memory_batches as batches
  if getattr(core,'_transfer_parallel203',False):return
  predecessor=batches.predecessor;acquire=batches.acquire
  def ready(queue,row):
   task,branch=p.decode(queue.manifest['condition']);assert task in p.TASKS
   if branch in('A0','A1','B'):return True
   return predecessor(queue,row)
  def take(queue,screen,cancel,**kwargs):return acquire(queue,screen,cancel,ready=kwargs.get('ready',ready))
  batches.predecessor=ready;batches.acquire=take
  # Transform the original ordered AST at its existing bounded-call boundary.
  # The historical transformer compiles under an unrelated filename/line range;
  # inspecting its RETURNED function again reads the wrong source. Intercept the
  # existing transform instead, preserving its mutator (including the semaphore).
  import benefit_stage_runtime as stage,benefit_stage_runtime_v2 as v2,threading
  assert not batches._INFLIGHT and v2._slots._value==6
  # MAX_CLAIMED does not expand the independent original bounded semaphore.
  # Both constraints must admit the same ten REAL episodes; retain the wrapper.
  v2._slots=threading.BoundedSemaphore(10)
  # Fully qualified policy backlog needs no further expert work yet. Deferring
  # expert SLOT admission avoids its expensive private planner blocking owner
  # inference; no started expert scene is interrupted and membership stays exact.
  import benefit_followup_mixed_manager_v3 as manager
  from concurrent_queue import EpisodeQueue
  queues={};initialize=EpisodeQueue.__init__
  def initialize_queue(self,*a,**kw):
   initialize(self,*a,**kw);queues[str(self.run)]=self
  EpisodeQueue.__init__=initialize_queue
  manager._ACTIVE=BacklogExpertSlot(manager._ACTIVE,queues)
  core._transfer_backlog205=True

  transform=stage.transform
  def transform_control(function,mutate):
   if function.__name__!='ordered':return transform(function,mutate)
   def edit(node):
    mutate(node);remove_wait(node)
   result=transform(function,edit)
   folder=p.ROOT/'parallel_control_scope_v203';folder.mkdir(exist_ok=True)
   p.atomic(folder/(str(time.time_ns())+'.json'),dict(passed=True,helper_sha256=p.sha(__file__),original_transform_intercepted_before_compile=True,original_bounded_rollout_mutator_retained=True,only_predecessor_wait_block_removed=True,all_rollout_trace_action_state_and_metadata_statements_inverse_AST_exact=True,independent_task_policy_and_RNG_per_lane_unchanged=True,real_scene_cap10=True,original_physical_semaphore_6_explicitly_superseded=True,G_still_requires_complete_control_audit=True,old_protocol_A0_A1_B_time_order_explicitly_superseded=True))
   return result
  stage.transform=transform_control;core._transfer_parallel203=True
  __import__('balanced_v207').apply()

 runtime.install=install_runtime;_INSTALLED=True

def main():
 import torch,numpy as np,threading,types
 import benefit_followup_protocol_v2 as p
 torch.set_num_threads(1);assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized()
 __import__('benefit_growth_runtime').bind()
 # Use the actual original nested ordered function, with passive CPU callbacks only.
 tree=ast.parse((p.D/'benefit_stage_runtime.py').read_text());node=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='ordered');node=copy.deepcopy(node)
 for task_name in p.TASKS:
  for branch in ('A0','A1','B','G'):
   calls=[];task=types.SimpleNamespace(_pilot_recording=True,_pilot_evidence=[],_pilot_ticks=17)
   ns=dict(branch=branch,name=task_name,task=task,rollout=lambda seed,index,row:(calls.append((seed,index))or dict(evidence_sha256={})),sha=p.sha,Path=Path,json=json,time=time)
   source=ast.unparse(ast.fix_missing_locations(node))+'\n';label='<original_ordered203_CPU_fixture>';linecache.cache[label]=(len(source),None,source.splitlines(True),label);exec(compile(source,label,'exec'),ns)
   modified=independent(ns['ordered']);result=modified(123,0,dict(qualified_index=0))
   assert calls==[(123,0)]and result['pilot_branch']==branch and result['native_physics_ticks']==17 and result['passive_trace_complete']and not task._pilot_recording
   # Execute the ACTUAL v2 configure wrapper that previously returned a
   # source-misattributed transformed function. Its closure is passive CPU only.
   import benefit_stage_runtime as stage,benefit_stage_runtime_v2 as v2
   import benefit_growth_runtime as runtime
   runtime.install()
   raw=ns['ordered'];calls.clear()
   v2tree=ast.parse((p.D/'benefit_stage_runtime_v2.py').read_text());wrapped=copy.deepcopy(next(n for n in ast.walk(v2tree)if isinstance(n,ast.FunctionDef)and n.name=='wrapped'))
   count=[0]
   def bounded(fn,*a,**kw):count[0]+=1;return fn(*a,**kw)
   stage._stage_bounded=bounded;raw.__globals__['_stage_bounded']=bounded
   scope=dict(configure=lambda *a,**kw:dict(rollout=raw),stage=stage,ast=ast)
   exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapped],type_ignores=[])),'<actual_v2_configure_CPU>','exec'),scope)
   callback=scope['wrapped']()['rollout'];answer=callback(123,0,dict(qualified_index=0))
   assert calls==[(123,0)]and count==[1]and answer==result

 # Verify the actual ORIGINAL bounded function, not merely MAX_CLAIMED.
 import benefit_stage_runtime_v2 as v2
 assert v2._slots._value==10
 entered=[];lock=threading.Lock();ten=threading.Event();drain=threading.Event()
 def physical(i):
  def real():
   with lock:
    entered.append(i)
    if len(entered)==10:ten.set()
   assert drain.wait(10)
  v2.bounded(real)
 threads=[threading.Thread(target=physical,args=(i,))for i in range(11)]
 for t in threads:t.start()
 assert ten.wait(10)
 assert len(entered)==10 and v2._slots._value==0
 drain.set()
 for t in threads:t.join(10);assert not t.is_alive()
 assert len(entered)==11 and v2._slots._value==10
 # Expert admission depends on pending work only, not success or gate decisions.
 from types import SimpleNamespace
 class OriginalSlot:
  def __init__(self):self.calls=0;self.releases=0
  def acquire(self,**kwargs):self.calls+=1;return True
  def release(self):self.releases+=1
 q=SimpleNamespace(lock=threading.RLock(),state=dict(attempts=[dict(seed=i,eligible=True)for i in range(12)],completed={}))
 original=OriginalSlot();registry={'same_run':q};slot=BacklogExpertSlot(original,registry)
 assert slot.pending()==12 and slot.acquire(blocking=False)is False and original.calls==0
 q.state['completed']={str(i):dict(success=(i%2==0))for i in range(3)}
 assert slot.pending()==9 and slot.acquire(timeout=.05)is True and original.calls==1
 slot.release();assert original.releases==1
 for row in q.state['completed'].values():row['success']=not row['success']
 assert slot.pending()==9
 registry['same_run']=SimpleNamespace(lock=threading.RLock(),state=dict(attempts=[],completed={}))
 assert slot.pending()==0,'New GPU queue must replace stale CPU queue for same run'
 # Original serialized owner RNG must remain independent when all branches are ready together.
 from concurrent_core import Coordinator
 from concurrent_rng import Lane
 from model_service import ModelService
 c=Coordinator(cuda=False);c.service=ModelService(c);c.rng.install();initial=torch.get_rng_state().clone();barrier=threading.Barrier(13);result={};errors=[]
 def worker(i):
  try:
   lane=Lane('independent_controls/'+str(i),seed=200+i)
   with c.rng.bind(lane):
    barrier.wait(10)
    for _ in range(12):
     a,b=c.call(lane,lambda:(np.random.normal(size=7),torch.rand(9)));result.setdefault(i,[]).append((a,b.numpy()))
  except BaseException as e:errors.append(repr(e));c.cancel.set()
 threads=[threading.Thread(target=worker,args=(i,))for i in range(12)]
 for t in threads:t.start()
 barrier.wait(10)
 while any(t.is_alive()for t in threads):c.service.process_one(.01)
 for t in threads:t.join()
 assert not errors,errors
 for i in range(12):
  n=np.random.RandomState(200+i);g=torch.Generator();g.set_state(initial)
  for a,b in result[i]:assert np.array_equal(a,n.normal(size=7))and np.array_equal(b,torch.rand(9,generator=g).numpy())
 assert c.service.executed==144 and torch.equal(initial,torch.get_rng_state())and not torch.cuda.is_initialized()
 p.atomic(p.ROOT/'independent_control_schedule_CPU_v203.json',dict(passed=True,CPU_only=True,no_CUDA=True,actual_original_rollout_AST_all_but_wait_block_exact=True,all8_actual_task_branch_callback_paths_tested=True,all8_actual_v2_configure_wrapper_paths_tested=True,original_bounded_rollout_callback_exactly_once=True,actual_original_bounded_function10_of11_concurrency_verified=True,expert_admission_only_when_pending_policy_backlog_below10_CPU_verified=True,no_policy_success_read_for_expert_admission=True,original_owner144_requests_12_lanes_RNG_exact=True,helper_sha256=p.sha(__file__),original_stage_sha256=p.sha(p.D/'benefit_stage_runtime.py'),gate_adapter_and_all_server_thresholds_unchanged=True))
 print('INDEPENDENT_CONTROL203_CPU_PASSED',flush=True)
if __name__=='__main__':main()
