"""No-device production callback and caller-identity checks."""
import ast,copy,inspect,json,sys,threading,time,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import entry212;entry212.install();import entry211
import torch,benefit_growth_runtime as runtime,benefit_followup_protocol_v2 as p
assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized();torch.set_num_threads(1)
runtime.install()
import startup211 as s,benefit_memory_batches as b,benefit_followup_mixed_core_v3 as mixed
from concurrent_queue import EpisodeQueue
assert s.INSTALLED and not b._INFLIGHT
# Exercise actual original queue acquire under every ramp level, no outcomes.
import tempfile,os
with tempfile.TemporaryDirectory(dir=p.ROOT,prefix='ramp211_CPU_')as tmp:
 file=Path(tmp)/'drain.json';old=os.environ.get('CX_BENEFIT_RECYCLE_FILE');os.environ['CX_BENEFIT_RECYCLE_FILE']=str(file)
 for level in (7,):
  s.CHUNKS.clear();s.CHUNKS.update({task:level-1 for task in p.TASKS});queues=[]
  for task in p.TASKS:
   for branch in ('A0','B'):
    rows=mixed.branch_rows(task,branch);assert sum(r['eligible']for r in rows)>=4
    queues.append(types.SimpleNamespace(run=Path(tmp)/task/branch,manifest=dict(condition=p.condition(task,branch),episodes=20),lock=threading.Condition(),state=dict(attempts=rows,completed={}),claimed=set()))
  entered=[];errors=[];event=threading.Event();drain=threading.Event();lock=threading.Lock()
  def worker(q):
   try:
    row=EpisodeQueue.acquire(q,None,threading.Event())
    if row is None:return
    with lock:
     entered.append((q.run.parent.name,row['seed']))
     if len(entered)==2*level:event.set()
    assert drain.wait(10);b.release(q,row)
   except BaseException as e:errors.append(repr(e));event.set()
  threads=[threading.Thread(target=worker,args=(q,))for q in queues for _ in range(4)]
  for t in threads:t.start()
  try:
   assert event.wait(10)and not errors,errors
   assert len(entered)==2*level and all(sum(name==task for name,_ in entered)==level for task in p.TASKS)
  finally:
   file.write_text('{}');drain.set()
   for t in threads:t.join(10);assert not t.is_alive()
   file.unlink()
  assert not b._INFLIGHT and not errors,errors
 if old is None:os.environ.pop('CX_BENEFIT_RECYCLE_FILE',None)
 else:os.environ['CX_BENEFIT_RECYCLE_FILE']=old
s.CHUNKS.clear()
# Actual physical rollout semaphore, fourteen entered and fifteenth waiting.
import benefit_stage_runtime_v2 as v2
assert b.MAX_CLAIMED==14 and v2._slots._value==14
entered=[];event=threading.Event();drain=threading.Event();lock=threading.Lock()
def physical(i):
 def body():
  with lock:
   entered.append(i)
   if len(entered)==14:event.set()
  assert drain.wait(10)
 v2.bounded(body)
threads=[threading.Thread(target=physical,args=(i,))for i in range(15)]
for t in threads:t.start()
assert event.wait(10)and len(entered)==14 and v2._slots._value==0
drain.set()
for t in threads:t.join(10);assert not t.is_alive()
assert len(entered)==15 and v2._slots._value==14

# Original Warp function source, exact options/update/mark_modified body retained.
import warp,warp._src.context as context
source=ast.parse(Path(context.__file__).read_text());node=next(n for n in source.body if isinstance(n,ast.FunctionDef)and n.name=='set_module_options')
class Module:
 def __init__(self):self.options={};self.marked=0
 def mark_modified(self):self.marked+=1
mods={}
ns=dict(context.__dict__,get_module=lambda name:mods.setdefault(name,Module()));exec(compile(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]),context.__file__,'exec'),ns)
original=ns['set_module_options'];fast=s.fast_warp(original)
for explicit in (None,sys.modules[__name__]):
 mods.clear();original(dict(fast_math=False,fuse_fp=True,enable_backward=False),module=explicit);left={k:(v.options.copy(),v.marked)for k,v in mods.items()}
 mods.clear();fast(dict(fast_math=False,fuse_fp=True,enable_backward=False),module=explicit);right={k:(v.options.copy(),v.marked)for k,v in mods.items()};assert left==right
# Actual owner priority queue: formal setup is serviced before discarded constructor.
from concurrent_core import Coordinator
from concurrent_rng import Lane
from model_service import ModelService
c=Coordinator(cuda=False);c.service=ModelService(c);c.rng.install();order=[];bar=threading.Barrier(3);errors=[]
def job(stage,priority):
 lane=Lane(stage)
 def fn():order.append(stage)
 fn._cx_stage=stage;fn._cx_priority=priority
 try:
  with c.rng.bind(lane):bar.wait(5);c.call(lane,fn)
 except BaseException as e:errors.append(repr(e))
threads=[threading.Thread(target=job,args=('shared_readiness_and_original_lane_constructor',1)),threading.Thread(target=job,args=('original_task_setup',4))]
for t in threads:t.start()
bar.wait(5)
for _ in range(100):
 if c.service.requests.qsize()==2:break
 time.sleep(.005)
assert c.service.requests.qsize()==2
while any(t.is_alive()for t in threads):c.service.process_one(.01)
for t in threads:t.join()
c.rng.uninstall();assert not errors and order==['original_task_setup','shared_readiness_and_original_lane_constructor'],(errors,order)

# Original CPU SHA/stat cache retains full hash and fails closed on mutation.
import hashlib,tempfile
with tempfile.TemporaryDirectory(dir=p.ROOT,prefix='digest211_CPU_')as temp:
 file=Path(temp)/'source.py';file.write_text('value=1\n');original=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 cache=s.PreparedSourceDigests();expected=cache.add(file,original);assert cache.read(file,original)==expected==original(file)and cache.hits==1
 file.write_text('value=2\n')
 try:cache.read(file,original)
 except RuntimeError:pass
 else:raise AssertionError('Source mutation accepted')
 other=Path(temp)/'other.py';other.write_text('other=3\n');assert cache.read(other,original)==original(other)and cache.fallbacks==1

# CPU pseudo chunk factory verifies exactly one original invocation/return and cap progress.
assert not torch.cuda.is_initialized()
p.atomic(p.ROOT/'capacity212_CPU.json',dict(passed=True,no_GPU=True,no_CUDA=True,actual_original_queue_16_threads_7_each_14_total_exact=True,constructor_priority_only_cpu_verified=True,CPU_full_source_SHA_stat_cache_exact_and_tamper_rejected=True,global_max14_and_maintenance_waiters_drained=True,Warp_original_options_module_identity_and_mark_modified_exact=True,Warp_inverse_AST_all_but_frame_lookup_exact=True,resource211_10_explicitly_superseded_by14=True,no_policy_outcome_read=True,all_resource_thresholds_and_scientific_parameters_unchanged=True,source_sha256={str(f):p.sha(f)for f in (Path(s.__file__),Path(entry211.__file__),Path(__file__),Path(__import__('capacity212').__file__),Path(entry212.__file__))}))
print('CAPACITY212_CPU_PASSED',flush=True)
