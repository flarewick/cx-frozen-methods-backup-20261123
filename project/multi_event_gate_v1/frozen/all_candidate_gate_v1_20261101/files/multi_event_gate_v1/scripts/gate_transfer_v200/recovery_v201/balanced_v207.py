"""Five genuine claimed episodes per task, ten total; no outcome selection."""
import ast,copy,inspect,linecache,textwrap,time
from pathlib import Path
_APPLIED=False

def full(queue):
 import benefit_followup_protocol_v2 as p,benefit_memory_batches as b
 task,branch=p.decode(queue.manifest['condition'])
 return sum(Path(run).parent.name==task for run,seed in b._INFLIGHT)>=5

def patch(function):
 original=ast.parse(textwrap.dedent(inspect.getsource(function)));tree=copy.deepcopy(original);changes=[]
 class Edit(ast.NodeTransformer):
  def visit_Compare(self,node):
   if ast.unparse(node)in ('len(_INFLIGHT) >= MAX_CLAIMED','len(batches._INFLIGHT) >= batches.MAX_CLAIMED'):
    changes.append(ast.dump(node));return ast.copy_location(ast.BoolOp(op=ast.Or(),values=[node,ast.Call(func=ast.Name(id='_transfer_task_full207',ctx=ast.Load()),args=[ast.Name(id='queue',ctx=ast.Load())],keywords=[])]),node)
   return self.generic_visit(node)
 Edit().visit(tree);assert len(changes)==2
 restored=copy.deepcopy(tree)
 class Undo(ast.NodeTransformer):
  def visit_BoolOp(self,node):
   if len(node.values)==2 and isinstance(node.values[1],ast.Call)and ast.unparse(node.values[1].func)=='_transfer_task_full207':return node.values[0]
   return self.generic_visit(node)
 Undo().visit(restored);assert ast.dump(restored)==ast.dump(original)
 namespace=dict(function.__globals__);namespace.update({k:c.cell_contents for k,c in zip(function.__code__.co_freevars,function.__closure__ or ())});namespace['_transfer_task_full207']=full
 text=ast.unparse(ast.fix_missing_locations(tree))+'\n';label='<balanced_task_admission207_'+function.__module__+'>';linecache.cache[label]=(len(text),None,text.splitlines(True),label);exec(compile(text,label,'exec'),namespace);return namespace[function.__name__]

def apply():
 global _APPLIED
 if _APPLIED:return
 import benefit_memory_batches as b
 from concurrent_queue import EpisodeQueue
 from concurrent_core import closure_set
 assert not b._INFLIGHT and b.MAX_CLAIMED==10
 wrapper=b.acquire;assert wrapper.__name__=='take'
 original=wrapper.__closure__[wrapper.__code__.co_freevars.index('acquire')].cell_contents
 closure_set(wrapper,'acquire',patch(original));EpisodeQueue.acquire=patch(EpisodeQueue.acquire)
 _APPLIED=True

def main():
 import os,tempfile,threading,json,torch,types
 import benefit_followup_protocol_v2 as p,benefit_growth_runtime as runtime
 assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized();torch.set_num_threads(1)
 runtime.install()
 import benefit_memory_batches as b,benefit_followup_mixed_core_v3 as core
 from concurrent_queue import EpisodeQueue
 assert __import__('balanced_v207')._APPLIED and not b._INFLIGHT
 with tempfile.TemporaryDirectory(dir=p.ROOT,prefix='explicit_CPU_balance_fixture_')as tmp:
  old=os.environ.get('CX_BENEFIT_RECYCLE_FILE');file=Path(tmp)/'drain.json';os.environ['CX_BENEFIT_RECYCLE_FILE']=str(file)
  queues=[]
  for task in p.TASKS:
   rows=core.branch_rows(task,'A0');assert sum(r['eligible']for r in rows)>=6
   queues.append(types.SimpleNamespace(run=Path(tmp)/task/'A0',manifest=dict(condition=p.condition(task,'A0'),episodes=20),lock=threading.Condition(),state=dict(attempts=rows,completed={}),claimed=set()))
  ten=threading.Event();drain=threading.Event();lock=threading.Lock();claimed=[];errors=[]
  def worker(q):
   try:
    row=EpisodeQueue.acquire(q,None,threading.Event())
    if row is None:return
    with lock:
     claimed.append((q.run.parent.name,row['seed']))
     if len(claimed)==10:ten.set()
    assert drain.wait(10)
    b.release(q,row)
   except BaseException as e:errors.append(repr(e));ten.set()
  threads=[threading.Thread(target=worker,args=(q,))for q in queues for _ in range(7)]
  for t in threads:t.start()
  try:
   assert ten.wait(10)and not errors,errors
   assert len(claimed)==10 and all(sum(t==task for t,seed in claimed)==5 for task in p.TASKS)
   assert all(full(q)for q in queues)
  finally:
   file.write_text('{}');drain.set()
   for t in threads:t.join(10);assert not t.is_alive()
   if old is None:os.environ.pop('CX_BENEFIT_RECYCLE_FILE',None)
   else:os.environ['CX_BENEFIT_RECYCLE_FILE']=old
  assert not errors and not b._INFLIGHT,errors
 p.atomic(p.ROOT/'balanced_task_admission_CPU_v207.json',dict(passed=True,CPU_only=True,no_CUDA=True,helper_sha256=p.sha(__file__),actual_original_incomplete_queue_acquire_14_threads=True,exactly5_real_claims_per_task10_total=True,waiters_and_original_maintenance_drain_verified=True,only_two_capacity_predicates_extended_per_acquire_inverse_AST_exact=True,no_success_or_intervention_outcome_read=True,all_server_thresholds_unchanged=True))
 print('BALANCED207_CPU_PASSED',flush=True)
if __name__=='__main__':main()
