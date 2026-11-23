"""Host workflow: admit only ready fixed scenes; drain on memory maintenance.

The original scientific rollouts, sampler, EpisodeQueue.complete and validators
remain in force. A partial batch is explicitly incomplete, never a result claim.
"""
import ast,copy,inspect,json,os,textwrap,threading,time
from pathlib import Path
MAX_CLAIMED=6
_INFLIGHT=set();_LOCK=threading.Lock()
def request_path():
 value=os.environ.get('CX_BENEFIT_RECYCLE_FILE');return Path(value)if value else None
def requested():
 p=request_path();return p is not None and p.exists()
_CACHE_LOCK=threading.Lock();_PLAN_PATHS=None;_DONE={}
def predecessor(queue,row):
 global _PLAN_PATHS
 from benefit_stage_protocol import ROOT,decode,selected
 task,branch=decode(queue.manifest['condition'])
 if branch in ('A0','G'):return True
 prior='A1'if branch=='B'and selected('A1',row['qualified_index'])else'A0'
 with _CACHE_LOCK:
  if _PLAN_PATHS is None:
   plan=json.loads((ROOT/'collect_plan.json').read_text())
   _PLAN_PATHS={(j['task'],j['branch']):Path(j['run'])/'concurrency_queue.json'for j in plan['jobs']}
  path=_PLAN_PATHS[task,prior];now=time.monotonic();entry=_DONE.get(path)
  if entry is None or now-entry[0]>=.25:
   entry=(now,set(json.loads(path.read_text())['completed']));_DONE[path]=entry
  return str(row['seed'])in entry[1]

def acquire(queue,screen,cancel,ready=predecessor):
 # Called only for the frozen prequalified benefit cohort; no new screening.
 eligible=[r for r in queue.state['attempts']if r['eligible']]
 if len(eligible)!=queue.manifest['episodes']:raise RuntimeError('Maintenance needs all frozen qualified rows')
 with queue.lock:
  while True:
   if cancel.is_set():raise RuntimeError('Queue cancelled by peer/supervisor')
   if requested():return None
   with _LOCK:full=len(_INFLIGHT)>=MAX_CLAIMED
   if full:queue.lock.wait(.1);continue
   pending=[r for r in eligible if str(r['seed'])not in queue.state['completed']and r['seed']not in queue.claimed]
   if not pending:return None
   for row in pending:
    if not ready(queue,row):continue
    with _LOCK:
     if requested():return None
     if len(_INFLIGHT)>=MAX_CLAIMED:break
     key=(str(queue.run),row['seed'])
     if key in _INFLIGHT:raise RuntimeError('Duplicate host scene claim')
     _INFLIGHT.add(key);queue.claimed.add(row['seed'])
     return copy.deepcopy(row)
   queue.lock.wait(.05)
def release(queue,row):
 with _LOCK:
  key=(str(queue.run),row['seed'])
  if key not in _INFLIGHT:raise RuntimeError('Untracked host scene completion')
  _INFLIGHT.remove(key)
def partial_finalizer(original):
 node=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
 tests=[n for n in node.body if isinstance(n,ast.If)and ast.unparse(n.test)=="len(eps) != manifest['episodes']"]
 if len(tests)!=1:raise RuntimeError('Original complete-count guard changed')
 node.body.remove(tests[0])
 result=next(n.value for n in node.body if isinstance(n,ast.Return));assert isinstance(result,ast.Call)
 for kw in result.keywords:
  if kw.arg=='status':kw.value=ast.Constant('maintenance_checkpoint')
  if kw.arg=='complete':kw.value=ast.Constant(False)
  if kw.arg=='success_rate':kw.value=ast.IfExp(test=ast.Name(id='eps',ctx=ast.Load()),body=kw.value,orelse=ast.Constant(None))
 # Original source/runtime/evidence loops remain byte-for-byte AST-identical.
 before=[ast.dump(n,include_attributes=False)for n in ast.parse(textwrap.dedent(inspect.getsource(original))).body[0].body if isinstance(n,ast.For)]
 after=[ast.dump(n,include_attributes=False)for n in node.body if isinstance(n,ast.For)]
 if before!=after:raise RuntimeError('Original provenance validation changed')
 ns=dict(original.__globals__);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),ns)
 return ns[node.name]
def install():
 from concurrent_queue import EpisodeQueue
 if getattr(EpisodeQueue,'_host_memory_batches',False):return
 original_acquire=EpisodeQueue.acquire;original_complete=EpisodeQueue.complete
 def take(self,screen,cancel):
  from benefit_stage_protocol import decode
  try:decode(self.manifest['condition'])
  except ValueError:return original_acquire(self,screen,cancel)
  return acquire(self,screen,cancel)
 def complete(self,row,episode):
  result=original_complete(self,row,episode)
  with _LOCK:tracked=(str(self.run),row['seed'])in _INFLIGHT
  if tracked:release(self,row)
  return result
 EpisodeQueue.acquire=take;EpisodeQueue.complete=complete;EpisodeQueue._host_memory_batches=True
 import task_adapters.eval_resume as resume
 original=resume.final_result;partial=partial_finalizer(original)
 def finalize(run,manifest):
  if requested():
   state=resume.load_journal(run,manifest)
   if len(state['episodes'])!=manifest['episodes']:return partial(run,manifest)
  return original(run,manifest)
 resume.final_result=finalize
