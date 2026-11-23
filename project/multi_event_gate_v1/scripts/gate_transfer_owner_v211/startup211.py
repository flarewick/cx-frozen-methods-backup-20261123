"""Work-progress admission ramp and exact Warp caller lookup; no tensor changes."""
import ast,copy,functools,inspect,json,sys,threading,textwrap,time
from pathlib import Path
CHUNKS={};LOCK=threading.Lock();INSTALLED=False

def cap(task):
 with LOCK:return min(5,1+CHUNKS.get(task,0))

def full(queue):
 import benefit_memory_batches as b,benefit_followup_protocol_v2 as p
 task,branch=p.decode(queue.manifest['condition'])
 return sum(Path(run).parent.name==task for run,seed in b._INFLIGHT)>=cap(task)

def caller_frame():return sys._getframe(2)

def fast_warp(function):
 original=ast.parse(textwrap.dedent(inspect.getsource(function)));tree=copy.deepcopy(original);hits=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Call)and ast.unparse(n)=='inspect.getmodule(inspect.stack()[1][0])':
   hits.append(n);n.args=[ast.Call(func=ast.Name(id='_warp_caller_frame211',ctx=ast.Load()),args=[],keywords=[])]
 assert len(hits)==1
 restored=copy.deepcopy(tree)
 for n in ast.walk(restored):
  if isinstance(n,ast.Call)and ast.unparse(n)=='inspect.getmodule(_warp_caller_frame211())':n.args=ast.parse('inspect.getmodule(inspect.stack()[1][0])',mode='eval').body.args
 assert ast.dump(restored)==ast.dump(original)
 ns=dict(function.__globals__,_warp_caller_frame211=caller_frame)
 exec(compile(ast.fix_missing_locations(tree),function.__code__.co_filename,'exec'),ns)
 return ns[function.__name__]

def install():
 global INSTALLED
 if INSTALLED:return
 import warp,warp._src.context as context
 original=context.set_module_options;patched=fast_warp(original)
 assert warp.set_module_options is original
 context.set_module_options=warp.set_module_options=patched
 from model_service import ModelService
 before=ModelService.call
 @functools.wraps(before)
 def call(self,lane,fn,*args,**kwargs):
  if getattr(fn,'_cx_stage',None)=='shared_readiness_and_original_lane_constructor':
   assert fn._cx_priority==1;fn._cx_priority=5
  return before(self,lane,fn,*args,**kwargs)
 ModelService.call=call
 install_CPU_source_reads()
 INSTALLED=True

class PreparedSourceDigests:
 """CPU full SHA, then fail-closed same-file stat reuse during factory setup."""
 def __init__(self):self.rows={};self.hits=0;self.fallbacks=0
 @staticmethod
 def signature(path):
  st=Path(path).stat();return(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
 def add(self,path,digest):
  path=Path(path);before=self.signature(path);value=digest(path);after=self.signature(path)
  if before!=after:raise RuntimeError('Source changed during full CPU SHA')
  self.rows[str(path)]=(after,value);return value
 def read(self,path,original):
  row=self.rows.get(str(Path(path)))
  if row is None:self.fallbacks+=1;return original(path)
  if self.signature(path)!=row[0]:raise RuntimeError('CPU-verified source file changed before/during GPU callback')
  self.hits+=1;return row[1]

def install_CPU_source_reads():
 import torch,shared6_eval_worker as worker
 previous_cpu=worker.prepare_cpu
 def prepare(*args,**kwargs):
  value=previous_cpu(*args,**kwargs)
  assert not torch.cuda.is_initialized()
  from cx_tta.audit import digest
  cache=PreparedSourceDigests();began=time.monotonic()
  for row in value['lanes']:
   root=row['runtime'];actual={str(f.relative_to(root)):cache.add(f,digest)for f in root.rglob('*.py')}
   assert actual==row['job']['manifest']['task_runtime_sha256'],'Original full runtime SHA differs'
  value['CPU_source_digests211']=cache
  if value['master']:
   import benefit_followup_protocol_v2 as p
   p.atomic(value['master'].run/'startup211_CPU_source_digest_receipt.json',dict(passed=True,CPU_only=True,no_CUDA=True,files=len(cache.rows),seconds=time.monotonic()-began,original_full_SHA_compared_to_every_lane_manifest=True,all_new_GPU_session_file_stats_fail_closed=True,helper_sha256=p.sha(__file__)))
  return value
 worker.prepare_cpu=prepare
 previous_gpu=worker.run_gpu
 def run_gpu(args,prepared):
  from task_adapters import evaluate
  cache=prepared['CPU_source_digests211'];original=evaluate.digest
  def read(path):return cache.read(path,original)
  evaluate.digest=read
  try:return previous_gpu(args,prepared)
  finally:evaluate.digest=original
 worker.run_gpu=run_gpu
