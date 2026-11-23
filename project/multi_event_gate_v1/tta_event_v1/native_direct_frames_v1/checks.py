"""Real installed APIs, original checks, generated frames, unprofiled CPU timing."""
from pathlib import Path
import sys,inspect,types,time,dataclasses,random,statistics
from native_policy_execution_v1.common import T,read,write,sha
from .frames import candidates,install

def main():
 import torch,numpy as np
 torch.set_num_threads(1)
 project=T.parents[1]
 rt=Path('/data/cj/robo_pi/250010033/qfw/robotwin')
 sys.path.insert(0,str(rt/'RoboTwin/policy/pi05/src'))
 import warp as wp
 import warp._src.context as context
 import jaxtyping,jaxtyping._decorator as decorator,beartype
 from openpi.shared import array_typing
 old_type=array_typing._check_dataclass_annotations;old_warp=wp.set_module_options
 new_type,new_warp=candidates()
 cpu=torch.get_rng_state().clone();python=random.getstate();numpy=np.random.get_state()
 original_threads=len(list(Path('/proc/self/task').iterdir()));warp_runtime=context.runtime
 x=torch.ones((2,3));before=x.clone()
 @jaxtyping.jaxtyped(typechecker=beartype.beartype)
 @dataclasses.dataclass
 class Typed:
  x:jaxtyping.Float[torch.Tensor,'2 3']
 def outcome(check,value):
  decorator._check_dataclass_annotations=check
  try:return ('value',Typed(value).x)
  except Exception as error:return ('error',type(error).__name__,str(error))
 try:
  for value in [x,torch.ones((3,3)),None]:
   a=outcome(old_type,value);b=outcome(new_type,value)
   if a[0]=='value':assert b[0]=='value'and torch.equal(a[1],b[1])
   else:assert a==b,(a,b)
 finally:decorator._check_dataclass_annotations=old_type
 # Check every original whitelist branch without importing a GPU/JAX backend.
 token=object();calls=[]
 def stub(*args):calls.append(args);return token
 def stubbed(function):
  return types.FunctionType(function.__code__,dict(function.__globals__,_original_check_dataclass_annotations=stub),function.__name__,function.__defaults__,function.__closure__)
 old_stub,new_stub=stubbed(old_type),stubbed(new_type)
 for module,skip in [('jax._src.tree_util',True),('flax.nnx.transforms.compilation',True),('unrecognized.module',False)]:
  namespace={'__name__':module};exec(compile('def invoke(fn):\n return fn(None,None)\n','<CPU_whitelist_case>','exec'),namespace)
  for fn in [old_stub,new_stub]:
   count=len(calls);value=namespace['invoke'](fn)
   assert (value is None and len(calls)==count)if skip else(value is token and len(calls)==count+1)
 # Keep the original inspect.getmodule result, all options, and mark_modified.
 module=inspect.getmodule(sys._getframe());assert module is not None
 actual=context.get_module(module.__name__);saved=dict(actual.options)
 cases=[]
 for explicit in [False,True]:
  outcomes=[]
  for fn in [old_warp,new_warp]:
   actual.options=dict(saved);actual.hashers={'CPU':object()};actual.failed_builds={'CPU'}
   result=fn({'enable_backward':False},module=module)if explicit else fn({'enable_backward':False})
   outcomes.append((result,dict(actual.options),dict(actual.hashers),set(actual.failed_builds)))
  assert outcomes[0]==outcomes[1]and not outcomes[0][2]and not outcomes[0][3]
  cases.append(dict(explicit_module=explicit,original_options_and_invalidation_exact=True))
 # Generated ancestor frames reproduce expensive missing-source stack walks.
 namespace={'__name__':__name__}
 exec(compile('def generated(fn,*args):\n return fn(*args)\n','<CPU_generated_native_call>','exec'),namespace)
 generated=namespace['generated']
 def nested(depth,fn):
  if depth:return generated(nested,depth-1,fn)
  return fn()
 timings={};iterations=16;depth=12
 for name,old,new in [('typing',lambda:old_stub(None,None),lambda:new_stub(None,None)),('warp',lambda:old_warp({'enable_backward':False}),lambda:new_warp({'enable_backward':False}))]:
  rows=[]
  for label,fn in [('original',old),('candidate',new),('candidate',new),('original',old)]:
   started=time.perf_counter()
   for _ in range(iterations):nested(depth,fn)
   rows.append(dict(kind=label,seconds=time.perf_counter()-started))
  baseline=statistics.mean(r['seconds']for r in rows if r['kind']=='original');candidate=statistics.mean(r['seconds']for r in rows if r['kind']=='candidate')
  timings[name]=dict(iterations=iterations,generated_depth=depth,ABBA=rows,original_seconds=baseline,candidate_seconds=candidate,CPU_speedup=baseline/candidate)
 actual.options=saved
 install()
 assert decorator._check_dataclass_annotations is array_typing._check_dataclass_annotations and wp.set_module_options is context.set_module_options
 assert torch.equal(cpu,torch.get_rng_state())and torch.equal(x,before)and python==random.getstate()
 after=np.random.get_state();assert numpy[0]==after[0]and np.array_equal(numpy[1],after[1])and numpy[2:]==after[2:]
 assert not torch.cuda.is_initialized()and context.runtime is warp_runtime
 assert original_threads==len(list(Path('/proc/self/task').iterdir()))
 files=[Path(__file__),Path(__file__).with_name('frames.py'),Path(__file__).with_name('AGENTS.md'),Path(inspect.getsourcefile(old_type)),Path(inspect.getsourcefile(old_warp))]
 out=T/'outputs/native_direct_frames_v1';out.mkdir(parents=True,exist_ok=True)
 write(out/'CPU_contract_checks_v1.json',dict(passed=True,real_typed_Torch_dataclass_valid_invalid_exception_exact=True,
  complete_JAX_Flax_whitelist_and_unknown_checks_preserved=True,actual_Warp_options_and_mark_modified_exact=cases,
  AST_exact_except_unused_stack_source_materialization=True,Torch_numpy_python_RNG_tensor_exact=True,
  original_OS_threads=original_threads,actual_OS_threads=len(list(Path('/proc/self/task').iterdir())),CUDA_initialized=False,Warp_runtime_unchanged=True,
  loaded_modules=len(sys.modules),unprofiled_CPU_timings=timings,no_GPU_or_native_or_TTA_efficacy_claim=True,
  source_sha256={str(f):sha(f)for f in files}))
 print('DIRECT_FRAMES_REAL_API_CPU_PASS',timings,flush=True)

if __name__=='__main__':main()
