"""Keep original tensor math; shared model probe, checked Python validation, weak clocks."""
import ast,copy,weakref,sys
import benefit_stage_runtime_v2 as prior
from benefit_stage_protocol import D,ROOT,sha,atomic

def install_sources():
 prior.install_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_stage_v3_sources',False):return
 before=backend.sources
 backend.sources=lambda:dict(before(),**{n:sha(D/n)for n in ('benefit_stage_runtime_v3.py','benefit_stage_typecheck.py')})
 backend._stage_v3_sources=True

def clock(task):
 # Neither a bound Scene.step nor the task is retained by its own Scene.
 scene=task.scene;method=scene.step
 if not hasattr(method,'__func__'):raise RuntimeError('Scene.step is not the verified Python method')
 original=method.__func__;scene_ref=weakref.ref(scene);task_ref=weakref.ref(task)
 def step(*args,**kwargs):
  current=scene_ref();owner=task_ref()
  if current is None or owner is None:raise RuntimeError('Closed scene clock used')
  result=original(current,*args,**kwargs)
  if owner._pilot_recording:owner._pilot_ticks+=1
  return result
 scene.step=step

def shared_probe(c,lane,previous,metadata):
 def probe(*args,**kwargs):
  def execute():
   if not hasattr(c,'_benefit_shared_probe_receipt'):
    quantum=sys.getswitchinterval();sys.setswitchinterval(.005)
    try:result=previous(*args,**dict(kwargs,maximum=6))
    finally:sys.setswitchinterval(round(quantum*1e6)/1e6)
    if not result['passed']:raise RuntimeError('Shared original L0/zero sampler parity failed')
    receipt={k:v for k,v in result.items()if k!='base_values'}
    receipt.update(cpu_switch_interval_during_original_probe=.005,after_probe_switch_interval=sys.getswitchinterval(),shared_immutable_model=True,original_probe_executed=True,first_lane=lane.name,policy_actions=0)
    file=metadata/'shared_model_warmup.json';atomic(file,receipt)
    c._benefit_shared_probe_receipt=dict(file=str(file),sha256=sha(file),first_lane=lane.name)
    return result
   # This is a receipt about an already checked model, not reused predictions
   # and not a claim that a new observation's L0/zero probe was performed.
   receipt=c._benefit_shared_probe_receipt
   return dict(passed=True,base_values=None,original_probe_executed=False,shared_immutable_model=True,shared_probe_reference=receipt,scope='Shared frozen model already passed its original L0/zero test; this lane skips redundant discarded warmup. Every formal scene is recreated and every formal query is recomputed, with original three-query GPU parity per condition.',policy_actions=0)
  execute._cx_priority=1;execute._cx_stage='shared_immutable_model_probe_admission'
  return c.call(lane,execute)
 return probe

def install():
 install_sources()
 import shared6_eval_backend as backend,shared6_eval_worker as worker
 if getattr(backend,'_stage_v3_installed',False):return
 prior.install()
 import benefit_pilot_trace as trace
 trace.install_scene_clock=clock
 before=worker.prepare_cpu
 def prepare(args):
  prepared=before(args)
  from benefit_stage_typecheck import install as fast
  if prepared['lanes']:fast()
  return prepared
 worker.prepare_cpu=prepare
 import shared6_eval_core as core
 from concurrent_core import closure_set
 configure=core.configure
 def wrapped(callbacks,c,lane):
  result=configure(callbacks,c,lane)
  ordered=result['rollout'];rollout=ordered.__globals__['rollout']
  previous=rollout.__closure__[rollout.__code__.co_freevars.index('steady_probe')].cell_contents
  closure_set(rollout,'steady_probe',shared_probe(c,lane,previous,c.parity_path.parent))
  return result
 core.configure=wrapped
 backend._stage_v3_installed=True

def main():
 from shared6_eval_prepared import main as prelude
 def edit(node):
  index=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='torch.set_num_threads')
  node.body[index+1:index+1]=ast.parse("__import__('benefit_stage_runtime_v3').install()").body
 prior.stage.transform(prelude,edit)()
if __name__=='__main__':main()
