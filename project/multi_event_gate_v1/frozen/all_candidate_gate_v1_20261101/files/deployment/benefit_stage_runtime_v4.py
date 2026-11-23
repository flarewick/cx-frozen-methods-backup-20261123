"""Shared sampler warmup plus each lane's required original discarded constructor."""
import ast,json
import benefit_stage_runtime_v3 as prior
from benefit_stage_protocol import D,ROOT,sha
_base_probe=prior.shared_probe

def install_sources():
 prior.install_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_stage_v4_sources',False):return
 before=backend.sources
 backend.sources=lambda:dict(before(),**{'benefit_stage_runtime_v4.py':sha(__file__)})
 backend._stage_v4_sources=True

def initialize(c,lane,args):
 from cx_tta.evaluate_b2 import seed_all
 from envs.utils.create_actor import UnStableError
 task,config,seed,instruction,wrappers=args
 name=type(task).__name__;q=json.loads((ROOT/'qualified.json').read_text())
 seed=100003 if name=='dump_bin_bigbin'else next(r['seed']for r in q['tasks'][name]['attempts']if r['eligible'])
 phase=getattr(lane,'phase',None);lane.phase='discarded_probe'
 try:
  for attempt in range(1,6):
   seed_all(seed)
   try:
    with c.scene_gate.write():task.setup_demo(now_ep_num=0,seed=seed,**dict(config,need_plan=True,save_data=False,is_test=True))
    c.note('shared_model_discarded_constructor_only',lane=lane.name,seed=seed,attempt=attempt,sampler_queries=0,policy_actions=0)
    break
   except UnStableError:
    if attempt==5:raise
   finally:
    with c.scene_gate.write():task.close_env(clear_cache=False)
 finally:
  lane.phase=phase
  for policy in wrappers.values():policy.reset_history()

def compose(c,lane,previous,metadata,initializer):
 base=_base_probe(c,lane,previous,metadata)
 def probe(*args,**kwargs):
  def execute():
   already=hasattr(c,'_benefit_shared_probe_receipt')
   if already:initializer(args)
   result=base(*args,**kwargs)
   if already:result.update(discarded_constructor_only_initialized=True,discarded_sampler_queries=0)
   return result
  execute._cx_priority=1;execute._cx_stage='shared_readiness_and_original_lane_constructor'
  return c.call(lane,execute)
 return probe

def install():
 install_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_stage_v4_installed',False):return
 prior.shared_probe=lambda c,lane,previous,metadata:compose(c,lane,previous,metadata,lambda args:initialize(c,lane,args))
 prior.install();backend._stage_v4_installed=True

def main():
 from shared6_eval_prepared import main as prelude
 def edit(node):
  index=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='torch.set_num_threads')
  node.body[index+1:index+1]=ast.parse("__import__('benefit_stage_runtime_v4').install()").body
 prior.prior.stage.transform(prelude,edit)()
if __name__=='__main__':main()
