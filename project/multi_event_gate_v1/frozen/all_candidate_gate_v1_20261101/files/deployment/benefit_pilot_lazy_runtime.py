"""Reuse proven deferred expert planners for the frozen-qualification pilot."""
import ast,json
import benefit_pilot_fast_runtime as fast
from benefit_pilot_protocol import D,sha

def install_sources():
 fast.install_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_pilot_lazy_sources',False):return
 before=backend.sources
 backend.sources=lambda:dict(before(),**{n:sha(D/n)for n in('benefit_pilot_lazy_runtime.py','deferred_planner.py')})
 decorate=backend.decorate
 def wrapped(*args,**kwargs):
  from cx_tta.evaluation_resume import identity
  value=decorate(*args,**kwargs)
  value['deployment_pilot_planner']=dict(protocol='deferred_original_expert_pair_for_fixed_qualified_scenes',mutable_policy_access=False,qualification_from_frozen_cache=True,source_sha256=sha(D/'deferred_planner.py'))
  value['evaluation_identity']=identity(value);return value
 backend.decorate=wrapped;backend._pilot_lazy_sources=True

def install():
 fast.install();install_sources()
 import shared6_planner_pool as pool
 from deferred_planner import install as deferred
 pool.install=deferred

def main():
 from shared6_eval_prepared import main as prelude
 def edit(node):
  index=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='torch.set_num_threads')
  node.body[index+1:index+1]=ast.parse("__import__('benefit_pilot_lazy_runtime').install()").body
 fast.original.transform(prelude,edit)()
if __name__=='__main__':main()
