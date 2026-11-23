"""Six real active scenes, dynamic A0/B throughput, original50 files intact."""
import ast,threading
import benefit_stage_runtime as stage
from benefit_stage_protocol import D,ROOT,sha
_slots=threading.BoundedSemaphore(6)
def install_sources():
 stage.install_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_stage_v2_sources',False):return
 before=backend.sources
 backend.sources=lambda:dict(before(),**{'benefit_stage_runtime_v2.py':sha(__file__)})
 backend._stage_v2_sources=True

def bounded(rollout,*args,**kwargs):
 # Acquire AFTER predecessor waits, and include discarded probes/formal scenes.
 with _slots:return rollout(*args,**kwargs)

def install():
 install_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_stage_v2_installed',False):return
 def allocation(node):
  class Edit(ast.NodeTransformer):
   def visit_Assign(self,n):
    if any(ast.unparse(t)=='lane_count'for t in n.targets):
     n.value=ast.parse("(2 if branch=='G'else 1 if branch=='A1'else 3)",mode='eval').body
    if any(ast.unparse(t)=="manifest['deployment_shared6']['condition_lanes']"for t in n.targets):
     return[n,*ast.parse("manifest['deployment_benefit_scene_slots']=dict(max_active=6,condition_lanes=lane_count,predecessor_wait_outside_slots=True,policy_actions_unchanged=True)").body]
    return n
  Edit().visit(node)
 stage.make_plan=stage.transform(stage.make_plan,allocation)
 stage.install()
 import shared6_eval_core as core
 configure=core.configure;stage._stage_bounded=bounded
 def wrapped(*args,**kwargs):
  result=configure(*args,**kwargs)
  def edit(node):
   hits=[n for n in ast.walk(node)if isinstance(n,ast.Call)and ast.unparse(n.func)=='rollout']
   if len(hits)!=1:raise RuntimeError('Original ordered rollout boundary changed')
   hits[0].args.insert(0,ast.Name(id='rollout',ctx=ast.Load()));hits[0].func=ast.Name(id='_stage_bounded',ctx=ast.Load())
  result['rollout']=stage.transform(result['rollout'],edit)
  return result
 core.configure=wrapped;backend._stage_v2_installed=True

def main():
 from shared6_eval_prepared import main as prelude
 def edit(node):
  index=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='torch.set_num_threads')
  node.body[index+1:index+1]=ast.parse("__import__('benefit_stage_runtime_v2').install()").body
 stage.transform(prelude,edit)()
if __name__=='__main__':main()
