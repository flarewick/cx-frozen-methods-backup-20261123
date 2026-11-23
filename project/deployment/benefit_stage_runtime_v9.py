"""Six original concurrent scenes; exact early-height success shortcircuit."""
import ast,functools
import benefit_stage_runtime_v7 as prior
from benefit_stage_protocol import D,sha

def install_sources():
 prior.install_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_stage_v9_sources',False):return
 before=backend.sources
 backend.sources=lambda:dict(before(),**{'benefit_success_shortcircuit.py':sha(D/'benefit_success_shortcircuit.py'),'benefit_stage_runtime_v9.py':sha(__file__)})
 backend._stage_v9_sources=True

def after_setup(task,setup):
 @functools.wraps(setup)
 def call(*args,**kwargs):
  result=setup(*args,**kwargs)
  from benefit_success_shortcircuit import install as early
  early(task)
  return result
 return call

def install():
 install_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_stage_v9_installed',False):return
 prior.install()
 import shared6_eval_core as core
 original=core.Shared6Coordinator.prepare_task
 def prepare(self,task,lane):
  result=original(self,task,lane)
  task.setup_demo=after_setup(task,task.setup_demo)
  return result
 core.Shared6Coordinator.prepare_task=prepare
 backend._stage_v9_installed=True

def main():
 from shared6_eval_prepared import main as prelude
 import benefit_stage_runtime as stage
 def edit(node):
  index=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='torch.set_num_threads')
  node.body[index+1:index+1]=ast.parse("__import__('benefit_stage_runtime_v9').install()").body
 stage.transform(prelude,edit)()
if __name__=='__main__':main()
