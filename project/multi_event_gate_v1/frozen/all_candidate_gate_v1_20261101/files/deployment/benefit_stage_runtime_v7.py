"""Original base graphs initialize first; checked prefix replay follows warmup."""
import ast
import benefit_stage_runtime_v4 as prior
from benefit_stage_protocol import D,sha

def install_sources():
 prior.install_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_stage_v7_sources',False):return
 before=backend.sources
 backend.sources=lambda:dict(before(),**{'benefit_prefix_graph.py':sha(D/'benefit_prefix_graph.py'),'benefit_stage_runtime_v7.py':sha(__file__)})
 backend._stage_v7_sources=True

def after_original_warmup(coordinator,original,activate):
 def finish(*args,**kwargs):
  result=original(*args,**kwargs)
  activate(coordinator)
  return result
 return finish

def activate(c):
 import torch,benefit_prefix_graph
 c._stage_prefix_encoder=torch.compiler.disable(benefit_prefix_graph.build_encode())
 c._stage_prefix_ready=True

def install():
 install_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_stage_v7_installed',False):return
 prior.install()
 import shared6_eval_core as core
 predict=core.AuditedPolicyGate.predict
 def checked(self,*args,**kwargs):
  c=self._coordinator
  if getattr(c,'_stage_prefix_ready',False):self._wrapped.encode_eager=c._stage_prefix_encoder
  return predict(self,*args,**kwargs)
 core.AuditedPolicyGate.predict=checked
 configure=core.configure
 def wrapped(callbacks,c,lane):
  result=configure(callbacks,c,lane)
  if not getattr(c,'_stage_prefix_activation_hook',False):
   if c.cache_export is None:raise RuntimeError('Original model warmup/export callback is required')
   c.cache_export=after_original_warmup(c,c.cache_export,activate)
   c._stage_prefix_activation_hook=True
  return result
 core.configure=wrapped
 backend._stage_v7_installed=True

def main():
 from shared6_eval_prepared import main as prelude
 import benefit_stage_runtime as stage
 def edit(node):
  index=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='torch.set_num_threads')
  node.body[index+1:index+1]=ast.parse("__import__('benefit_stage_runtime_v7').install()").body
 stage.transform(prelude,edit)()
if __name__=='__main__':main()
