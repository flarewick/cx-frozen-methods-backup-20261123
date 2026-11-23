"""Explicit source-bound rebinding of unchanged v9 science to independent new cohorts."""
import ast,importlib,json,sys
import benefit_growth_protocol as new
from benefit_growth_protocol import D,ROOT,BASE,sha
BOUND=False
NAMES=('ROOT','TASKS','BRANCHES','VAL','REPEAT','condition','decode','split','selected','branch_rows','prepare')
def bind():
 global BOUND
 if BOUND:return
 new.prepare()
 modules=['benefit_stage_protocol','benefit_stage_runtime','benefit_stage_policy','benefit_stage_runtime_v2','benefit_stage_runtime_v3','benefit_stage_runtime_v4','benefit_stage_runtime_v7','benefit_stage_runtime_v9']
 for name in modules:
  if name=='benefit_stage_policy'and name not in sys.modules:continue
  module=importlib.import_module(name)
  for key in NAMES:
   if hasattr(module,key):setattr(module,key,getattr(new,key))
 import benefit_stage_runtime as stage
 original_transform=stage.transform
 def transform(function,mutate):
  def edit(node):
   mutate(node)
   if node.name=='make_plan':
    counts=[n for n in ast.walk(node)if isinstance(n,ast.Assign)and any(ast.unparse(t)=='count'for t in n.targets)]
    if len(counts)!=1:raise RuntimeError('Count assignment changed')
    counts[0].value=ast.parse("4 if branch=='A1' else len(__import__('benefit_growth_protocol').VAL) if branch=='G' else 20",mode='eval').body
    lanes=[n for n in ast.walk(node)if isinstance(n,ast.Assign)and any(ast.unparse(t)=='lane_count'for t in n.targets)]
    if len(lanes)!=1:raise RuntimeError('Lane allocation changed')
    lanes[0].value=ast.parse("1 if branch=='A1' else 3",mode='eval').body
    # The actual per-job episode requests stay bound to frozen cohort membership.
    for n in ast.walk(node):
     if isinstance(n,ast.keyword)and n.arg=='episodes'and isinstance(n.value,ast.IfExp):
      n.value=ast.parse("88 if mode=='collect' else 2*len(__import__('benefit_growth_protocol').VAL)",mode='eval').body
  return original_transform(function,edit)
 stage.transform=transform
 BOUND=True

def install_sources():
 bind()
 import benefit_stage_runtime_v9 as prior,shared6_eval_backend as backend
 prior.install_sources()
 if getattr(backend,'_growth_sources',False):return
 before=backend.sources
 backend.sources=lambda:dict(before(),**{n:sha(D/n)for n in ('benefit_growth_protocol.py','benefit_growth_runtime.py','benefit_growth_gate.py')})
 backend._growth_sources=True

def install():
 install_sources()
 import benefit_stage_runtime_v9 as prior
 prior.install()
 import benefit_stage_gate,benefit_growth_gate
 benefit_stage_gate.predict=benefit_growth_gate.predict

def main():
 bind()
 from shared6_eval_prepared import main as prelude
 import benefit_stage_runtime as stage
 def edit(node):
  positions=[i for i,n in enumerate(node.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='torch.set_num_threads']
  if len(positions)!=1:raise RuntimeError('Original prelude thread setup changed')
  node.body[positions[0]+1:positions[0]+1]=ast.parse("__import__('benefit_growth_runtime').install()").body
 stage.transform(prelude,edit)()
if __name__=='__main__':importlib.import_module('benefit_growth_runtime').main()
