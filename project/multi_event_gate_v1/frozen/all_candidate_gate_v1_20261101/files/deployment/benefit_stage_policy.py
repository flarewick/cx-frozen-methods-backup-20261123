"""One decision at the first event; frozen archived adapter and action math."""
import ast,inspect,textwrap,json
from benefit_pilot_policy import PilotPolicy,Original
from benefit_pilot_protocol import condition as old_condition
from benefit_stage_protocol import ROOT,decode
class StagePolicy(PilotPolicy):
 def __init__(self,base,training_run,condition,config):
  try:task,branch=decode(condition)
  except ValueError:
   super().__init__(base,training_run,condition,config);self.stage_branch=None;return
  self.stage_branch=branch;self.stage_first_query=None;self.stage_scores=None
  self.stage_model=json.loads((ROOT/'gate.json').read_text())if branch=='G'else None
  super().__init__(base,training_run,old_condition(task,'B'if branch in ('B','G')else'A0'),config)
  self.condition=condition
 def reset_history(self):
  super().reset_history();self.stage_first_query=None;self.stage_scores=None
 def stage_admit(self,q,packet,context):
  if getattr(self,'stage_branch',None)!='G':return dict(both_fail=0.,rescue=1.,harm=0.,both_success=0.)
  if self.stage_first_query is None:
   from benefit_stage_gate import predict
   self.stage_first_query=q;self.stage_scores=predict(self.stage_model,packet,context)
  if q==self.stage_first_query:return self.stage_scores.copy()
  return dict(both_fail=1.,rescue=0.,harm=0.,both_success=0.)

def install_scorer():
 if getattr(Original,'_stage_scorer',False):return
 node=ast.parse(textwrap.dedent(inspect.getsource(Original.predict))).body[0];changed=0
 for n in ast.walk(node):
  if isinstance(n,ast.FunctionDef)and n.name=='admit':n.body=ast.parse('return self.stage_admit(q,_packet,_context)').body;changed+=1
  if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='super'and not n.args:n.args=[ast.Name(id='Original',ctx=ast.Load()),ast.Name(id='self',ctx=ast.Load())];changed+=1
 if changed!=2:raise RuntimeError('Original policy admission hook changed')
 ns=dict(Original.predict.__globals__,Original=Original)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),ns)
 Original.predict=ns['predict'];Original._stage_scorer=True
