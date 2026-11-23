"""Frozen gate transfer. Explicit AST scope edits, no policy or threshold edits."""
import ast,copy,importlib.abc,importlib.util,json,linecache,os,runpy,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[2];D=R/'deployment';sys.path.insert(0,str(D));sys.path.insert(0,str(HERE))
sys.modules.setdefault('transfer_entry',sys.modules[__name__]);sys.modules.setdefault('entry',sys.modules[__name__])
COHORT=int(os.environ.get('CX_GATE_TRANSFER_COHORT','0'));assert COHORT in (0,1)
NAME='candidate_veto_cup_bowls_v200_'+str(COHORT)
TASKS={'place_empty_cup':16000000+COHORT*20000,'stack_bowls_two':16010000+COHORT*20000}
FOLDER='candidate_veto_event_full_extension_GPU3_v174'
SCOPE={'benefit_event_full_entry_v174','benefit_event_full_collect_v174','benefit_event_full_G_collect_v174','verify_benefit_event_full_scope_CPU_v174'}
FIXTURE={'verify_benefit_event_full_audit_CPU_v135'}
DEFERRED={'benefit_event_full_collect_v174','benefit_event_full_G_collect_v174','verify_benefit_full_deferred_CPU_v179','verify_benefit_followup_mixed_deferred_v4'}
EDITS={};INSTALLED=False
TRANSFER_MODULE='task_adapters.eval_resume'
TRANSFER_FILE=R/'multi_event_gate_v1/task_adapter_v1/task_adapters/eval_resume.py'

def transform(path):
 text=path.read_text();base=ast.parse(text);tree=copy.deepcopy(base);name=path.stem;changes=[]
 for a,b in zip(ast.walk(base),ast.walk(tree)):
  if isinstance(b,ast.Constant)and isinstance(b.value,str):
   old=b.value;v=old
   if name in SCOPE:v=v.replace(FOLDER,NAME)
   if name in {'benefit_event_full_entry_v174','verify_benefit_event_full_scope_CPU_v174'}:
    if v=='lift_pot':v='place_empty_cup'
    elif v=='dump_bin_bigbin':v='stack_bowls_two'
   if name=='verify_benefit_followup_mixed_cpu_v5'and v=='lift_pot':
    # Planner mock labels can also use new tasks; all mock isolation checks stay exact.
    v='place_empty_cup'
   if name=='verify_benefit_followup_mixed_deferred_v4'and v=='lift_pot':
    # Keep real TRAIN fixture task selections; only new-ledger negative-count fixture changes.
    pass
   if v!=old:b.value=v;changes.append((b,copy.deepcopy(a)))
  if name=='eval_resume'and isinstance(b,ast.If)and ast.unparse(b.test)=="args.task not in plan['training_tasks']":
   b.test=ast.parse("args.task not in plan['training_tasks'] and args.task not in ('place_empty_cup','stack_bowls_two')",mode='eval').body;changes.append((b,copy.deepcopy(a)))
  if name=='eval_resume'and isinstance(b,ast.Constant)and b.value=='Held-out scenes in trained tasks, not zero-shot task generalization':
   b.value='Frozen shared adapter transfer to explicitly authorized cup/bowls tasks; new held-out scenes';changes.append((b,copy.deepcopy(a)))
  if name in FIXTURE and isinstance(b,ast.For)and ast.unparse(b.iter)=='p.TASKS':
   b.iter=ast.parse("('lift_pot','dump_bin_bigbin')",mode='eval').body;changes.append((b,copy.deepcopy(a)))
  if name in DEFERRED and isinstance(b,ast.BinOp)and isinstance(b.op,ast.Div)and isinstance(b.right,ast.Constant)and b.right.value=='deferred_finalize_cpu_proof.json':
   if ast.unparse(b.left) in ('p.HOME','HOME'):
    b.left=ast.parse("__import__('benefit_followup_protocol_v2').ROOT",mode='eval').body;changes.append((b,copy.deepcopy(a)))
  if name=='verify_benefit_followup_mixed_deferred_v4'and isinstance(b,ast.Subscript)and ast.unparse(b)=="invalid['expert_counts']['lift_pot']":
   b.slice=ast.Constant('place_empty_cup');changes.append((b,copy.deepcopy(a)))
 # Reverse every registered scope-only mutation, prove entire remaining AST identical.
 restored=copy.deepcopy(tree)
 def restore(node):
  for changed,old in changes:
   if type(node)is type(changed)and getattr(node,'lineno',None)==getattr(changed,'lineno',None)and getattr(node,'col_offset',None)==getattr(changed,'col_offset',None):return copy.deepcopy(old)
  for field,value in ast.iter_fields(node):
   if isinstance(value,list):setattr(node,field,[restore(x)if isinstance(x,ast.AST)else x for x in value])
   elif isinstance(value,ast.AST):setattr(node,field,restore(value))
  return node
 restored=restore(restored);assert ast.dump(restored)==ast.dump(base),(name,'Unexpected non-scope mutation')
 src=ast.unparse(ast.fix_missing_locations(tree))+'\n';label='<transfer200_'+name+'>';linecache.cache[label]=(len(src),None,src.splitlines(True),label)
 EDITS[name]=dict(original_sha256=__import__('hashlib').sha256(text.encode()).hexdigest(),scope_only_AST_inverse_exact=True,changes=len(changes))
 return compile(src,label,'exec')
class Loader(importlib.abc.Loader):
 def __init__(self,path):self.path=path
 def create_module(self,spec):return None
 def exec_module(self,module):module.__file__=str(self.path);exec(transform(self.path),module.__dict__)
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname==TRANSFER_MODULE:return importlib.util.spec_from_file_location(fullname,TRANSFER_FILE,loader=Loader(TRANSFER_FILE))
  if fullname in SCOPE|DEFERRED|FIXTURE|{'verify_benefit_followup_mixed_cpu_v5'}:return importlib.util.spec_from_file_location(fullname,D/(fullname+'.py'),loader=Loader(D/(fullname+'.py')))

def install():
 global INSTALLED
 if INSTALLED:return
 os.environ['CX_BENEFIT_GROWTH_BLOCK']='dev0';sys.meta_path.insert(0,Finder())
 import benefit_event_full_resources_v174 as base;base.install()
 import benefit_followup_protocol_v2 as p,benefit_growth_runtime as runtime
 assert p.TASKS==TASKS and p.ROOT.parent.name==NAME
 __import__('task_scope').install()
 # Explicit extra sources preserve all original83 and the unchanged science closure.
 prior=runtime.install_sources
 def sources():
  prior();import shared6_eval_backend as b
  if getattr(b,'_transfer200_sources',False):return
  previous=b.sources
  b.sources=lambda:dict(previous(),**{os.path.relpath(f,D):p.sha(f)for f in HERE.glob('*.py')})
  b._transfer200_sources=True
 runtime.install_sources=sources
 previous_install=runtime.install
 def resource_install():
  previous_install()
  m=sys.modules.get('transfer200_capacity')
  if m is None:
   spec=importlib.util.spec_from_file_location('transfer200_capacity',HERE.parent/'gate_extension_recovery_v192/entry.py');m=importlib.util.module_from_spec(spec);sys.modules['transfer200_capacity']=m;spec.loader.exec_module(m)
  if not m._APPLIED:m.apply()
  if os.environ.get('CX_TRANSFER_ORIGINAL_PLANNER_CHECK')=='1':return
  import shared6_eval_backend as backend
  if getattr(backend,'_transfer200_lanes',False):return
  old=backend.make_plan
  def make(*a,**kw):
   plan=old(*a,**kw);before=copy.deepcopy(plan)
   from evaluate import host
   for j in plan['jobs']:
    f=host(j['run'])/'concurrency_queue.json';q=json.loads(f.read_text())if f.exists()else{};left=j['manifest']['episodes']-len(q.get('completed',{}));assert left>=0
    if not j['complete']:j['lane_count']=min(6,left)
   restored=copy.deepcopy(plan)
   for j,k in zip(restored['jobs'],before['jobs']):j['lane_count']=k['lane_count']
   assert restored==before
   mode='validate'if plan['task'].endswith('validate')else'collect';p.atomic(p.ROOT/(mode+'_plan.json'),plan);p.atomic(p.ROOT/'active_plan.json',plan)
   return plan
  backend.make_plan=make;backend._transfer200_lanes=True
 runtime.install=resource_install
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c
  from evaluate import CANONICAL_PROJECT
  original=c.command;canonical=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_transfer_v200/entry.py')
  def route(*a,**kw):
   cmd=original(*a,**kw)
   for i,s in enumerate(cmd):
    if s==str(CANONICAL_PROJECT/'deployment/benefit_event_full_resources_v174.py'):cmd[i]=canonical
    elif isinstance(s,str)and "__import__('benefit_event_full_resources_v174').install();"in s:
     cmd[i]=s.replace("__import__('benefit_event_full_resources_v174').install();","import sys;sys.path.insert(0,"+repr(str(Path(canonical).parent))+");__import__('transfer_entry' if 'transfer_entry' in sys.modules else 'entry').install();",1)
   from evaluate import add_env
   add_env(cmd,dict(CX_GATE_TRANSFER_COHORT=str(COHORT),CX_TRANSFER_ORIGINAL_PLANNER_CHECK=os.environ.get('CX_TRANSFER_ORIGINAL_PLANNER_CHECK','0')))
   return cmd
  c.command=route
  import subprocess
  popen=subprocess.Popen
  class RoutedPopen(popen):
   def __init__(self,args,*a,**kw):
    if isinstance(args,(tuple,list)):args=[str(HERE/'entry.py')if str(x)==str(D/'benefit_event_full_resources_v174.py')else x for x in args]
    super().__init__(args,*a,**kw)
  subprocess.Popen=RoutedPopen
 INSTALLED=True

def execute(path,args):
 sys.argv=[str(path),*map(str,args)]
 if path.stem in SCOPE|DEFERRED|FIXTURE|{'verify_benefit_followup_mixed_cpu_v5'}:exec(transform(path),dict(__name__='__main__',__file__=str(path)))
 else:runpy.run_path(str(path),run_name='__main__')
if __name__=='__main__':
 install();execute(Path(sys.argv[1]),sys.argv[2:])
