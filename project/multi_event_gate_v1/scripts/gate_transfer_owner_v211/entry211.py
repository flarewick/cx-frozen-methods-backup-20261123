"""Resource-only startup optimization, layered over frozen transfer scope."""
import ast,copy,os,sys,inspect,textwrap
from pathlib import Path
HERE=Path(__file__).resolve().parent;REC=HERE.parent/'gate_transfer_v200/recovery_v201';P=REC.parent;R=HERE.parents[2]
sys.path.insert(0,str(R/'deployment'));sys.path.insert(0,str(REC));sys.path.insert(0,str(HERE))
sys.modules.setdefault('entry211',sys.modules[__name__])
_INSTALLED=False

def install():
 global _INSTALLED
 if _INSTALLED:return
 import resource_entry as resource,benefit_growth_runtime as runtime
 import benefit_followup_protocol_v2 as p
 previous=runtime.install
 def installed():previous();__import__('startup211').install()
 runtime.install=installed
 previous_sources=runtime.install_sources
 def sources():
  previous_sources()
  import shared6_eval_backend as backend
  if getattr(backend,'_startup211_sources',False):return
  before=backend.decorate
  def decorate(*a,**kw):
   raw=before(*a,**kw)
   files=[HERE/'entry211.py',HERE/'startup211.py',p.ROOT/'startup211_CPU.json']
   raw['source_sha256'].update({str(f):p.sha(f)for f in files})
   raw['resource_startup211']=dict(one_GPU_model_owner=True,maximum_real_scenes=10,per_task_claim_cap=5,formal_setup_priority4_precedes_discarded_constructor_priority5=True,Warp_same_caller_frame_without_full_stack_formatting=True,original_runtime_full_SHA_moved_to_CPU_with_fail_closed_file_stat_checks=True,model_gate_action_physics_and_all_resource_thresholds_unchanged=True)
   from cx_tta.evaluation_resume import identity
   raw['evaluation_identity']=identity(raw);return raw
  backend.decorate=decorate;backend._startup211_sources=True
 runtime.install_sources=sources
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as container,subprocess
  from evaluate import CANONICAL_PROJECT
  previous_command=container.command
  old=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_transfer_v200/recovery_v201/resource_entry.py');new=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_transfer_owner_v211/entry211.py')
  def command(*a,**kw):
   cmd=previous_command(*a,**kw)
   for i,v in enumerate(cmd):
    if v==old:cmd[i]=new
    elif isinstance(v,str)and "__import__('resource_entry');"in v:cmd[i]=v.replace("__import__('resource_entry');","__import__('resource_entry');sys.path.insert(0,"+repr(str(Path(new).parent))+");__import__('entry211').install();",1)
   return cmd
  container.command=command
  # Preserve the existing child routing; replace its one destination constant.
  fn=resource.WrappedPopen.__init__;tree=ast.parse(textwrap.dedent(inspect.getsource(fn)));before=copy.deepcopy(tree);hits=[]
  for n in ast.walk(tree):
   if isinstance(n,ast.Call)and ast.unparse(n)=="str(HERE / 'resource_entry.py')":hits.append(n);n.args=[ast.Name(id='_startup211_entry',ctx=ast.Load())]
  assert len(hits)==2
  supers=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and ast.unparse(n)=='super()'];assert len(supers)==1
  supers[0].args=[ast.Name(id='_startup211_class',ctx=ast.Load()),ast.Name(id='self',ctx=ast.Load())]
  restored=copy.deepcopy(tree)
  for n in ast.walk(restored):
   if isinstance(n,ast.Call)and ast.unparse(n)=='str(_startup211_entry)':n.args=ast.parse("str(HERE/'resource_entry.py')",mode='eval').body.args
   if isinstance(n,ast.Call)and ast.unparse(n)=='super(_startup211_class, self)':n.args=[]
  assert ast.dump(restored)==ast.dump(before)
  ns=dict(fn.__globals__,_startup211_entry=HERE/'entry211.py',_startup211_class=resource.WrappedPopen);exec(compile(ast.fix_missing_locations(tree),str(HERE/'entry211.py'),'exec'),ns);resource.WrappedPopen.__init__=ns['__init__']
 _INSTALLED=True

if __name__=='__main__':
 __import__('entry211').install();__import__('resource_entry').entry.execute(Path(sys.argv[1]),sys.argv[2:])
