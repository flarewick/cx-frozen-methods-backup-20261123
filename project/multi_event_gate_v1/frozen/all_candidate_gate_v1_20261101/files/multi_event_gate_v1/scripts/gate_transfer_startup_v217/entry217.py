"""Explicit resource capacity amendment atop CPU-source/owner-priority211."""
import sys,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'gate_transfer_solver_v215';sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE));sys.modules.setdefault('entry217',sys.modules[__name__]);INSTALLED=False

def install():
 global INSTALLED
 if INSTALLED:return
 import entry215;entry215.install()
 import benefit_growth_runtime as runtime
 previous=runtime.install
 def installed():previous();sys.path.insert(0,str(HERE.parent/'gate_transfer_startbarrier_v214'));__import__('barrier214').install()
 runtime.install=installed
 prior=runtime.install_sources
 def sources():
  prior()
  import shared6_eval_backend as backend,benefit_followup_protocol_v2 as p
  if getattr(backend,'_barrier217_source',False):return
  before=backend.decorate
  def decorate(*a,**kw):
   raw=before(*a,**kw)
   raw['source_sha256'].update({str(f):p.sha(f)for f in (HERE/'entry217.py',HERE.parent/'gate_transfer_startbarrier_v214/barrier214.py',p.ROOT/'barrier217_CPU.json')})
   raw['resource_barrier217']=dict(original_request_priority1_restored=True,initial_real_scene_first_action_barrier=True,original14_capacity_and_solver215_cache_retained=True,original_initializations_queries_per_scene_RNG_physics_and_all_guards_retained=True)
   from cx_tta.evaluation_resume import identity
   raw['evaluation_identity']=identity(raw);return raw
  backend.decorate=decorate;backend._barrier217_source=True
 runtime.install_sources=sources
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c,resource_entry as resource
  from evaluate import CANONICAL_PROJECT
  resource.WrappedPopen.__init__.__globals__['_startup211_entry']=HERE/'entry217.py'
  previous_command=c.command;old=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_transfer_solver_v215/entry215.py');new=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_transfer_startup_v217/entry217.py')
  def command(*a,**kw):
   cmd=previous_command(*a,**kw)
   for i,v in enumerate(cmd):
    if v==old:cmd[i]=new
    elif isinstance(v,str)and "__import__('entry215').install();"in v:cmd[i]=v.replace("__import__('entry215').install();","__import__('entry215').install();sys.path.insert(0,"+repr(str(Path(new).parent))+");__import__('entry217').install();",1)
   return cmd
  c.command=command
 INSTALLED=True
if __name__=='__main__':__import__('entry217').install();__import__('resource_entry').entry.execute(Path(sys.argv[1]),sys.argv[2:])
