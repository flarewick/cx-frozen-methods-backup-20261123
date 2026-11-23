"""Explicit resource capacity amendment atop CPU-source/owner-priority211."""
import sys,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'gate_transfer_capacity_v212';sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE));sys.modules.setdefault('entry215',sys.modules[__name__]);INSTALLED=False

def install():
 global INSTALLED
 if INSTALLED:return
 import entry212;entry212.install()
 import benefit_growth_runtime as runtime
 previous=runtime.install
 def installed():previous();__import__('solver215').install()
 runtime.install=installed
 prior=runtime.install_sources
 def sources():
  prior()
  import shared6_eval_backend as backend,benefit_followup_protocol_v2 as p
  if getattr(backend,'_solver215_source',False):return
  before=backend.decorate
  def decorate(*a,**kw):
   raw=before(*a,**kw)
   raw['source_sha256'].update({str(f):p.sha(f)for f in (HERE/'entry215.py',HERE/'solver215.py',p.ROOT/'solver215_CPU.json')})
   raw['resource_solver215']=dict(original_full_solver_availability_result_cached=True,environment_directory_stats_path_finders_and_optional_module_identities_checked_every_call=True,original_TOPPRA_solver_selection_and_entire_trajectory_algorithm_unchanged=True,original14_capacity_and_all_resource_thresholds_retained=True)
   from cx_tta.evaluation_resume import identity
   raw['evaluation_identity']=identity(raw);return raw
  backend.decorate=decorate;backend._solver215_source=True
 runtime.install_sources=sources
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c,resource_entry as resource
  from evaluate import CANONICAL_PROJECT
  resource.WrappedPopen.__init__.__globals__['_startup211_entry']=HERE/'entry215.py'
  previous_command=c.command;old=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_transfer_capacity_v212/entry212.py');new=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_transfer_solver_v215/entry215.py')
  def command(*a,**kw):
   cmd=previous_command(*a,**kw)
   for i,v in enumerate(cmd):
    if v==old:cmd[i]=new
    elif isinstance(v,str)and "__import__('entry212').install();"in v:cmd[i]=v.replace("__import__('entry212').install();","__import__('entry212').install();sys.path.insert(0,"+repr(str(Path(new).parent))+");__import__('entry215').install();",1)
   return cmd
  c.command=command
 INSTALLED=True
if __name__=='__main__':__import__('entry215').install();__import__('resource_entry').entry.execute(Path(sys.argv[1]),sys.argv[2:])
