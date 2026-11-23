"""Explicit resource capacity amendment atop CPU-source/owner-priority211."""
import sys,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'gate_transfer_startup_v217';sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE));sys.modules.setdefault('entry218',sys.modules[__name__]);INSTALLED=False

def install():
 global INSTALLED
 if INSTALLED:return
 import entry217;entry217.install()
 import benefit_growth_runtime as runtime
 previous=runtime.install
 def installed():previous();__import__('client218').install()
 runtime.install=installed
 prior=runtime.install_sources
 def sources():
  prior()
  import shared6_eval_backend as backend,benefit_followup_protocol_v2 as p
  if getattr(backend,'_pool218_source',False):return
  before=backend.decorate
  def decorate(*a,**kw):
   raw=before(*a,**kw)
   raw['source_sha256'].update({str(f):p.sha(f)for f in (HERE/'entry218.py',HERE/'client218.py',HERE/'protocol218.py',HERE/'worker218.py',p.ROOT/'pool218_CPU.json')})
   raw['resource_pool218']=dict(original_mplib_TOPP_in_4_no_GPU_CPU_workers=True,only_joint_path_limits_step_verbose_sent=True,all_trajectory_arrays_durations_and_original_errors_retained=True,transport_failure_aborts_not_original_fallback=True,one_GPU_PID_shared_frozen_model_and_all_original_guards=True)
   from cx_tta.evaluation_resume import identity
   raw['evaluation_identity']=identity(raw);return raw
  backend.decorate=decorate;backend._pool218_source=True
 runtime.install_sources=sources
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c,resource_entry as resource
  from evaluate import CANONICAL_PROJECT
  resource.WrappedPopen.__init__.__globals__['_startup211_entry']=HERE/'entry218.py'
  previous_command=c.command;old=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_transfer_startup_v217/entry217.py');new=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_transfer_topp_pool_v218/entry218.py')
  def command(*a,**kw):
   cmd=previous_command(*a,**kw)
   for i,v in enumerate(cmd):
    if v==old:cmd[i]=new
    elif isinstance(v,str)and "__import__('entry217').install();"in v:cmd[i]=v.replace("__import__('entry217').install();","__import__('entry217').install();sys.path.insert(0,"+repr(str(Path(new).parent))+");__import__('entry218').install();",1)
   if os.environ.get('CX_TOPP218_CONFIG'):
    from evaluate import add_env
    add_env(cmd,dict(CX_TOPP218_CONFIG=os.environ['CX_TOPP218_CONFIG']))
   return cmd
  c.command=command
 INSTALLED=True
if __name__=='__main__':__import__('entry218').install();__import__('resource_entry').entry.execute(Path(sys.argv[1]),sys.argv[2:])
