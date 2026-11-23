"""Versioned mixed-work entry; bind every child before importing frozen runtime."""
import os,sys,runpy
_INSTALLED=False

def install():
 global _INSTALLED
 if _INSTALLED:return
 import benefit_followup_bootstrap_v2 as prior
 prior.install()
 import benefit_growth_runtime as runtime
 from benefit_followup_protocol_v2 import D,sha
 previous_sources=runtime.install_sources
 extra=('benefit_followup_stream_v3.py','benefit_followup_mixed_core_v3.py','benefit_followup_mixed_manager_v3.py','benefit_followup_mixed_entry_v3.py','collection_pool.py','collection_native_step.py','collection_step.cpp','collection_native_control.py','collection_control.cpp','native_physics.py','benefit_growth_parallel_screen_v4.py','benefit_followup_screen_v5.py')
 def sources():
  previous_sources()
  import shared6_eval_backend as b
  if not getattr(b,'_followup_mixed_sources',False):
   before=b.sources;b.sources=lambda:dict(before(),**{n:sha(D/n)for n in extra});b._followup_mixed_sources=True
 runtime.install_sources=sources
 previous_install=runtime.install
 def install_runtime():
  runtime.bind()
  import benefit_followup_mixed_core_v3 as core
  core.before_runtime();previous_install();core.after_runtime()
 runtime.install=install_runtime
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c
  from evaluate import CANONICAL_PROJECT
  previous=c.command;old=str(CANONICAL_PROJECT/'deployment/benefit_followup_bootstrap_v2.py');new=str(CANONICAL_PROJECT/'deployment/benefit_followup_mixed_entry_v3.py')
  def route(*args,**kwargs):
   cmd=previous(*args,**kwargs)
   for i,value in enumerate(cmd):
    if value==old:cmd[i]=new
    elif isinstance(value,str)and "__import__('benefit_followup_bootstrap_v2').install();"in value:
     cmd[i]=value.replace("__import__('benefit_followup_bootstrap_v2').install();","__import__('benefit_followup_mixed_entry_v3').install();",1)
   return cmd
  c.command=route
 _INSTALLED=True

def main():
 install()
 if len(sys.argv)<2:raise SystemExit('Usage: mixed_entry TARGET_SCRIPT [ARGS]')
 sys.argv=sys.argv[1:]
 if sys.argv[0].endswith('/benefit_growth_deferred_finalize.py'):
  # Calling a second __main__ copy would bypass the mixed module wrapper.
  # Assemble the same runtime before dispatching the imported original function.
  import argparse
  import benefit_growth_runtime as runtime
  runtime.bind();__import__('benefit_memory_batches').install()
  import benefit_growth_deferred_finalize as deferred
  deferred.install();runtime.install()
  parser=argparse.ArgumentParser();parser.add_argument('--intent',required=True);args=parser.parse_args(sys.argv[1:])
  deferred.finalize(args.intent);return
 runpy.run_path(sys.argv[0],run_name='__main__')
if __name__=='__main__':main()
