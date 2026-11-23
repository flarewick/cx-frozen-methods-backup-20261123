"""Explicit source-recorded namespace binding; original files are never edited."""
import os,sys,runpy
from pathlib import Path
INSTALLED=False

def install():
 global INSTALLED
 if INSTALLED:return
 import benefit_followup_protocol_v2 as p
 if 'benefit_growth_protocol'in sys.modules and sys.modules['benefit_growth_protocol']is not p:raise RuntimeError('Followup must bind before importing growth protocol')
 sys.modules['benefit_growth_protocol']=p
 # Pure necessary-height helper remains bound to its real archived proof ROOT.
 import benefit_success_shortcircuit as height
 assert height.ROOT==p.BASE
 import benefit_growth_runtime as growth
 original_sources=growth.install_sources
 def install_sources():
  original_sources()
  import shared6_eval_backend as backend
  if not getattr(backend,'_followup_sources',False):
   before=backend.sources;backend.sources=lambda:dict(before(),**{n:p.sha(p.D/n)for n in p.EXTRA});backend._followup_sources=True
 growth.install_sources=install_sources
 original_install=growth.install
 def install_runtime():
  original_install()
  import benefit_stage_gate,benefit_followup_gate_v2
  benefit_stage_gate.predict=benefit_followup_gate_v2.predict
 growth.install=install_runtime
 # Every child CPU/GPU process receives the same explicit binding before imports.
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as container
  from evaluate import CANONICAL_PROJECT,add_env
  original_command=container.command;entry=str(CANONICAL_PROJECT/'deployment/benefit_followup_bootstrap_v2.py');deployment=str(CANONICAL_PROJECT/'deployment')
  def routed(argv,*args,**kwargs):
   values=list(argv)
   for i,value in enumerate(values):
    if str(value).endswith('/deployment/task_adapter_python.sh'):
     j=i+1
     if j<len(values)and values[j]=='-c':values[j+1]='import sys;sys.path.insert(0,'+repr(deployment)+");__import__('benefit_followup_bootstrap_v2').install();"+values[j+1]
     elif j<len(values)and str(values[j]).endswith('.py')and str(values[j])!=entry:values[j:j+1]=[entry,values[j]]
     break
   cmd=original_command(values,*args,**kwargs);add_env(cmd,dict(CX_BENEFIT_FOLLOWUP_CONTAINER='1'));return cmd
  container.command=routed
 INSTALLED=True

def main():
 install()
 if len(sys.argv)<2:raise SystemExit('Usage: bootstrap TARGET_SCRIPT [ARGS]')
 sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')
if __name__=='__main__':main()
