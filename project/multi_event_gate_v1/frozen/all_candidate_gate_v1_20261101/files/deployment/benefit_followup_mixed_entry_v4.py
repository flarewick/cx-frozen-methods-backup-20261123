"""Separate repaired-encoder namespace; never mix the preserved source79 episodes."""
import os,sys
_INSTALLED=False

def install():
 global _INSTALLED
 if _INSTALLED:return
 import benefit_followup_protocol_v2 as p
 p.HOME=p.W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4';p.ROOT=p.HOME/p.BLOCK
 if not(p.HOME/'namespace_amendment.json').exists():raise RuntimeError('Explicit expert-only ancestry amendment required')
 import benefit_followup_mixed_entry_v3 as prior
 prior.install()
 import benefit_growth_runtime as runtime
 before_sources=runtime.install_sources
 def sources():
  before_sources()
  import shared6_eval_backend as backend
  if not getattr(backend,'_constant_owner_v4_sources',False):
   previous=backend.sources
   amendment=os.path.relpath(p.HOME/'namespace_amendment.json',p.D)
   extra=('benefit_prefix_graph_lifetime_v10.py','benefit_prefix_original_reference_v11.py','benefit_followup_mixed_entry_v4.py',amendment)
   backend.sources=lambda:dict(previous(),**{name:p.sha(p.D/name)for name in extra});backend._constant_owner_v4_sources=True
 runtime.install_sources=sources
 before_install=runtime.install
 def runtime_install():
  __import__('benefit_prefix_graph_lifetime_v10').install();before_install();__import__('benefit_prefix_original_reference_v11').install()
 runtime.install=runtime_install
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c
  from evaluate import CANONICAL_PROJECT
  previous=c.command;old=str(CANONICAL_PROJECT/'deployment/benefit_followup_mixed_entry_v3.py');new=str(CANONICAL_PROJECT/'deployment/benefit_followup_mixed_entry_v4.py')
  def route(*a,**kw):
   result=previous(*a,**kw)
   for i,value in enumerate(result):
    if value==old:result[i]=new
    elif isinstance(value,str)and "__import__('benefit_followup_mixed_entry_v3').install();"in value:result[i]=value.replace("__import__('benefit_followup_mixed_entry_v3').install();","__import__('benefit_followup_mixed_entry_v4').install();",1)
   return result
  c.command=route
 _INSTALLED=True

def main():
 install();__import__('benefit_followup_mixed_entry_v3').main()
if __name__=='__main__':main()
