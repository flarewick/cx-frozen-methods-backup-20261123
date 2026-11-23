"""Separate all-candidate scope, original optimized execution and truthful GPU0 transport."""
import os,sys,json,copy,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[2];D=R/'deployment';W=R/'multi_event_gate_v1'
sys.path.insert(0,str(D));sys.path.insert(0,str(HERE));os.environ['CX_BENEFIT_GROWTH_BLOCK']='dev0'
COHORT=int(os.environ.get('CX_GATE_TRANSFER_COHORT','0'));assert COHORT in (0,1)
BASE=HERE.parent/'gate_transfer_v200'
sys.path.insert(0,str(BASE));import entry as transfer
transfer.NAME='candidate_veto_all_events_v237_'+str(COHORT)
# The task/seed windows remain exactly those in the completed frozen transfer.
if os.environ.get('CX_ALL237_ORIGINAL_CPU')=='1':transfer.install()
else:
 sys.path.insert(0,str(HERE.parent/'gate_transfer_topp_pool_v218'));import entry218;entry218.install()
import benefit_followup_protocol_v2 as p,benefit_growth_runtime as runtime
assert p.ROOT.parent.name==transfer.NAME and not runtime.BOUND
# Mutate only the explicit prospective protocol descriptor, not an existing result.
for cell in p.prepare.__closure__ or ():
 value=cell.cell_contents
 if isinstance(value,dict) and value.get('schema')=='frozen_full_event_candidate_veto_validation_v134':
  value.update(candidate_veto_scope='all_legal_scored_candidates',later_event_scores_uncalibrated_transfer=True,comparison='Frozen118 first-candidate vs all-candidate transfer diagnostic; existing EVAL scenes, no training',ordering='reuse completed expert membership and audited controls; run independent all-event G only')
old_sources=runtime.install_sources

def sources():
 old_sources()
 if os.environ.get('CX_ALL237_ORIGINAL_CPU')=='1':return
 __import__('transport231').install()
 import shared6_eval_backend as b
 if getattr(b,'_all237_manifest',False):return
 prior=b.decorate
 def decorate(*a,**kw):
  raw=prior(*a,**kw);raw['all_candidate_v237']=dict(frozen_model_sha256=p.sha(p.ROOT/'gate.json'),same_gate_every_legal_candidate=True,rejection_does_not_disable_future_events=True,no_retraining=True,later_event_scores_not_calibrated=True,source_registry_sha256=p.sha(p.ROOT.parent/'preregistration237.json'))
  raw['source_sha256'].update({str(p.ROOT.parent/'preregistration237.json'):p.sha(p.ROOT.parent/'preregistration237.json')})
  from cx_tta.evaluation_resume import identity
  raw['evaluation_identity']=identity(raw);return raw
 b.decorate=decorate;b._all237_manifest=True
runtime.install_sources=sources
previous=runtime.install

def installed():
 previous();__import__('allgate237').install()
runtime.install=installed
sys.path.insert(0,str(HERE.parent/'gate_transfer_migration_v231'))
if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1' and os.environ.get('CX_ALL237_ORIGINAL_CPU')!='1':
 import user_container as c,resource_entry
 from evaluate import CANONICAL_PROJECT,add_env
 resource_entry.WrappedPopen.__init__.__globals__['_startup211_entry']=HERE/'entry237.py'
 before=c.command;old=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_transfer_topp_pool_v218/entry218.py');new=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_all_candidate_v237/entry237.py')
 def command(*a,**kw):
  cmd=before(*a,**kw)
  for i,v in enumerate(cmd):
   if v==old:cmd[i]=new
   elif isinstance(v,str)and "__import__('entry218').install();"in v:cmd[i]="import sys;sys.path.insert(0,"+repr(str(Path(new).parent))+");__import__('entry237');"+v
  add_env(cmd,dict(CX_GATE_TRANSFER_COHORT=str(COHORT),CX_ALL237_ORIGINAL_CPU='0'))
  return cmd
 c.command=command
if __name__=='__main__':
 if os.environ.get('CX_ALL237_ORIGINAL_CPU')=='1':transfer.execute(Path(sys.argv[1]),sys.argv[2:])
 else:__import__('resource_entry').entry.execute(Path(sys.argv[1]),sys.argv[2:])
