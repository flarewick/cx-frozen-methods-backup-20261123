"""Independent full-event candidate-veto validation; sealed first-event TEST untouched."""
import ast,copy,json,os,time
from pathlib import Path
_INSTALLED=False

def install():
 global _INSTALLED
 if _INSTALLED:return
 assert os.environ.get('CX_BENEFIT_GROWTH_BLOCK')=='dev0'
 import benefit_followup_mixed_entry_v4 as original;original.install()
 import benefit_followup_protocol_v2 as p
 home=p.HOME;folder=home/'candidate_veto_event_full_extension_GPU3_v174';planfile=folder/'prospective_plan.json';implfile=folder/'implementation_v174.json';plan=json.loads(planfile.read_text());impl=json.loads(implfile.read_text());assert impl['entry_sha256']==p.sha(__file__)
 modelpath=home/'fixed_ensemble_TRAIN472_v118/diagnostic_gate.json';assert p.sha(modelpath)==plan['model_sha256']and p.sha(p.CANDIDATE)==plan['adapter_sha256']
 assert all(p.sha(p.D/n)==h for n,h in impl['source_sha256'].items())
 p.ROOT=folder/'dev0';p.BLOCK='dev0';p.SPLIT='DEV';p.TASKS=plan['seed_starts'];p.START=p.TASKS['lift_pot'];p.VAL=tuple(range(20));p.REPEAT=(0,1,6,11)
 import benefit_growth_runtime as runtime
 assert not runtime.BOUND;runtime.ROOT=p.ROOT
 template=json.loads((home/'train1/protocol.json').read_text());template.update(schema='frozen_full_event_candidate_veto_validation_v134',tasks=p.TASKS,block=p.BLOCK,split=p.SPLIT,val_qualified_indices=list(p.VAL),gated_validation_episodes=40,first_event_only=False,later_events_always_rejected=False,event_full_budget=3,candidate_veto_scope='first_scored_candidate_only',experiment_sha256=p.sha(planfile),scope_entry_sha256=p.sha(__file__),implementation_sha256=p.sha(implfile),gate_model_sha256=p.sha(modelpath),success_above_event_full_not_required_for_reporting=True,original_event_full_controller_and_executor_unchanged=True,no_fitting_or_threshold_tuning=True)
 originals=json.loads((home/'learning_v6/preregistration.json').read_text())['collection_source_sha256']
 def prepare():
  assert all(p.sha(p.D/n)==h for n,h in {**originals,**impl['source_sha256']}.items());p.ROOT.mkdir(parents=True,exist_ok=True);f=p.ROOT/'protocol.json'
  if f.exists():assert json.loads(f.read_text())==template
  else:p.atomic(f,template)
  gate=p.ROOT/'gate.json'
  if gate.exists():assert gate.read_bytes()==modelpath.read_bytes()
  else:gate.write_bytes(modelpath.read_bytes())
  return copy.deepcopy(template)
 p.prepare=prepare;p.condition=lambda task,branch:('M_'if branch in('B','G')else'P_')+'fullveto174_'+task+'_'+branch
 before_bind=runtime.bind
 def bind():
  before_bind()
  import benefit_stage_runtime as stage
  if getattr(stage,'_full_veto174_plan',False):return
  previous=stage.transform
  def transform(function,mutate):
   def edit(node):
    mutate(node)
    if node.name=='make_plan':
     found=0
     for n in ast.walk(node):
      if isinstance(n,ast.keyword)and n.arg=='max_interventions':
       assert ast.unparse(n.value)=="1 if branch in ('B', 'G') else 0";n.value=ast.parse("3 if branch in ('B','G') else 0",mode='eval').body;found+=1
     assert found==1
   return previous(function,edit)
  stage.transform=transform;stage._full_veto174_plan=True
 runtime.bind=bind
 before_sources=runtime.install_sources
 def sources():
  before_sources()
  import shared6_eval_backend as b
  if getattr(b,'_full_veto174_sources',False):return
  old=b.sources;b.sources=lambda:dict(old(),**{n:p.sha(p.D/n)for n in impl['source_sha256']});b._full_veto174_sources=True
 runtime.install_sources=sources
 before_runtime=runtime.install
 def install_runtime():
  before_runtime()
  import benefit_stage_policy as stage
  from benefit_event_full_policy_v142 import FrozenFullPolicyProbeCompatible as FrozenFullPolicy
  stage.StagePolicy=FrozenFullPolicy
 runtime.install=install_runtime
 import benefit_followup_policy_first_entry_v17 as parent;parent.install()
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c
  from evaluate import CANONICAL_PROJECT,add_env
  previous=c.command;old=str(CANONICAL_PROJECT/'deployment/benefit_followup_policy_first_entry_v17.py');new=str(CANONICAL_PROJECT/'deployment/benefit_event_full_entry_v174.py')
  def route(argv,*args,**kwargs):
   result=previous(argv,*args,**kwargs)
   for i,s in enumerate(result):
    if s==old:result[i]=new
    elif isinstance(s,str)and "__import__('benefit_followup_policy_first_entry_v17').install();"in s:result[i]=s.replace("__import__('benefit_followup_policy_first_entry_v17').install();","__import__('benefit_event_full_entry_v174').install();",1)
   add_env(result,dict(CX_BENEFIT_GROWTH_BLOCK='dev0'));return result
  c.command=route
 _INSTALLED=True

def main():install();__import__('benefit_followup_mixed_entry_v3').main()
if __name__=='__main__':main()
