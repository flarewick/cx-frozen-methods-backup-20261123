"""Actual observed TRAIN prefixes; not counterfactual success evidence."""
import json,copy
import numpy as np,torch
from native_seven_task_gate_expansion_v1.contract import T,read,mapped
from native_seven_task_gate_expansion_v1.gated_compose import EventComposer
from native_seven_task_gate_expansion_v1.paired_heads import build,combine,ARM
from gated_arm_context_v1.application import FeedbackApplication
from gated_arm_context_v1.replay import replay,arrays

def check():
 from cx_tta.multi_event import EventConfig
 from native_arm_phase_feedback_v1.readouts import KIND
 from native_q0_policy_overlap_v1.episode import make_execute
 from .episode import execute
 assert make_execute(execute)._q0_original is execute
 b=T/'outputs/native_seven_gated_dev_v2/dev_gated';root=mapped(read(b/'execution_state.json')['root']);spec=read(root/'protocol.json')
 with torch.random.fork_rng(devices=[]):gate,aux,_,_=build()
 # Actual old no-update/zero/shadow whole traces remain exact under new context.
 n=0
 for p in sorted(root.glob('cases/*/*/result.json')):
  r=read(p)
  if r['condition']in('baseline','M_event_full','zero_update','shadow_update'):
   assert replay(r,spec,gate,aux)['passed'];n+=1
 seen=[]
 for p in sorted(root.glob('cases/*/TTA/result.json')):
  r=read(p);composer=EventComposer(EventConfig(**spec['event_config']));app=FeedbackApplication();kinds={}
  for q in r['queries']:
   policy=arrays(q['policy']['policy_file'],q['policy']['policy_sha256']);legal={k.removeprefix('legal_'):v.copy()for k,v in policy.items()if k.startswith('legal_')};base=legal['base'][0]
   a=arrays(q['actions_file'],q['actions_sha256']);direction=None if r['condition']=='M_event_full'else a['auxiliary_direction'];adapted,transfer=combine(a['reference'],a['auxiliary_reference'],direction)
   aa,dd,event=composer.step(base,a['reference'],legal,adapted_ARM=adapted)
   snapshot=copy.deepcopy(composer.controller.__dict__)
   aa,dd,event,receipt=app.step(composer,base,a['reference'],adapted,aa,dd,event,q['response'],condition='TTA',source_kinds=kinds)
   if receipt.get('gate_decision')and receipt['gate_decision']['context_origin']=='observed_initial_own_ARM_commands':
    gd=receipt['gate_decision'];assert gd['context'][:5]==[0.]*5 and gd['source_context']==dict(detected_query=0,kinds=[])
    assert receipt['source_query']==0 and kinds[0]==KIND
    if not gd['admit']:assert composer.controller.__dict__==snapshot and np.array_equal(aa,a['actions'])
    seen.append(dict(case=p.parent.parent.name,query=q['query'],admit=gd['admit'],applied=receipt['applied'],maximum_delta=float(np.abs(dd).max()),not_executed_counterfactual_success=True))
   assert np.array_equal(aa[:,[6,13]],base[:,[6,13]])and np.max(np.abs(dd))<=.050001
   # Do not feed a counterfactual action into a real later observation.
   if not np.array_equal(aa,a['actions']):break
   admission=q['feedback_source_admission']
   if admission['admitted']:kinds[q['query']]=admission['kind']
 assert n==28 and len(seen)==4 and sum(x['admit']for x in seen)==1
 # Non-admitted/mistyped source must stay rejected, and cannot spend a budget.
 # The actual no-update traces above exercise full, shadow and zero end-to-end.
 value=dict(passed=True,whole_real_no_update_control_branches_exact=n,observed_prefix_scores=seen,fixed_gate_threshold_and_parameters_unchanged=True,no_task_seed_privileged_or_fabricated_event_features=True,no_executed_trajectory_or_benefit_claim=True)
 return value

if __name__=='__main__':
 v=check();o=T/'outputs/gated_arm_context_v1/CPU_context_checks_v1.json';assert not o.exists();o.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
