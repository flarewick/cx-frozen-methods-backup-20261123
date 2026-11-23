"""Real legal-packet inference plus controller rejection/rearm/history checks, no GPU."""
import copy,json,sys
from pathlib import Path
import numpy as np,torch
assert not torch.cuda.is_initialized()and not list(Path('/dev').glob('nvidia*'));torch.set_num_threads(1)
import benefit_followup_protocol_v2 as p,benefit_growth_runtime as runtime
runtime.install_sources()
from allgate237 import AllCandidateVeto
from benefit_event_full_veto_v132 import FirstCandidateVeto
from cx_tta.multi_event import LocalController,EventConfig
from cx_tta.empirical_onset_gate import candidate_packet
from cx_tta.structured_action_adapter import FORWARD_KEYS
from benefit_pilot_audit import episode
root=p.HOME/'supervision_growth_canonical_v96/train2';a=json.loads((root/'collection_audit.json').read_text());plan=json.loads((root/'collect_plan.json').read_text());assert a['passed']and a['complete']
case=a['cases'][0];job=next(j for j in plan['jobs']if j['task']==case['task']and j['branch']=='B');e=episode(job,case['seed']);f=e[3]/(e[4]+'_q000.npz')
with np.load(f,allow_pickle=False)as z:packet=candidate_packet({k:z['legal__'+k].copy()for k in FORWARD_KEYS|{'base'}},z['raw_delta'].copy())
model=json.loads((p.ROOT/'gate.json').read_text());gate=AllCandidateVeto(LocalController(EventConfig()),model)
from benefit_fixed_ensemble_gate_v119 import predict
from benefit_fixed_ensemble_gate_v119 import effect
from cx_tta.multi_event import event_context
ctx=event_context(dict(kinds=['close_left'],detected_query=0),0,0,None);expected=predict(model,packet,ctx)
r=gate.step(0,dict(id=0,detected_query=0,kinds=['close_left']),packet,None,enabled=True)
assert gate.last_veto_records[0]['scores']==expected
# Actual wrapper with a controlled scorer tests semantics, not a claim of learned benefit.
import benefit_fixed_ensemble_gate_v119 as inference
old=inference.predict;seen=[]
def fake(model,packet,context):
 seen.append(np.asarray(context).copy());ok=len(seen)!=1
 return dict(both_fail=0.,rescue=float(ok),harm=float(not ok),both_success=0.)
inference.predict=fake
try:
 gate=AllCandidateVeto(LocalController(EventConfig()),model)
 for q in range(7):
  ev=dict(id=q//2,detected_query=q,kinds=['close_left'])if q%2==0 else None
  gate.step(q,ev,packet,None,enabled=True)
 assert len(seen)==4 and gate.interventions==3 and seen[1][6]==0 and seen[-1][6]==2/8
 gate.reset();assert gate.interventions==0 and gate.first_scored_query is None
 copied=copy.deepcopy(gate);assert copied.interventions==0
 # Both wrappers exactly preserve original controller when every score admits.
 inference.predict=lambda *a:dict(both_fail=0.,rescue=1.,harm=0.,both_success=0.)
 one=AllCandidateVeto(LocalController(EventConfig()),model);reference=LocalController(EventConfig())
 for q in range(8):
  ev=dict(id=q//2,detected_query=q,kinds=['close_left'])if q%2==0 else None
  assert one.step(q,ev,packet,None,enabled=True)==reference.step(q,ev,packet,lambda *a:inference.predict(),enabled=True)
finally:inference.predict=old
p.atomic(p.ROOT/'all_candidate_CPU237.json',dict(passed=True,no_GPU=True,actual_TRAIN_legal_packet_original118_prediction_exact=True,original_v118_effect=float(effect(model,packet,ctx)),first_rejection_then_later_three_admissions=True,actual_prior_admission_context_verified=True,reset_and_deepcopy_verified=True,all_admit_controller_exact=True,model_sha256=p.sha(p.ROOT/'gate.json'),no_fitting_or_physical_benefit_claim=True,helper_sha256=p.sha(__file__)))
print('237_ALL_CANDIDATE_CPU_PASSED',flush=True)
