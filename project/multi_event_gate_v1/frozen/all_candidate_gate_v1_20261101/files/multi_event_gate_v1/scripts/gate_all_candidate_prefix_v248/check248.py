"""Stronger all-event causal prefix check vs B and first-only G, no fitting or GPU."""
import os,json,copy
from pathlib import Path
import numpy as np,torch
assert not torch.cuda.is_initialized()and not list(Path('/dev').glob('nvidia*'));torch.set_num_threads(1)
import benefit_followup_protocol_v2 as p
from benefit_pilot_audit import episode
from benefit_growth_canonical_audit import exact
from cx_tta.structured_action_adapter import FORWARD_KEYS
cohort=int(os.environ['CX_GATE_TRANSFER_COHORT']);old=p.HOME/('candidate_veto_cup_bowls_v200_'+str(cohort))/'dev0'
from benefit_supplement_alias_audit_v27 import install
install()
def read(f):return json.loads(Path(f).read_text())
def jobpath(j):
 j=dict(j);f=Path(j['run'])
 if not f.exists():j['run']=str(p.R/f.relative_to('/data/qiufw/cx_migration/cx_new_direction_guide_TTA'))
 return j
cp=read(old/'collect_plan.json');fp=read(old/'validate_plan.json');gp=read(p.ROOT/'validate_plan.json');cases=read(p.ROOT.parent/'final.json')['cases'];proofs=[]
def semantic(v):
 if isinstance(v,dict):return {k:semantic(x)for k,x in v.items()if k not in ('scores','margin')}
 if isinstance(v,list):return [semantic(x)for x in v]
 return v
for c in cases:
 task,seed=c['task'],c['seed'];g=episode(next(j for j in gp['jobs']if j['task']==task),seed)
 for label,plan,branch in [('event_full',cp,'B'),('first_gate',fp,'G')]:
  ref=episode(jobpath(next(j for j in plan['jobs']if j['task']==task and j['branch']==branch)),seed)
  n=min(len(ref[1]),len(g[1]));first=next((q for q in range(n)if semantic(ref[1][q]['multi_event'])!=semantic(g[1][q]['multi_event'])),None)
  differences=exact(ref,g,first)
  assert not differences,(task,seed,label,'prefix',first,differences)
  # At the first controller decision change the proposed candidate must itself be exact.
  if first is not None:
   f1=ref[3]/(ref[4]+'_q%03d.npz'%first);f2=g[3]/(g[4]+'_q%03d.npz'%first)
   with np.load(f1,allow_pickle=False)as a,np.load(f2,allow_pickle=False)as b:
    keys=['raw_delta',*['legal__'+k for k in FORWARD_KEYS|{'base'}]]
    assert all(np.array_equal(a[k],b[k])for k in keys),(task,seed,label,first,'candidate differs before decision')
  active1=[r['multi_event']['active']for r in ref[1]];active2=[r['multi_event']['active']for r in g[1]]
  full=False
  if active1==active2:
   assert not exact(ref,g),(task,seed,label,'same activation schedule must exactly replay');full=True
  proofs.append(dict(task=task,seed=seed,reference=label,first_controller_decision_difference=first,common_action_physics_RNG_base_and_current_candidate_exact=True,same_activation_schedule_full_replay=full))
assert len(proofs)==80
p.atomic(p.ROOT/'all_candidate_extended_prefix248.json',dict(passed=True,complete=True,scenes=40,comparisons=80,proofs=proofs,no_new_training_labels=True,helper_sha256=p.sha(__file__)))
print('248_STRONG_B_FIRST_GATE_PREFIX_PASSED',len(proofs),flush=True)
