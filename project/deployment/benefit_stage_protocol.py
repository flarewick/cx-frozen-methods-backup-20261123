"""Preregistered new TRAIN/VAL first-event benefit collection and gate validation."""
import json,hashlib
from pathlib import Path
from benefit_pilot_protocol import D,R,W,TRAINING,CANDIDATE,sha,atomic
ROOT=W/'outputs/benefit_stage_v1/stage40_20261026_v1'
TASKS={'lift_pot':8420000,'dump_bin_bigbin':8430000}
BRANCHES=('A0','A1','B','G')
VAL=(0,5,10,15)
REPEAT=(0,1,6,11)
def condition(task,branch):return ('M_'if branch in ('B','G')else'P_')+'benefit40_'+task+'_'+branch
def decode(value):
 for task in TASKS:
  for branch in BRANCHES:
   if value==condition(task,branch):return task,branch
 raise ValueError('Unknown independent benefit40 condition')
def split(index):return 'VAL'if index in VAL else'TRAIN'
def selected(branch,index):return index in (REPEAT if branch=='A1'else VAL)if branch in ('A1','G')else True
def branch_rows(task,branch):
 import copy
 v=json.loads((ROOT/'qualified.json').read_text());out=[];n=0
 for original in v['tasks'][task]['attempts']:
  r=copy.deepcopy(original);qualified_index=r['episode_index'];r['qualified_index']=qualified_index
  if r['eligible']and not selected(branch,qualified_index):r.update(eligible=False,reason='preregistered_branch_subset',original_expert_eligible=True)
  r['episode_index']=n;n+=int(r['eligible']);out.append(r)
 return out

def prepare():
 file=ROOT/'protocol.json'
 if file.exists():
  v=json.loads(file.read_text());assert v['candidate_sha256']==sha(CANDIDATE);return v
 plan=json.loads((TRAINING/'training_plan.json').read_text());known={r['seed']for r in plan['records']};inputs={str(TRAINING/'training_plan.json'):sha(TRAINING/'training_plan.json')}
 for p in (W/'outputs/task_adapter_eval').glob('*/*/eval_*/journal.json'):
  v=json.loads(p.read_text());known.update(r['seed']for r in v.get('attempts',[]));known.update(r['seed']for r in v.get('episodes',[]));inputs[str(p)]=sha(p)
 old=W/'outputs/benefit_pilot_v1/pilot_20261025_v1/qualified.json';v=json.loads(old.read_text());inputs[str(old)]=sha(old)
 for task in v['tasks'].values():known.update(r['seed']for r in task['attempts'])
 assert sha(CANDIDATE)=='6e0de463a03d8bac0e160aa46cf955f8575b78ecd38d1e799392e253a7d289e7'
 for start in TASKS.values():assert not known.intersection(range(start,start+256))
 v=dict(schema='first_event_benefit40_v1',tasks=TASKS,eligible_per_task=20,maximum_screen_attempts_per_task=256,val_qualified_indices=list(VAL),aa_qualified_indices=list(REPEAT),train_scenes=32,val_scenes=8,collection_episodes=88,gated_validation_episodes=8,candidate_sha256=sha(CANDIDATE),candidate_step=250,alpha=.5,gripper_residual=0,physical_episode_index=0,instruction_generator_count=20,canonical_clutter_bounds=dict(xlim=[-.59,.59],ylim=[-.34,.34],zlim=[.741]),task_config='demo_randomized',instruction_type='unseen',ordering='A0 -> A1 if preselected -> B; G only after gate frozen',no_policy_outcome_scene_selection=True,first_event_only=True,later_events_always_rejected=True,known_seed_count=len(known),separation_evidence_sha256=inputs,learning=dict(method='L2 multinomial logistic CPU float64',classes=['both_fail','rescue','harm','both_success'],l2=1.,max_iterations=100,harm_cost=2.,admit_margin_strictly_greater_than=0.,train_only_normalization=True,validation_not_used_for_fitting_or_tuning=True,no_rescue_fail_closed=True,features='fixed causal summaries in benefit_stage_gate.py; no task/seed/outcome/GT inputs'),resource_policy='startup-inclusive device/PID cumulative and rolling300 >=33, early280, VRAM4096..24563, CPU40..1600, RSS128GiB, <=16 logical/card, <=3 account GPUs, <=6h',prior_pilot_separate=True)
 atomic(file,v);return v
if __name__=='__main__':print(json.dumps(prepare(),ensure_ascii=False))
