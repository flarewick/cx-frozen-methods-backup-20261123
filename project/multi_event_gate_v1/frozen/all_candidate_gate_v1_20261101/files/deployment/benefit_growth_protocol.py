"""New fixed cohorts for candidate utility learning; archived benefit40 stays immutable."""
import os,json,copy,hashlib
from pathlib import Path
from benefit_pilot_protocol import D,R,W,TRAINING,CANDIDATE,sha,atomic
BASE=W/'outputs/benefit_stage_v1/stage40_20261026_v1'
HOME=W/'outputs/benefit_growth_v1/growth_20261026_v1'
BLOCK=os.environ.get('CX_BENEFIT_GROWTH_BLOCK','train0')
BLOCKS={'train0':(8440000,'TRAIN'),'train1':(8460000,'TRAIN'),'dev0':(8480000,'DEV'),'test0':(8500000,'TEST')}
if BLOCK not in BLOCKS:raise ValueError('Unknown preregistered cohort')
START,SPLIT=BLOCKS[BLOCK];ROOT=HOME/BLOCK
TASKS={'lift_pot':START,'dump_bin_bigbin':START+10000}
BRANCHES=('A0','A1','B','G');VAL=() if SPLIT=='TRAIN' else tuple(range(20));REPEAT=(0,1,6,11)
def condition(task,branch):return ('M_'if branch in ('B','G')else'P_')+'growth_'+BLOCK+'_'+task+'_'+branch
def decode(value):
 for task in TASKS:
  for branch in BRANCHES:
   if value==condition(task,branch):return task,branch
 raise ValueError('Unknown independent growth condition')
def split(index):
 if not 0<=index<20:raise ValueError('Qualified index out of cohort')
 return SPLIT
def selected(branch,index):return index in REPEAT if branch=='A1' else index in VAL if branch=='G' else True
def branch_rows(task,branch):
 v=json.loads((ROOT/'qualified.json').read_text());out=[];n=0
 for original in v['tasks'][task]['attempts']:
  row=copy.deepcopy(original);idx=row['episode_index'];row['qualified_index']=idx
  if row['eligible']and not selected(branch,idx):row.update(eligible=False,reason='preregistered_branch_subset',original_expert_eligible=True)
  row['episode_index']=n;n+=int(row['eligible']);out.append(row)
 return out
def prepare():
 if sha(CANDIDATE)!='6e0de463a03d8bac0e160aa46cf955f8575b78ecd38d1e799392e253a7d289e7':raise RuntimeError('Frozen adapter changed')
 global_file=HOME/'experiment.json'
 if not global_file.exists():
  known=set();evidence={}
  paths=[TRAINING/'training_plan.json',BASE/'qualified.json',W/'outputs/benefit_pilot_v1/pilot_20261025_v1/qualified.json']
  paths+=list((W/'outputs/task_adapter_eval').glob('*/*/eval_*/journal.json'))
  for path in paths:
   value=json.loads(path.read_text());evidence[str(path)]=sha(path)
   def walk(v):
    if isinstance(v,dict):
     if isinstance(v.get('seed'),int):known.add(v['seed'])
     for a in v.values():walk(a)
    elif isinstance(v,list):
     for a in v:walk(a)
   walk(value)
  windows=[]
  for name,(start,part)in BLOCKS.items():
   for off in (0,10000):
    seeds=set(range(start+off,start+off+256));assert not known.intersection(seeds)
    assert all(not seeds.intersection(old)for old in windows);windows.append(seeds)
  atomic(global_file,dict(schema='candidate_gate_growth_v1',blocks={n:dict(start=s,split=p,scenes=40,collection_episodes=88)for n,(s,p)in BLOCKS.items()},old32_TRAIN_may_be_used=True,old8_VAL_remains_held_out=True,new_TRAIN_scenes=80,new_DEV_scenes=40,new_TEST_scenes=40,test_only_after_model_freeze=True,adapter_sha256=sha(CANDIDATE),old_audit_sha256=sha(BASE/'collection_audit.json'),known_seed_count=len(known),separation_evidence_sha256=evidence,no_policy_outcome_scene_selection=True,selection='TRAIN fits and normalization only; DEV selects fixed model and threshold; TEST once after frozen candidate',target='actual gated success count strictly above both matched baseline A0 and ungated first-event B; otherwise goal remains unachieved',no_significance_or_generalization_claim_without_evidence=True,resource_policy='original startup-inclusive PID/device cumulative+rolling300 >=33; 4096..24563MiB; CPU40..1600; RSS128GiB;16cores;max3account cards;6h;first arrival'))
 experiment=json.loads(global_file.read_text());assert experiment['adapter_sha256']==sha(CANDIDATE)
 value=dict(schema='candidate_gate_growth_cohort_v1',block=BLOCK,split=SPLIT,tasks=TASKS,eligible_per_task=20,maximum_screen_attempts_per_task=256,val_qualified_indices=list(VAL),aa_qualified_indices=list(REPEAT),collection_episodes=88,gated_validation_episodes=len(VAL)*2,candidate_sha256=sha(CANDIDATE),candidate_step=250,alpha=.5,gripper_residual=0,physical_episode_index=0,instruction_generator_count=20,canonical_clutter_bounds=dict(xlim=[-.59,.59],ylim=[-.34,.34],zlim=[.741]),task_config='demo_randomized',instruction_type='unseen',ordering='A0 -> A1 if preselected -> B; G after model frozen',first_event_only=True,later_events_always_rejected=True,no_policy_outcome_scene_selection=True,experiment_sha256=sha(global_file),old_pilot_and_VAL_not_fit=True)
 file=ROOT/'protocol.json'
 if file.exists():assert json.loads(file.read_text())==value,'Cohort protocol frozen'
 else:atomic(file,value)
 return value
if __name__=='__main__':print(json.dumps(prepare()))
