"""New independent supervision cohorts; completed growth-v1 remains immutable."""
import os,json,copy
from pathlib import Path
from benefit_pilot_protocol import D,R,W,TRAINING,CANDIDATE,sha,atomic
BASE=W/'outputs/benefit_stage_v1/stage40_20261026_v1'
PRIOR=W/'outputs/benefit_growth_v1/growth_20261026_v1'
HOME=W/'outputs/benefit_followup_v2/followup_20261027_v2'
BLOCK=os.environ.get('CX_BENEFIT_GROWTH_BLOCK','train0')
BLOCKS={'train0':(8520000,'TRAIN'),'train1':(8540000,'TRAIN'),'train2':(8560000,'TRAIN'),'dev0':(8580000,'DEV'),'test0':(8600000,'TEST')}
if BLOCK not in BLOCKS:raise ValueError('Unknown prospective followup cohort')
START,SPLIT=BLOCKS[BLOCK];ROOT=HOME/BLOCK;TASKS={'lift_pot':START,'dump_bin_bigbin':START+10000}
BRANCHES=('A0','A1','B','G');VAL=()if SPLIT=='TRAIN'else tuple(range(20));REPEAT=(0,1,6,11)
EXTRA=('benefit_followup_protocol_v2.py','benefit_followup_bootstrap_v2.py','benefit_followup_gate_v2.py','benefit_gate_dynamics_v3.py','benefit_gate_latent_v4.py','benefit_gate_paired_v5.py')
def condition(task,branch):return ('M_'if branch in('B','G')else'P_')+'followup2_'+BLOCK+'_'+task+'_'+branch
def decode(value):
 for task in TASKS:
  for branch in BRANCHES:
   if value==condition(task,branch):return task,branch
 raise ValueError('Unknown followup condition')
def split(index):
 if not 0<=index<20:raise ValueError('Qualified index outside fixed20')
 return SPLIT
def selected(branch,index):return index in REPEAT if branch=='A1'else index in VAL if branch=='G'else True
def branch_rows(task,branch):
 v=json.loads((ROOT/'qualified.json').read_text());out=[];n=0
 for original in v['tasks'][task]['attempts']:
  row=copy.deepcopy(original);index=row['episode_index'];row['qualified_index']=index
  if row['eligible']and not selected(branch,index):row.update(eligible=False,reason='preregistered_branch_subset',original_expert_eligible=True)
  row['episode_index']=n;n+=int(row['eligible']);out.append(row)
 return out

def prepare():
 HOME.mkdir(parents=True,exist_ok=True);ROOT.mkdir(exist_ok=True)
 if sha(CANDIDATE)!='6e0de463a03d8bac0e160aa46cf955f8575b78ecd38d1e799392e253a7d289e7':raise RuntimeError('Frozen adapter changed')
 old=json.loads((PRIOR/'analysis_preregistration.json').read_text())['source_sha256']
 if any(sha(D/n)!=h for n,h in old.items()):raise RuntimeError('Original61 science files changed')
 file=HOME/'experiment.json'
 if not file.exists():
  known=set();evidence={};paths=[TRAINING/'training_plan.json',BASE/'qualified.json',W/'outputs/benefit_pilot_v1/pilot_20261025_v1/qualified.json'];paths+=list((W/'outputs/task_adapter_eval').glob('*/*/eval_*/journal.json'))
  def walk(v):
   if isinstance(v,dict):
    if isinstance(v.get('seed'),int):known.add(v['seed'])
    for z in v.values():walk(z)
   elif isinstance(v,list):
    for z in v:walk(z)
  for path in paths:
   v=json.loads(path.read_text());walk(v);evidence[str(path)]=sha(path)
  prior=json.loads((PRIOR/'experiment.json').read_text());prior_windows=[]
  for row in prior['blocks'].values():
   for offset in(0,10000):prior_windows.append(set(range(row['start']+offset,row['start']+offset+256)))
  windows=[]
  for name,(start,part)in BLOCKS.items():
   for offset in(0,10000):
    values=set(range(start+offset,start+offset+256));assert not values.intersection(known)and not any(values.intersection(q)for q in prior_windows+windows);windows.append(values)
  train_sources=[(BASE,'TRAIN',32),(PRIOR/'train0','TRAIN',40),(PRIOR/'train1','TRAIN',40),(PRIOR/'dev0','DEV',40)]
  atomic(file,dict(schema='candidate_gate_followup_v2',started_after_prior_TEST_closed=True,prior_TEST_excluded_from_all_fit_and_selection=True,old_VAL8_excluded=True,prior_DEV_reclassified_TRAIN_explicitly=True,base_training_sources=[dict(root=str(p),original_split=s,followup_split='TRAIN',scenes=n,audit_sha256=sha(p/'collection_audit.json'))for p,s,n in train_sources],base_TRAIN_scenes=152,new_TRAIN_block_order=['train0','train1','train2'],new_TRAIN_min_scenes=40,new_TRAIN_max_scenes=120,stop_only_after_complete_cohort_if_combined_TRAIN_rescue_and_harm_at_least=12,all_zero_and_nonzero_benefit_scenes_retained=True,no_policy_outcome_scene_selection_within_cohort=True,new_DEV_scenes=40,new_TEST_scenes=40,blocks={n:dict(start=s,split=p,scenes=40,collection_episodes=88)for n,(s,p)in BLOCKS.items()},candidate_freeze_before_TEST=True,TRAIN_cohort_CV_selects_model_and_normalization=True,DEV_checked_only_after_TRAIN_selection=True,TEST_once_after_ready_DEV_and_frozen_gate=True,target='actualG strictly above both A0 and ungated first-event B; full selected trajectories and scores exact',adapter_sha256=sha(CANDIDATE),original61_source_sha256=old,additional_runtime_source_sha256={n:sha(D/n)for n in EXTRA},prior_experiment_sha256=sha(PRIOR/'experiment.json'),seed_separation_evidence_sha256=evidence,resource_policy='unchanged strict startup-inclusive PID/device cumulative and rolling300>=33; 4096..24563MiB; CPU40..1600%;RSS128GiB;16cores;account max3 at all hours;6h;first-arrival'))
  value=json.loads(file.read_text())
  for name in ('typecheck_gpu_parity.json','prefix_graph_gpu_proof.json'):
   for certificate in(PRIOR/'card_certificates').glob('*/'+name):
    destination=HOME/'card_certificates'/certificate.parent.name/name;destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():destination.write_bytes(certificate.read_bytes())
 exp=json.loads(file.read_text());assert exp['adapter_sha256']==sha(CANDIDATE)and exp['additional_runtime_source_sha256']=={n:sha(D/n)for n in EXTRA}
 value=dict(schema='candidate_gate_followup_cohort_v2',block=BLOCK,split=SPLIT,tasks=TASKS,eligible_per_task=20,maximum_screen_attempts_per_task=256,val_qualified_indices=list(VAL),aa_qualified_indices=list(REPEAT),collection_episodes=88,gated_validation_episodes=len(VAL)*2,candidate_sha256=sha(CANDIDATE),candidate_step=250,alpha=.5,gripper_residual=0,physical_episode_index=0,instruction_generator_count=20,canonical_clutter_bounds=dict(xlim=[-.59,.59],ylim=[-.34,.34],zlim=[.741]),task_config='demo_randomized',instruction_type='unseen',ordering='A0 -> A1 if preselected -> B; G only after model frozen',first_event_only=True,later_events_always_rejected=True,no_policy_outcome_scene_selection=True,experiment_sha256=sha(file),old_TEST_and_VAL_not_fit=True)
 target=ROOT/'protocol.json'
 if target.exists():assert json.loads(target.read_text())==value,'Followup protocol changed'
 else:atomic(target,value)
 return value
