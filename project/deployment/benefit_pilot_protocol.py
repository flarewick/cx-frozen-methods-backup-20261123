"""Frozen new TRAIN/VAL eligibility recipe; no policy outcomes select scenes."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent;W=R/'multi_event_gate_v1'
ROOT=W/'outputs/benefit_pilot_v1/pilot_20261025_v1'
TRAINING=W/'outputs/task_adapters/shared6_2642e508106d2049/training'
CANDIDATE=W/'outputs/training_diagnostics/shared6_vs_single_20261025/shared6_lambda0p0/best_scaled.pt'
TASKS={'lift_pot':8400000,'dump_bin_bigbin':8410000}
BRANCHES=('A0','A1','B')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def condition(task,branch):return ('M_'if branch=='B'else'P_')+'pilot_'+task+'_'+branch
def decode(value):
 for task in TASKS:
  for branch in BRANCHES:
   if value==condition(task,branch):return task,branch
 raise ValueError('Unknown pilot branch')
def prepare():
 plan=json.loads((TRAINING/'training_plan.json').read_text());known={r['seed']for r in plan['records']};inputs={str(TRAINING/'training_plan.json'):sha(TRAINING/'training_plan.json')}
 for p in (W/'outputs/task_adapter_eval').glob('*/*/eval_*/journal.json'):
  v=json.loads(p.read_text());known.update(r['seed']for r in v.get('attempts',[]));known.update(r['seed']for r in v.get('episodes',[]));inputs[str(p)]=sha(p)
 for task,start in TASKS.items():
  if known.intersection(range(start,start+64)):raise RuntimeError('Pilot seed window overlaps previous data')
 if sha(CANDIDATE)!='6e0de463a03d8bac0e160aa46cf955f8575b78ecd38d1e799392e253a7d289e7':raise RuntimeError('Frozen shared checkpoint changed')
 value=dict(schema='new_native_intervention_benefit_pilot_v1',tasks=TASKS,maximum_screen_attempts_per_task=64,eligible_per_task=2,split_by_eligible_index={'0':'TRAIN','1':'VAL'},branches=list(BRANCHES),episodes=12,candidate_sha256=sha(CANDIDATE),candidate_step=250,alpha=.5,gripper_residual=0,branch_dependency='A0 complete -> A1 complete -> B, independently for each scene',continuation='B changes only the first original event; all later actions base-only',task_config='demo_randomized',instruction_type='unseen',no_policy_outcome_scene_selection=True,new_seed_disjoint_from_previous=True,known_seed_count=len(known),known_seed_set_sha256=hashlib.sha256(json.dumps(sorted(known)).encode()).hexdigest(),separation_evidence_sha256=inputs,policy_labels_created=False)
 file=ROOT/'protocol.json'
 if file.exists():
  old=json.loads(file.read_text())
  # New pilot outputs must never alter the prior-data separation certificate.
  normalized=json.loads(json.dumps(old).replace('/data/qiufw/cx_migration/cx_new_direction_guide_TTA',str(R)))
  if normalized!=value:raise RuntimeError('Frozen pilot protocol differs')
 else:atomic(file,value)
 return value
if __name__=='__main__':print(json.dumps(prepare(),ensure_ascii=False))
