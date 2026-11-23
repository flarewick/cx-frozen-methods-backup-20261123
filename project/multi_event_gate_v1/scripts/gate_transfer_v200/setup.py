"""Prospective transfer metadata only. Never copies evaluation episodes or proof claims."""
import hashlib,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[2];D=R/'deployment';H=R/'multi_event_gate_v1/outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4';OLD=H/'candidate_veto_event_full_extension_GPU3_v174'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 text=json.dumps(x,sort_keys=True,indent=2)+'\n'
 if p.exists():assert json.loads(p.read_text())==x,('Existing prospective data differs',p)
 else:p.write_text(text)
def main():
 oldplan=json.loads((OLD/'prospective_plan.json').read_text());oldimpl=json.loads((OLD/'implementation_v174.json').read_text());seen={};paths=list(H.glob('*/prospective_plan.json'))
 for path in paths:
  if path.parent.name.startswith('candidate_veto_cup_bowls_v200_'):continue
  d=json.loads(path.read_text());seen[str(path)]=dict(sha256=sha(path),seed_starts=d.get('seed_starts',{}))
 helpers=[*HERE.glob('*.py'),HERE.parent/'gate_extension_recovery_v192/entry.py',HERE.parent/'gate_extension_recovery_v192/scope_v196.py']
 for cohort in (0,1):
  tasks={'place_empty_cup':16000000+cohort*20000,'stack_bowls_two':16010000+cohort*20000}
  for task,start in tasks.items():
   for e in seen.values():assert all(not set(range(start,start+256)).intersection(range(other,other+256))for other in e['seed_starts'].values())
  v=H/('candidate_veto_cup_bowls_v200_'+str(cohort));root=v/'dev0';root.mkdir(parents=True,exist_ok=True)
  plan={k:val for k,val in oldplan.items()if k not in ('created_unix','prior_completion_sha256','extension_new_scenes_per_task','cumulative_target_per_task','seed_protocol_metadata_sha256')}
  plan.update(seed_starts=tasks,cohort=cohort,eligible_per_task=20,cumulative_target_per_task=40,two_independent_cohorts=True,new_tasks_only=True,previous_results_not_reused_as_controls=True,metadata_seed_window_checks=seen,transfer_no_model_or_threshold_edits=True,source_registration_before_rollouts=True)
  write(v/'prospective_plan.json',plan)
  impl=dict(oldimpl);impl['source_sha256']=dict(oldimpl['source_sha256'],**{os.path.relpath(p,D):sha(p)for p in helpers});impl['explicit_AST_scope_overlay']=True
  write(v/'implementation_v174.json',impl)
  write(root/'admission_recovery_v59.json',dict(original_host_sha256=sha(D/'benefit_growth_host_v3.py'),original_guards_unchanged=True,independent_transfer_scope=True))
  write(v/'cache_selection_CPU_v156.json',json.loads((OLD/'cache_selection_CPU_v156.json').read_text()))
  print(v)
if __name__=='__main__':main()
