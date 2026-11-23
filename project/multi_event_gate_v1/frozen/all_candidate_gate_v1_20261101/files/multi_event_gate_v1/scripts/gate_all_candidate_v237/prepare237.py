"""Register immutable comparison scope and reuse only expert membership, never old G as new G."""
import json,os,hashlib,copy,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[2];D=R/'deployment';H=R/'multi_event_gate_v1/outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4'
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def write(f,v):
 f=Path(f);f.parent.mkdir(parents=True,exist_ok=True)
 if f.exists():assert read(f)==v,('Immutable registration differs',str(f))
 else:f.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def main():
 helpers={os.path.relpath(f,D):sha(f)for f in HERE.glob('*.py')}
 for cohort in (0,1):
  old=H/('candidate_veto_cup_bowls_v200_'+str(cohort));q=old/'dev0';v=H/('candidate_veto_all_events_v237_'+str(cohort));root=v/'dev0';root.mkdir(parents=True,exist_ok=True)
  assert read(old/'final.json')['passed']and read(old/'final.json')['complete']
  oldplan=read(old/'prospective_plan.json');plan=copy.deepcopy(oldplan);plan.update(candidate_veto_scope='all_legal_scored_candidates',reuse_completed_expert_membership=True,reuse_audited_controls_with_explicit_origin=True,no_old_G_import_as_new_G=True,later_event_scores_uncalibrated_transfer=True,not_a_new_independent_confirmation_panel=True)
  write(v/'prospective_plan.json',plan)
  impl=read(old/'implementation_v174.json');impl['source_sha256'].update(helpers);write(v/'implementation_v174.json',impl)
  originfiles={str(f):sha(f)for f in [old/'final.json',q/'qualified.json',q/'collect_plan.json',q/'collection_audit.json',q/'full_event_control_extension_v134.json']}
  reg=dict(schema='all_candidate_scope_transfer237',cohort=cohort,tasks=oldplan['seed_starts'],scenes_per_task=20,total_across_cohorts_per_task=40,old_scope='first_scored_candidate_only',new_scope='all_legal_scored_candidates',adapter_and_gate_frozen=True,model_sha256=oldplan['model_sha256'],adapter_sha256=oldplan['adapter_sha256'],no_training_or_threshold_selection=True,not_calibrated_multi_event_router=True,existing_EVAL_scenes_never_relabelled_TRAIN=True,controls_origin_sha256=originfiles,only_new_all_event_G_rollouts=True,GPU_index=0,GPU_UUID='GPU-fa87bc41-1919-efe1-73f9-11f38008763e',CPU_helper_sources=helpers,acceptance='complete all80 G; original action clock and event/controller score replay; exact A0 prefix; exact full B if no veto; report harms avoided, rescues lost, corrections; never infer per-event labels from endpoint')
  write(v/'preregistration237.json',reg)
  # Unchanged resource/mechanics certificates are reused with truthful source paths.
  for name in ['admission_recovery_v59.json','cache_preference_v96.json','cache_provenance_CPU_v96.json','success_shortcircuit_CPU_v206.json','balanced_task_admission_CPU_v207.json','startup211_CPU.json','capacity212_CPU.json','solver215_CPU.json','barrier217_CPU.json','pool218_CPU.json']:
   src=q/name;dst=root/name
   if not dst.exists():shutil.copyfile(src,dst)
  # Same original2fc -> actualGPU0 transport and verified GPU0 driver cache; no foreign GL.
  src=H/'candidate_veto_cup_bowls_v200_1/dev0/migration231';m=root/'migration231';m.mkdir(exist_ok=True)
  for name in ['target.json','transport_CPU.json']:
   if not(m/name).exists():shutil.copyfile(src/name,m/name)
  actual=read(src/'actual_driver.json');oldbytes=Path(actual['root']);newbytes=m/'actual_driver_bytes'/oldbytes.name
  if not newbytes.exists():shutil.copytree(oldbytes,newbytes,copy_function=os.link)
  actual['root']=str(newbytes);actual['reused_actual_GPU0_driver_origin_sha256']=sha(src/'actual_driver.json');write(m/'actual_driver.json',actual)
  print(v)
if __name__=='__main__':main()
