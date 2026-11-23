"""One-time local content freeze from already audited evaluations. Never starts a GPU."""
import hashlib,json,shutil
from pathlib import Path
W=Path(__file__).resolve().parents[1];R=W.parent;D=R/'deployment';H=W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4';F=W/'frozen/all_candidate_gate_v1_20261101'
C='/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA'
def read(f):return json.loads(f.read_text())
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def host(f):return Path(str(f).replace(C,str(R),1))
def main():
 if(F/'manifest.json').exists():raise SystemExit('Already frozen; run verify_frozen_all_candidate_gate.py instead of overwriting')
 expected={};references=[];events=[];allkeys=set()
 def add_expected(file,h):
  file=Path(file).resolve();assert file.is_relative_to(R)
  if file in expected:assert expected[file]==h,('Conflicting audited digest',file)
  expected[file]=h
 for i in (0,1):
  root=H/f'candidate_veto_all_events_v237_{i}'
  plan=read(root/'dev0/validate_plan.json');reg=read(root/'preregistration237.json')
  assert read(root/'final.json')['science_passed']
  for j in plan['jobs']:
   events.append(j['manifest']['event_config'])
   for f,h in j['manifest']['source_sha256'].items():
    allkeys.add(f)
    if f.startswith(C+'/'):add_expected(host(f),h)
  for f,h in reg['CPU_helper_sources'].items():add_expected(D/f,h)
  for file in (root/'dev0/validate_plan.json',root/'preregistration237.json',root/'implementation_v174.json'):
   references.append(file)
 assert all(x==events[0]for x in events)
 for f,h in expected.items():assert f.is_file()and sha(f)==h,('Live file differs from audited source',f)
 # Operational sources also retain the exact completed controller's own commitments.
 state=read(H/'all_candidate_v249_state.json');assert state['evaluation_complete']
 for f,h in state['source_sha256'].items():assert sha(Path(f))==h;add_expected(f,h)
 for f,h in read(H/'all_candidate_completion_receipt250.json')['sha256'].items():assert sha(Path(f))==h
 local={f:'execution'for f in expected}
 science_names={'benefit_event_full_policy_v142.py','benefit_event_full_veto_v132.py','benefit_fixed_ensemble_gate_v119.py','benefit_followup_gate_v2.py','benefit_gate_latent_v4.py','benefit_gate_dynamics_v3.py','benefit_raw_features_TRAIN472_v108.py','candidate_eval_checkpoint.py','candidate_eval_policy.py','candidate_eval_reference.py'}
 for f in local:
  if f.name in science_names or '/cx_tta/'in str(f)or '/task_adapters/'in str(f)or f.name=='allgate237.py':local[f]='algorithm'
 def add(f,role):
  f=Path(f);assert f.is_file(),f
  if f not in local or role in ('model','algorithm'):local[f]=role
 for f in [H/'fixed_ensemble_TRAIN472_v118/diagnostic_gate.json',W/'outputs/training_diagnostics/shared6_vs_single_20261025/shared6_lambda0p0/best_scaled.pt',W/'outputs/task_adapters/shared6_2642e508106d2049/training/adapter_seed0.pt',R/'outputs/feature_caches/structured_readout_20261003_065755_140725/projection.npz']:
  add(f,'model')
 assert sha(H/'fixed_ensemble_TRAIN472_v118/diagnostic_gate.json')=='1534934341ee285b6dfb7653d1c9e06bbd0aa0c462e3d6963a6d42405550fdfc'
 assert sha(W/'outputs/training_diagnostics/shared6_vs_single_20261025/shared6_lambda0p0/best_scaled.pt')=='6e0de463a03d8bac0e160aa46cf955f8575b78ecd38d1e799392e253a7d289e7'
 for i in (0,1):
  root=H/f'candidate_veto_all_events_v237_{i}';assert sha(root/'dev0/gate.json')==sha(H/'fixed_ensemble_TRAIN472_v118/diagnostic_gate.json')
  for f in [root/'final.json',root/'dev0/all_candidate_extended_prefix248.json',root/'dev0/all_candidate_CPU237.json']:
   if f.exists():add(f,'evidence')
 for f in [H/'all_candidate_summary250.json',H/'all_candidate_v237_final.json',H/'all_candidate_completion_receipt250.json',H/'全事件门控v118对照结果.md']:
  add(f,'evidence')
 for f in(W/'scripts/gate_all_candidate_summary_v250').glob('*.py'):add(f,'evidence')
 add(W/'scripts/gate_all_candidate_prefix_v248/check248.py','evidence')
 for f in [W/'docs/全事件候选门控方案冻结说明_v1_20261101.md',W/'scripts/verify_frozen_all_candidate_gate.py',Path(__file__)]:add(f,'documentation')
 files=[]
 for f,role in sorted(local.items(),key=lambda x:str(x[0])):
  rel=f.relative_to(R);dest=F/'files'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  value=sha(f)
  if dest.exists():assert sha(dest)==value
  else:shutil.copyfile(f,dest)
  assert sha(dest)==value;dest.chmod(0o444)
  files.append(dict(project_relative_path=str(rel),original_host_path=str(f),snapshot=str(dest.relative_to(F)),sha256=value,bytes=f.stat().st_size,role=role,previously_committed_by_audited_plan=f in expected))
 refs=[]
 for f in references:
  cohort=f.parent.parent.name if f.parent.name=='dev0'else f.parent.name
  name=cohort+'_'+f.name;dest=F/'reference'/name;dest.parent.mkdir(exist_ok=True)
  if dest.exists():assert sha(dest)==sha(f)
  else:shutil.copyfile(f,dest)
  dest.chmod(0o444);refs.append(dict(source=str(f),snapshot=str(dest.relative_to(F)),sha256=sha(dest)))
 summary=read(H/'all_candidate_summary250.json');assert summary['science_passed']and summary['new_protected_attempts_passed']and not summary['all_attempts_resources_passed']
 contract=dict(candidate='M_shared250_a050',step=250,alpha=0.5,adapter_sha256='6e0de463a03d8bac0e160aa46cf955f8575b78ecd38d1e799392e253a7d289e7',gate_sha256='1534934341ee285b6dfb7653d1c9e06bbd0aa0c462e3d6963a6d42405550fdfc',gate_schema='fixed_six_jackknife_effect_v118',scope='every original legal controller scoring opportunity including fresh pending rescore',features=['latent_event72','dynamics33','observation488'],feature_dimensions=593,raw_linear_members=12,gate_tau='mean(values)+std(values,ddof=0)',score_rule='rescue-2*harm>0; equality rejects',calibrated_success_probabilities=False,explicit_context_dimensions=8,complete_rejection_history_encoder=False,arm_raw_cap_rad=0.1,arm_applied_cap_rad=0.05,gripper_residual=0,event_config=events[0],action_shape=[32,14],sampler_batch=1,compile_mode='max-autotune',gemm_backends='ATEN',no_retraining=True,later_event_scores_uncalibrated=True)
 manifest=dict(schema='local_all_candidate_gate_freeze_v1',freeze_id='all_candidate_gate_v1_20261101',frozen_date_Asia_Shanghai='2026-11-01',contract=contract,files=files,reference_registries=refs,previously_audited_project_files=len(expected),complete_reference_source_keys=len(allkeys),external_dependencies_not_snapshotted=['original user container and /opt/cx_venv','base VLA weights and original state proofs','RoboTwin assets/runtime outside project','GPU-specific compiler/native/driver bytes with original ownership proofs'],not_standalone_runnable_or_signed_or_remote_backup=True,acceptance=dict(science_passed=True,extended_prefix_comparisons_passed=160,new_protected_attempts_passed=True,all_attempts_resources_passed=False,exploratory_40_scenes_per_task_not_independent_confirmation=True),reference_metrics=summary['metrics'])
 target=F/'manifest.json';target.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n');(F/'manifest.sha256').write_text(sha(target)+'  manifest.json\n');target.chmod(0o444);(F/'manifest.sha256').chmod(0o444)
 print(json.dumps(dict(freeze_id=manifest['freeze_id'],files=len(files),snapshot_bytes=sum(x['bytes']for x in files),reference_registries=len(refs),audited_project_files=len(expected),manifest_sha256=sha(target)),indent=2))
if __name__=='__main__':main()
