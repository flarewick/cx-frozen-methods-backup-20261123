"""Register one frozen-candidate full-event validation; no GPU or old TEST reads."""
import ast,json,time
from pathlib import Path
from benefit_pilot_protocol import D,W,sha,atomic
H=W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4';V=H/'candidate_veto_event_full_extension_GPU3_v174'
def main():
 proof=H/'event_full_veto_CPU_v132/result.json';a=json.loads(proof.read_text());assert a['passed']and a['actual_recorded_queries']==396 and a['full_controller_budget_unchanged3']and a['helper_sha256']==sha(D/'benefit_event_full_veto_v132.py')
 model=H/'fixed_ensemble_TRAIN472_v118/diagnostic_gate.json';api=H/'fixed_ensemble_TRAIN472_v118/production_packet_CPU_proof_v120.json';assert json.loads(api.read_text())['passed'];m=json.loads(model.read_text());assert m['TRAIN_scenes']==472 and m['training_source_sha256']==sha(D/'benefit_fixed_ensemble_TRAIN472_v118.py')
 original=json.loads((H/'learning_v6/preregistration.json').read_text())['collection_source_sha256'];assert len(original)==83 and all(sha(D/n)==v for n,v in original.items())
 # Read only prior protocol metadata, never old TEST episodes/outcomes.
 starts={'lift_pot':15040000,'dump_bin_bigbin':15050000};newwindows=[set(range(v,v+256))for v in starts.values()];checked={}
 paths=set()
 for pattern in ['protocol.json','*/protocol.json','*/*/protocol.json','*/*/*/protocol.json']:
  paths.update(H.glob(pattern))
 for path in sorted(paths):
  if any(z.startswith(('execution','control_','finite_bootstrap_control'))for z in path.relative_to(H).parts[:-1]):continue
  d=json.loads(path.read_text());tasks=d.get('tasks',{})
  if not isinstance(tasks,dict):continue
  for v in tasks.values():
   if isinstance(v,int):assert all(not w.intersection(range(v,v+256))for w in newwindows),'Reserved protocol seed overlap'
  checked[str(path)]=sha(path)
 names=['benefit_event_full_entry_v174.py','benefit_event_full_resources_v174.py','benefit_event_full_policy_v142.py','benefit_event_full_veto_v132.py','benefit_fixed_ensemble_gate_v119.py','benefit_raw_features_TRAIN472_v108.py','benefit_fixed_cache_bind_GPU3_v156.py','prepare_benefit_event_full_extension_GPU3_v174.py','verify_benefit_event_full_scope_CPU_v174.py','benefit_event_full_audit_v134.py']
 names+=['verify_benefit_full_source_CPU_v177.py','verify_benefit_full_deferred_CPU_v179.py','verify_benefit_event_full_productive_CPU_v178.py','verify_benefit_event_full_factory_CPU_v149.py','verify_benefit_event_full_audit_CPU_v135.py','benefit_event_full_collect_v174.py','benefit_event_full_G_collect_v174.py']
 names+=['benefit_event_full_scope_audit_v175.py','benefit_full_resource_audit_v164.py']
 for n in names:ast.parse((D/n).read_text())
 V.mkdir(exist_ok=True);assert not(V/'prospective_plan.json').exists()
 plan=dict(schema='frozen_first_candidate_veto_on_unchanged_event_full_v134',created_unix=time.time(),seed_starts=starts,eligible_per_task=20,max_screen_attempts=256,model_sha256=sha(model),adapter_sha256=sha(W/'outputs/training_diagnostics/shared6_vs_single_20261025/shared6_lambda0p0/best_scaled.pt'),CPU_controller_proof_sha256=sha(proof),API472_proof_sha256=sha(api),source_model_selected_before_new_data=True,old_first_event_DEV_not_reinterpreted_as_passing=True,old_first_event_TEST_remains_sealed=True,new_scope_has_full_event_control_B_and_actual_veto_G=True,first_candidate_only_due_to_supervision_scope=True,later_candidates_original_always_admit=True,event_config_budget=3,alpha=.5,controls88_then_G40=128,report='harmful corrections avoided, no-terminal-benefit candidates filtered, useful corrections retained, intervention counts, paired success; no success>event_full requirement',no_fit_or_threshold_tuning=True,seed_protocol_metadata_sha256=checked,GPU3_same2fc_sampler=True,prior_completion_sha256=sha(H/'candidate_veto_event_full_GPU3_v154/completion_receipt_v172.json'),extension_new_scenes_per_task=20,cumulative_target_per_task=40,old_raw_results_not_merged=True,goal_achieved=False)
 atomic(V/'prospective_plan.json',plan);atomic(V/'implementation_v174.json',dict(entry_sha256=sha(D/'benefit_event_full_entry_v174.py'),source_sha256={n:sha(D/n)for n in names},original83_bytes_unchanged=True,original_event_full_admission_controller_executor_unchanged=True,explicit_gate_only_wrapper=True));print('FULL134_PREPARED_NO_GPU',json.dumps(starts),flush=True)
if __name__=='__main__':main()
