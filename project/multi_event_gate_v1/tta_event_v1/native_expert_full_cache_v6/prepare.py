"""Freeze all remaining original96 identities and existing unchanged native mechanisms."""
from native_expert_full_cache_v6.paths import *
def prepare():
 f=T/'outputs/native_expert_full_inputs_v1/manifest.json';inputs_plan=read(f)
 oldfile=T/'outputs/native_feedback_confirm_v5/protocol.json';old=read(oldfile)
 policyfile=T/'outputs/native_expert_policy_v1/protocol.json';policy=read(policyfile)
 framefile=T/'outputs/native_expert_frames_v1/protocol.json';frames=read(framefile)
 packetfile=T/'outputs/native_expert_packets_v1/manifest.json';packets=read(packetfile)
 assert inputs_plan['complete']and packets['pilot_cache_interface_passed']and len(packets['packets'])==12
 remaining=inputs_plan['remaining_original_indices'];assert len(remaining)==len(set(remaining))==96
 assert set(remaining).isdisjoint({0,1,40,41})and inputs_plan['counts']=={'stack_bowls_two/adapter_fit':40,'stack_blocks_two/adapter_fit':40,'stack_bowls_two/adapter_validation':10,'stack_blocks_two/adapter_validation':10}
 scenes=[inputs_plan['records'][i]for i in remaining];assert all(r['new_GPU_collection_required']for r in scenes)
 sources={**old['source_sha256'],**policy['source_sha256'],**frames['source_sha256']};inputs={}
 checks_file=T/'outputs/native_expert_full_cache_v1/CPU_interface_checks.json';checks=read(checks_file)
 assert checks['passed']and checks['actual_prior_queries']==36 and checks['actual_prior_arrived_frames']==1007
 original=T/'native_expert_full_cache_v1'
 for filename in ('audit.py','capture.py','policy.py','cpu_checks.py'):
  tested=original/filename;assert checks['source_sha256'][str(tested)]==sha(tested)
  assert (C/filename).read_text().replace('native_expert_full_cache_v6','native_expert_full_cache_v1')==tested.read_text()
 inputs[str(checks_file)]=sha(checks_file)
 failed_file=T/'outputs/native_expert_full_failure_v5/report.json';failed=read(failed_file)
 assert failed['passed']and failed['actual_host_exit_code']==1 and not failed['resource_qualification_passed']
 assert failed['terminal_cases']==34 and failed['usable_native_cases']==33 and failed['remaining_whole_original_cases']==62 and failed['checks']==77082
 previous_file=T/'outputs/native_expert_full_cache_v5/protocol.json';previous=read(previous_file)
 from native_expert_full_cache_v5.paths import verify as verify_previous
 verify_previous(previous);sources.update(previous['source_sha256']);assert previous['scenes']==scenes
 recovered=[]
 for record in failed['rows']:
  copy=dict(record);assert copy['case_id']==scenes[copy['index']]['case_id']and sha(copy['capture_file'])==copy['capture_sha256']
  for path,h in copy['source_files_sha256'].items():assert sha(path)==h;inputs[path]=h
  recovered.append(copy)
 assert len(recovered)==34 and sum(r['available']for r in recovered)==33
 for p in(failed_file,previous_file):inputs[str(p)]=sha(p)
 staged_checks_file=O/'startup_checks.json';staged_checks=read(staged_checks_file)
 assert staged_checks['passed']and staged_checks['waiting_lanes_blocked_until_real_release']and staged_checks['cancelled_waiters_terminate']
 for p,h in staged_checks['source_sha256'].items():assert sha(p)==h
 inputs[str(staged_checks_file)]=sha(staged_checks_file)
 CPU_getters_file=T/'outputs/native_telemetry_batch_v1/CPU_native_getter_checks.json';CPU_getters=read(CPU_getters_file)
 assert CPU_getters['passed']and CPU_getters['checks']==24576 and CPU_getters['every_field_dtype_shape_value_byte_exact']and not CPU_getters['approved_for_collector']
 bridge_file=T/'outputs/native_telemetry_bridge_v1/CPU_contract_checks.json';bridge=read(bridge_file);assert bridge['passed']and bridge['value_dtype_mismatch_and_original_failure_poison_case']
 buildfile=T/'outputs/native_telemetry_batch_v1/build/build.json';build=read(buildfile);assert sha(build['binary'])==build['binary_sha256']
 for proof in(CPU_getters,bridge):
  for p,h in proof['source_sha256'].items():assert sha(p)==h
 for p in(CPU_getters_file,bridge_file,buildfile,Path(build['binary'])):inputs[str(p)]=sha(p)
 for directory in(T/'native_telemetry_batch_v1',T/'native_telemetry_bridge_v1'):
  for p in directory.glob('*'):
   if p.suffix in('.py','.cpp'):sources[str(p.relative_to(P))]=sha(p)
 devicefile=O/'device_plan.json';device=read(devicefile);placementfile=O/'cpu_placement_plan.json';placement=read(placementfile)
 assert device['gpu_uuid']==placement['gpu_uuid']and device['gpu_name']=='NVIDIA GeForce RTX 4090'
 certificate_file=O/('gpu-doctor-'+str(device['gpu_index'])+'.json');certificate=read(certificate_file);assert certificate['uuid']==device['gpu_uuid']
 for p in(devicefile,placementfile,certificate_file,O/'doctor_source_binding.json'):inputs[str(p)]=sha(p)
 for scene in scenes:
  for path,h in ((scene['file'],scene['sha256']),(scene['first_saved_identity_file'],scene['first_saved_identity_sha256'])):
   assert sha(path)==h;inputs[path]=h
 for p in (f,oldfile,policyfile,framefile,packetfile,T/'outputs/native_feedback_confirm_v5/direction_training_authorization.json',
           Path(old['projection']),Path(old['training'])/'adapter_seed0.pt',Path(policy['CPU_placement']),
           T/'native_resource_policy_v1/policy.json',T/'native_resource_policy_v1/guard.py',T/'native_terminal_guard_v1/strict_guard.py',
           T/'outputs/native_expert_feedback_v1/terminal_audit.json'):
  inputs[str(p)]=sha(p)
 auth=read(T/'outputs/native_feedback_confirm_v5/direction_training_authorization.json');assert auth['direction_training_allowed']and auth['actual_host_return0']
 for p in C.rglob('*'):
  if p.is_file()and p.suffix in('.py','.sh'):sources[str(p.relative_to(P))]=sha(p)
 for p in [T/'native_expert_smoke_v2'/n for n in('observe.py','setup.py')]+[T/'native_expert_planner_v2/runtime.py']:
  sources[str(p.relative_to(P))]=sha(p)
 spec=dict(version='native_expert_full_cache_v6',scenes=scenes,requested=96,reused_original_indices=[40,0,41,1],
  recovered_cases=recovered,remaining_new_indices=failed['remaining_original96_indices'],new_GPU_cases=62,source_attempt_normal_exit_NOT_claimed=True,
  supervisor='existing user systemd service, Restart=no; guarded7200s bounded workload',
  throughput_change='same staged2then6; CPU-validated readonly C++ telemetry with same-world paired comparisons(first128/every256); no control/clock changes',
  commanded_readout_bridge='first128 per case plus every256, exact dtype/shape/value, abort mismatch before row acceptance',
  source_GPU_UUID=previous['required_gpu_uuid'],new_GPU_UUID=device['gpu_uuid'],recovered_and_new_hardware_provenance_separate=True,
  initial_simulation_lanes=2,remaining_lane_release='actual_once_policy_query_complete or retained initial-case terminal fallback',
  recovery_prepared_before_CUDA=True,previous_six_lane_resource_failure_retained=True,
  predecessor_resource_failure_retained=True,recovered_terminal_cases=34,recovered_native_available=33,recovered_initialization_failures=1,
  seed=17,event_config=old['event_config'],training=old['training'],projection=old['projection'],model_identity=old['model_identity'],
  runtime_source=frames['runtime_source'],CPU_placement=str(placementfile),required_gpu_uuid=device['gpu_uuid'],
  source_sha256=sources,input_sha256=inputs,max_ticks_per_chunk=5000,gpu_deadline_seconds=7200,simulation_lanes=6,
  original_split_unchanged=True,max_sources_per_episode=3,actual_expert_commands_define_phase=True,
  single_base_per_actual_preexecution_query=True,separate_policy_RNG=True,seed_reset_per_case=True,
  sampled_base_NOT_executed=True,expert_teacher_labels_offline_only=True,no_extra_actions_or_holds=True,
  tail_and_unsupported_and_failures_retained=True,full_original_expert_cache_ready=False,training_steps=0,online_updates=0,TTA_utility_proven=False)
 spec['identity']=identity(spec);out=O/'protocol.json'
 if out.exists():assert read(out)==spec,'Frozen remaining96 protocol changed'
 else:write(out,spec)
 verify(spec);return spec
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);a=p.parse_args();s=prepare();r,m=dataset(s,a.gpu_uuid)
 from native_expert_full_cache_v6.recovery import materialize
 materialize(s,r)
 print('FACTOR_PLAN',json.dumps(dict(root=str(r),identity=m['identity'],remaining=0 if(r/'result.json').exists()else 62)),flush=True)
