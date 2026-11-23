"""Freeze all remaining original96 identities and existing unchanged native mechanisms."""
from native_expert_full_cache_v1.paths import *
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
 checks_file=O/'CPU_interface_checks.json';checks=read(checks_file)
 assert checks['passed']and checks['actual_prior_queries']==36 and checks['actual_prior_arrived_frames']==1007
 for filename in ('audit.py','capture.py','policy.py','runtime.py','cpu_checks.py'):
  tested=C/filename;assert checks['source_sha256'][str(tested)]==sha(tested)
 inputs[str(checks_file)]=sha(checks_file)
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
 spec=dict(version='native_expert_full_cache_v1',scenes=scenes,requested=96,reused_original_indices=[40,0,41,1],
  seed=17,event_config=old['event_config'],training=old['training'],projection=old['projection'],model_identity=old['model_identity'],
  runtime_source=frames['runtime_source'],CPU_placement=policy['CPU_placement'],required_gpu_uuid=policy['required_gpu_uuid'],
  source_sha256=sources,input_sha256=inputs,max_ticks_per_chunk=5000,gpu_deadline_seconds=7200,simulation_lanes=4,
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
 print('FACTOR_PLAN',json.dumps(dict(root=str(r),identity=m['identity'],remaining=0 if(r/'result.json').exists()else 96)),flush=True)
