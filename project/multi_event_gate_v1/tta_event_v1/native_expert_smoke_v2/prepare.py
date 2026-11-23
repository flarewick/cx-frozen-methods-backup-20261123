"""Freeze four previously selected FIT cases for native execution/clock verification only."""
from native_expert_smoke_v2.paths import *
def prepare():
 planfile=T/'outputs/native_expert_replay_v1/input_plan.json';plan=read(planfile)
 authfile=T/'outputs/native_feedback_confirm_v5/direction_training_authorization.json';auth=read(authfile)
 terminal=T/'outputs/native_feedback_audit_v1/complete.json';audit=read(terminal)
 assert auth['direction_training_allowed']and auth['actual_host_return0']and audit['passed']
 assert plan['input_plan_frozen']and plan['interface_smoke_record_indices']==[40,0,41,1]
 oldfile=T/'outputs/native_feedback_confirm_v5/protocol.json';old=read(oldfile)
 planner_specfile=T/'outputs/native_expert_planner_v2/protocol.json';planner_spec=read(planner_specfile)
 roots=[f.parent for f in(T/'outputs/native_expert_planner_v2/collections').glob('*/manifest.json')if read(f)['protocol_identity']==planner_spec['identity']];assert len(roots)==1
 planner_root=roots[0];assert read(planner_root/'result.json')['complete']and read(planner_root/'interface_qualification.json')['passed']
 planner_rows={r['case_id']:r for r in read(planner_root/'GPU_frames_complete.json')['rows']}
 planner_items={r['case_id']:r for r in planner_spec['scenes']}
 exits=list((T/'outputs/native_expert_planner_v2').glob('host_run_*/exit.json'));assert len(exits)==1 and read(exits[0])['exit_code']==0
 placementfile=T/'outputs/native_feedback_confirm_v5/cpu_placement_plan.json';placement=read(placementfile)
 scenes=[]
 for i in plan['interface_smoke_record_indices']:
  r=plan['records'][i];actual=planner_rows[r['case_id']];assert actual['available']
  q0=next(v for v in actual['initializations']if not v['eval_mode']);scenes.append(dict(r,matched_initial_file=q0['file'],matched_initial_sha256=q0['sha256'],object_names=planner_items[r['case_id']]['object_names']))
 assert all(r['split']=='adapter_fit'for r in scenes)
 sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
 for f in(T/'native_expert_planner_v2').glob('*.py'):sources[str(f.relative_to(P))]=sha(f)
 for f in C.rglob('*'):
  if f.is_file()and f.suffix in('.py','.sh'):sources[str(f.relative_to(P))]=sha(f)
 for f in(planfile,authfile,terminal,oldfile,placementfile,planner_specfile,planner_root/'result.json',planner_root/'interface_qualification.json',planner_root/'GPU_frames_complete.json',exits[0],T/'outputs/native_expert_replay_v1/planner_trajectory_inventory.json',T/'native_resource_policy_v1/policy.json',T/'native_resource_policy_v1/guard.py',T/'native_terminal_guard_v1/strict_guard.py'):
  inputs[str(f)]=sha(f)
 for r in scenes:
  assert sha(r['file'])==r['sha256'];inputs[r['file']]=r['sha256']
  assert sha(r['matched_initial_file'])==r['matched_initial_sha256'];inputs[r['matched_initial_file']]=r['matched_initial_sha256']
 spec=dict(version='native_expert_smoke_v2',scenes=scenes,source_sha256=sources,input_sha256=inputs,
  runtime_source=old['runtime_source'],required_gpu_uuid=placement['gpu_uuid'],CPU_placement=str(placementfile),
  gpu_deadline_seconds=240,CPU_threads=2,one_time_warmup_per_task=True,max_ticks_per_chunk=5000,
  no_VLA_or_semantic_model_loaded=True,all_native_command_outcomes_retained=True,
  initialization_mode=False,actual_cached_planner_q0_binding_required=True,
  purpose='Four fixed FIT retimed open-loop command-reference episodes AFTER exact original expert q0 binding: real RGB/state at chunk boundaries, actual position/velocity/grip/dt and per-command committed tick ranges. Not a training cache, exact planner replay, or TTA effect test.',
  query_starts='Every32 source commands, variable tail without padding; stop on native termination or explicit5000tick chunk limit',
  no_extra_actions_or_holds=True,no_success_selection=True,native_cache_ready=False,direction_training=False,online_updates=0,TTA_utility_proven=False)
 spec['identity']=identity(spec);f=O/'protocol.json'
 if f.exists():assert read(f)==spec,'Frozen expert execution protocol changed'
 else:write(f,spec)
 verify(spec);return spec
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--gpu-uuid',required=True);args=a.parse_args();spec=prepare();root,m=dataset(spec,args.gpu_uuid)
 print('FACTOR_PLAN',json.dumps(dict(root=str(root),identity=m['identity'],remaining=0 if(root/'result.json').exists()else 4)),flush=True)
