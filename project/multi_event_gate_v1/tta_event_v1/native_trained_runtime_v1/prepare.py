"""Freeze the complete four-old-scene five-condition developer runtime."""
from .paths import *
def prepare():
 oldfile=T/'outputs/native_expert_full_cache_v8/protocol.json';old=read(oldfile)
 initialfile=T/'outputs/native_trained_initial_v2/protocol.json';initial=read(initialfile)
 seedresult=T/'outputs/native_trained_semantics_v1/result.json';seeds=read(seedresult)
 assert seeds['complete']and seeds['requested']==4 and not seeds['CUDA_initialized']and seeds['CPU_models_unchanged']
 qualification=T/'outputs/native_direction_training_v2/runs/run_37022586a0f3428b98ae611681a08c35/qualification.json';q=read(qualification)
 assert q['frozen']and q['actual_training_host_exit0']and q['actual_updates']==1000
 assert q['training_checkpoint_sha256']=='7392d9bbc8cdb293ed64f84d5fc9984824d531ef274f9de4f08b675f38f2dc8a'
 sources=dict(old['source_sha256']);inputs={str(p):sha(p)for p in(oldfile,initialfile,seedresult,qualification,Path(q['training_checkpoint']))}
 from native_trained_episode_v1.episode import CONDITIONS
 for name in('native_trained_runtime_v1','native_trained_episode_v1','native_policy_execution_v1','native_live_response_source_v1','native_trained_response_bridge_v1','native_response_binding_v1','native_response_action_v1','native_trained_semantics_v1','native_completed_memory_v1','native_case_release_v1','native_resource_policy_v2'):
  for p in(T/name).rglob('*'):
   if p.is_file()and p.suffix in('.py','.sh','.json')and'outputs'not in p.parts:sources[str(p.relative_to(P))]=sha(p)
 for group in('source_sha256','input_sha256'):
  for path,h in read(T/'outputs/native_trained_semantics_v1/protocol.json')[group].items():assert sha(path)==h;inputs[path]=h
 for name in('native_trained_initial_v2/terminal_audit.json','native_trained_runtime_v1/CPU_actual_initial_checks.json','native_trained_episode_v1/CPU_wrong_feedback_control_checks.json','native_trained_episode_v1/CPU_scheduling_fixture/report.json','native_trained_runtime_v1/cache_write_binding_audit.json'):
  file=T/'outputs'/name;assert read(file)['passed'];inputs[str(file)]=sha(file)
 for file in(T/'outputs/native_trained_semantics_v1').glob('*'):
  if file.is_file():inputs[str(file)]=sha(file)
 # Only execution dependencies of a historical check are evidence; unrelated
 # report-only check files were fixed later and are not runtime dependencies.
 schedule=read(T/'outputs/native_trained_episode_v1/CPU_scheduling_fixture/report.json')
 for name in('episode.py','controls.py','cpu_checks.py'):
  p=T/'native_trained_episode_v1'/name;assert schedule['source_sha256'][str(p)]==sha(p)
 wrong=read(T/'outputs/native_trained_episode_v1/CPU_wrong_feedback_control_checks.json')
 for path,h in wrong['source_sha256'].items():assert sha(path)==h
 initialcheck=read(T/'outputs/native_trained_runtime_v1/CPU_actual_initial_checks.json')
 for path,h in initialcheck['source_sha256'].items():assert sha(path)==h
 placement=O/'cpu_placement_plan.json';device=read(O/'device_plan.json');assert read(placement)['gpu_uuid']==device['gpu_uuid']
 for p in(placement,O/'device_plan.json',Path(old['training'])/'adapter_seed0.pt',Path(old['projection']),T/'native_terminal_guard_v1/strict_guard.py'):
  inputs[str(p)]=sha(p)
 for row in seeds['rows']:
  for path in (row['frame_file'],row.get('mask_file')):
   if path is not None:inputs[path]=sha(path)
 spec=dict(version='native_trained_runtime_v1',scope='four_existing_developer_cases_only',scenes=initial['scenes'],conditions=list(CONDITIONS),requested_developer_scenes=4,requested_native_episodes=20,simulation_lanes=4,
  source_sha256=sources,input_sha256=inputs,required_gpu_uuid=device['gpu_uuid'],CPU_placement=str(placement),
  training=old['training'],projection=old['projection'],runtime_source=old['runtime_source'],model_identity=old['model_identity'],event_config=old['event_config'],
  trained_qualification=str(qualification),checkpoint_sha256=q['training_checkpoint_sha256'],initialization_eval_mode=True,per_branch_warmup_reset=True,
  semantic_preparation='actual evaluation-mode q0; exact3RGB+command+language at query0/tick0; no old expert masks',
  online_inputs='current actual RGB/language/own commands/once-base and legal prior history only; no teacher/task/seed/GT/future',
  feedback='source-local query-only SAM2 prefix then origin/following actual arrivals; actual32/95percent+.25s and original25..125 windows; once-use3 packets',
  wrong_feedback='fixed cyclic right-shift1 of same-packet eligible own command summaries; receiver context/target/startpoints/issuing hidden unchanged; no-op cases retained',
  native_budget='original task eval_success/step_lim; ExecutionClock100000 only safety censor, not extra task actions',
  exact_controls=['M_event_full vs zero_update full actions/observations/traces/noise/physics/RNG','M_event_full vs shadow_update same','first actual changed action query common-prefix/current once-base and noise'],
  max_account_GPUs=3,max_logical_CPUs=16,max_RSS_GiB=128,GPU_memory_MiB=[4096,24563],min_cumulative_and_rolling300_utilization=34,gpu_deadline_seconds=7200,
  supervisor='existing user systemd Restart=no bounded; one CUDA child; actual waitpid and GPU release proof',
  frozen_training_steps=1000,new_training_steps=0,new_independent_scenes=0,TTA_utility_proven=False)
 assert {s['seed']for s in spec['scenes']}=={7100000,7100001}and len(spec['scenes'])==4
 spec['identity']=identity(spec);dest=O/'protocol.json'
 if dest.exists():assert read(dest)==spec,'Frozen native runtime changed'
 else:write(dest,spec)
 verify(spec);return spec
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);a=p.parse_args();s=prepare();root,m=dataset(s,a.gpu_uuid)
 print('FACTOR_PLAN',__import__('json').dumps(dict(root=str(root),identity=m['identity'],remaining=0 if(root/'result.json').exists()else 20)),flush=True)
