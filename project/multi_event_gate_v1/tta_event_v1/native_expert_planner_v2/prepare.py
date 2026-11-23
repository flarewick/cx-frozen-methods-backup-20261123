"""Four fixed FIT initialization pairs and original cached-planner executions, no fitting."""
from native_expert_planner_v2.paths import *
def prepare():
 inputsfile=O/'inputs.json';prepared=read(inputsfile);assert prepared['complete']and prepared['original_selected_FIT_indices']==[40,0,41,1]
 oldfile=T/'outputs/native_expert_smoke_v1/protocol.json';old=read(oldfile)
 terminal=T/'outputs/native_expert_smoke_analysis_v1/terminal_audit.json';assert read(terminal)['actual_host_exit']==0
 sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
 for name in('recording.py','stacking_recording.py','attribution_physics.py'):
  f=P/'multi_event_gate_v1/cx_tta'/name;sources[str(f.relative_to(P))]=sha(f)
 for f in C.rglob('*'):
  if f.is_file()and f.suffix in('.py','.sh'):sources[str(f.relative_to(P))]=sha(f)
 for f in(inputsfile,oldfile,terminal,O/'pre_GPU_amendment.json',T/'outputs/native_expert_planner_v1/protocol.json',T/'outputs/native_expert_smoke_analysis_v1/initialization_contract_audit.json'):inputs[str(f)]=sha(f)
 for r in prepared['records']:
  for file,digest in((r['reference_file'],r['reference_sha256']),(r['previous_eval_true_q0']['file'],r['previous_eval_true_q0']['sha256'])):assert sha(file)==digest;inputs[file]=digest
 spec=dict(version='native_expert_planner_v2',scenes=prepared['records'],source_sha256=sources,input_sha256=inputs,
  runtime_source=old['runtime_source'],required_gpu_uuid=old['required_gpu_uuid'],CPU_placement=old['CPU_placement'],gpu_deadline_seconds=240,CPU_threads=2,
  no_VLA_or_semantic_model_loaded=True,initialization_pair_modes=[True,False],
  original_expert_settings=dict(eval_mode=False,need_plan=False,save_data=True,save_freq=15),
  original_native_get_obs_and_picture_schedule_preserved=True,only_pickle_disk_sink_replaced=True,original_dense_commands_and_frame_ticks_compared=True,
  all_four_original_FIT_cases_retained=True,no_scene_replacement_or_success_selection=True,
  semantics='Original cached planner dense position/velocity control, NOT native32 policy-action execution or oracle correction on policy states',
  purpose='Measure actual scene changes from eval_mode and restore original expert physical/command clocks with real drive velocity recording; no training cache qualification',
  native_training_cache_ready=False,online_updates=0,TTA_utility_proven=False)
 spec['identity']=identity(spec);f=O/'protocol.json'
 if f.exists():assert read(f)==spec,'Frozen planner protocol changed'
 else:write(f,spec)
 verify(spec);return spec
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);a=p.parse_args();s=prepare();root,m=dataset(s,a.gpu_uuid)
 print('FACTOR_PLAN',json.dumps(dict(root=str(root),identity=m['identity'],remaining=0 if(root/'result.json').exists()else 4)),flush=True)
