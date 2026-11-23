"""CPU checks for native controls, original resource guard and absent-fit refusal."""
import ast
from pathlib import Path
import torch
from .common import C,T,read,sha,mapped
from native_policy_execution_v1.common import write


def main():
 assert not torch.cuda.is_initialized()
 for file in C.rglob('*.py'):ast.parse(file.read_text(),filename=str(file))
 from .episode import execute
 from native_arm_phase_feedback_v1.episode import execute as original
 assert execute._q0_original is original
 assert (C/'baseline.py').read_text()==(T/'native_onpolicy_collection_v27/baseline.py').read_text()
 from native_process_missing_guard_v2.terminal import terminal_ast_exact
 def fn(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
 assert ast.dump(fn(C/'host.py','terminal'))==ast.dump(fn(T/'native_onpolicy_collection_v27/host.py','terminal'))
 assert terminal_ast_exact(fn(C/'host.py','terminal'),fn(T/'native_onpolicy_collection_v17/host.py','terminal'))
 assert (C/'cpu_admission.py').read_text()==(T/'native_onpolicy_collection_v27/cpu_admission.py').read_text()
 from native_semantic_budget_pool_v1.transform import restore as restore_semantic
 for name,kind in [('semantic.py','dispatcher'),('semantic_worker.py','worker')]:
  assert restore_semantic((C/name).read_text(),kind)==restore_semantic((T/'native_onpolicy_collection_v32'/name).read_text(),kind).replace('native_onpolicy_collection_v32','native_calibrated_validation_v24')
 assert "value['CPU_semantic_adaptive_total8_threads']=True" in (C/'collection_plan.py').read_text()
 # The same real qualified saved policy/command replay now executes the initial-ARM episode.
 # Its fixture disables feedback, so it proves numerical full/query ordering, not ARM coverage.
 from native_q0_policy_overlap_v1 import checks as recorded
 import types
 namespace=dict(recorded.__dict__)
 namespace['original']=original
 source=Path(recorded.__file__).read_text()
 source=source.replace("out=T/'outputs/native_q0_policy_overlap_v1'","out=T/'outputs/native_calibrated_validation_v24'")
 source=source.replace("stage=T/'outputs/native_onpolicy_collection_v25/fit_005'","stage=T/'outputs/native_onpolicy_collection_v25/fit_005'")
 # Execute definitions in an isolated namespace; never mutate the frozen helper module.
 namespace['__name__']='CPU_initial_ARM_recorded_replay'
 source=source.replace('from native_feedback_application_v1.episode import execute as original','from native_arm_phase_feedback_v1.episode import execute as original')
 source=source.replace('CPU_recorded_policy_replay_v2','CPU_recorded_policy_replay_v8').replace('CPU_recorded_policy_order_checks_v2.json','CPU_recorded_policy_order_checks_v8.json')
 exec(compile(source,recorded.__file__,'exec'),namespace);namespace['main']()
 from .collection_plan import calibrated_qualification
 from native_visible_response_bridge_v1.bridge import load_model
 file,q=calibrated_qualification();model,checked=load_model(file)
 assert checked['minimum_distinct_FIT_cases']==7 and checked['actual_updates']==1000
 assert checked['original_training_host_exit_code']==1 and checked['additional_optimizer_steps_after_original1000']==0
 from native_onpolicy_recovery_v3.common import verify
 manifest=read(T/'outputs/native_onpolicy_calibration_data_v60/qualified_dataset.json');verify(manifest)
 assert manifest['actual_build_host_exit0']and manifest['identities']==325
 assert all(g['adapter_fit']['eligible_distinct_cases']>=7 and g['adapter_validation']['eligible_distinct_cases']>=1 for g in manifest['groups'].values())
 assert not torch.cuda.is_initialized()
 old_preload=(T/'native_calibrated_validation_v15/preload.py').read_text().replace('native_calibrated_validation_v15',C.name)
 undo=(C/'preload.py').read_text().replace("  signal_read,signal_write=os.pipe2(os.O_CLOEXEC)\n",'').replace("   os.close(signal_write);assert os.read(signal_read,1)==b'1';os.close(signal_read)\n",'').replace("  os.close(signal_read)\n  from native_signal_event_wait_v1.wait import attach,wait_and_record as observed_signal_wait\n  attach(child);os.write(signal_write,b'1');os.close(signal_write)\n",'').replace("code=observed_signal_wait(child,a.barrier/'normal_child_exit.json',original_wait=wait_and_record,events=root/'SIGINT_parent_signal_events.jsonl',","code=wait_and_record(child,a.barrier/'normal_child_exit.json',")
 assert undo==old_preload
 old_preflight=(T/'native_calibrated_validation_v15/preflight.py').read_text().replace('native_calibrated_validation_v15',C.name)
 undo=(C/'preflight.py').read_text().replace('  value=dict(base=base','  return dict(base=base').replace('  from native_oneway_model_transfer_v1.transfer import install\n  install(value,root)\n  return value\n','')
 assert undo==old_preflight
 undo_host=(C/'host.py').read_text().replace('from native_governor_margin_v2 import margin as guard','from native_process_missing_guard_v2 import guard').replace('  from native_governor_margin_v2.audit import audit as audit_governor_margin\n  audit_governor_margin(root,spec)\n','')
 assert undo_host.replace('for gpu in (1,6,5,4,2,3,0):','for gpu in (5,6,4,2,1,3):').replace("   from native_calibrated_eval_resume_v2.admission import certify\n   certify(root,gpu,info['uuid'])\n",'')==(T/'native_calibrated_validation_v16/host.py').read_text().replace('native_calibrated_validation_v16',C.name)
 from native_calibrated_eval_resume_v2.admission import carry,valid
 elapsed,carry_bindings=carry();assert elapsed==295.467
 fleet=dict(own_gpu_count=0,cards=[dict(index=1,uuid='fixture',available=True,pids=[],memory_mib=1,sm=0)])
 assert valid(fleet,1,'fixture')
 for key,value in [('pids',[1]),('memory_mib',65),('sm',6),('available',False)]:
  changed=dict(fleet,cards=[dict(fleet['cards'][0],**{key:value})])
  try:valid(changed,1,'fixture')
  except AssertionError:pass
  else:raise AssertionError('Occupied GPU admitted')
 from .preparation import collect
 from .semantic import content_key
 import numpy as np
 study=read(T/'outputs/native_calibrated_validation_v23/confirmation_000/study.json')
 cached=collect(study,0,study['initial_screen_state'])
 assert cached['cached_valid_cases']==8 and set(cached['task_cache'])==set(study['tasks'])
 assert not cached['primary_policy_results_imported']and not cached['old_masks_or_teacher_outputs_imported']
 for row in cached['requests']:
  with np.load(mapped(row['frame_file']),allow_pickle=False)as z:
   assert content_key([z['cam'+str(i)]for i in range(3)],z['command_state'],row['instruction'])==row['content_key']
 from .python_scheduling import install
 import sys
 before=sys.getswitchinterval();install(T/'outputs/native_calibrated_validation_v24/confirmation_000/CPU_actual_python_interval.json');assert abs(sys.getswitchinterval()-.0001)<1e-12;sys.setswitchinterval(before)
 from native_calibrated_confirm_resume_v1.carry import carry as confirmation_carry
 confirmation_seconds,confirmation_bindings=confirmation_carry();assert confirmation_seconds==282.305
 inputs=[*map(Path,confirmation_bindings),*map(Path,carry_bindings),file,file.parent/'actual_recovery_host_exit.json',file.parent/'actual_finalize_host_exit.json',T/'outputs/native_onpolicy_calibration_data_v60/qualified_dataset.json',T/'outputs/native_onpolicy_calibration_data_v60/build_host_exit.json',T/'outputs/native_audit_path_identity_v1/CPU_actual_checks_v1.json',T/'outputs/native_audit_path_identity_v1/CPU_actual_host_exit_v1.json',T/'outputs/native_calibrated_validation_v24/CPU_recorded_policy_order_checks_v8.json',T/'outputs/native_signal_event_wait_v1/checks_v2/CPU_actual_signal_wait_checks_v1.json',T/'outputs/native_signal_event_wait_v1/CPU_actual_host_exit_v1.json',T/'outputs/native_oneway_model_transfer_v1/CPU_contract_checks_v1.json']
 write(T/'outputs/native_calibrated_validation_v24/CPU_contract_checks_v8.json',dict(passed=True,
  original_ARM_episode_and_original_pure_base_preserved=True,original_resource_terminal_AST_exact=True,
  original_bounded_CPU_only_admission_preserved=True,actual_recorded_full_policy_and_commands_initial_ARM_CPU_replay=True,
  actual_full325_and_recovered_fixed1000_qualification_required=True,
  actual_pure_cached_inputs=cached['cached_valid_cases'],all_five_tasks_cached=True,actual100us_interval_checked=True,actual_VLA_or_native_physics_NOT_rerun=True,
  no_GPU_optimizer_or_native_qualification=True,
  source_sha256={str(f):sha(f)for f in C.rglob('*')if f.is_file()and f.suffix in('.py','.sh','.md')},
  input_sha256={str(f):sha(f)for f in inputs}))
 print('CALIBRATED_VALIDATION_CPU_CONTROLS_AND_ACTUAL_QUALIFIED_FIT_GATE_PASS',flush=True)


if __name__=='__main__':main()
