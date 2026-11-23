"""CPU source scope for a prospective fixed75 augmentation; no dataset promotion."""
from pathlib import Path
import ast
from .common import T,C,read,sha,mapped,TASKS
from native_policy_execution_v1.common import write

def main():
 for p in C.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 from native_onpolicy_collection_v24.tracker_checks import experiment
 assert experiment(C/'runtime.py'),'Actual production tracker must dispatch on original owner'
 expected=(T/'native_onpolicy_collection_v17/runtime.py').read_text().replace('native_onpolicy_collection_v17','native_onpolicy_arm_collection_v35').replace('native_onpolicy_mixed_teacher_v3','native_onpolicy_arm_teacher_v1');expected=expected.replace(" install_forward_cache(c,torch,models,base,cache,root/'forward_allocator_reclamation.jsonl')"," check_allocator=install_forward_cache(c,torch,models,base,cache,root/'forward_allocator_reclamation.jsonl')").replace('      with sam2_profile():return fn(*args,**kwargs)','      check_allocator()\n       with sam2_profile():return fn(*args,**kwargs)');expected=expected.replace("  key=prepared['seed_store'].submit(initial,instruction)","  key=prepared['seed_store'].submit(initial,instruction)\n  startup.release_q0(initial,name+':'+str(seed),key)");from native_q0_policy_overlap_v1.runtime import restore
 assert expected==restore((C/'runtime.py').read_text().replace('native_q0_handoff_atomic_v1.handoff','native_q0_task_handoff_v3.handoff'))
 from .episode import execute
 from native_arm_phase_feedback_v1.episode import execute as original_execute
 assert execute._q0_original is original_execute
 assert(C/'startup.py').read_text()==(T/'native_onpolicy_collection_v25/startup.py').read_text()
 assert(C/'cpu_admission.py').read_text()==(T/'native_onpolicy_collection_v26/cpu_admission.py').read_text()
 assert(C/'host.py').read_text()==(T/'native_onpolicy_collection_v38/host.py').read_text().replace('native_onpolicy_collection_v38','native_onpolicy_arm_collection_v35')
 from native_semantic_budget_pool_v1.transform import restore as restore_semantic
 for name,kind in [('semantic.py','dispatcher'),('semantic_worker.py','worker')]:
  actual=restore_semantic((C/name).read_text(),kind)
  expected_sem=restore_semantic((T/'native_onpolicy_collection_v38'/name).read_text(),kind).replace('native_onpolicy_collection_v38','native_onpolicy_arm_collection_v35')
  assert actual==expected_sem
 assert "v['own_actual_cold_ARM_workload_qualification_required']=True" in (C/'collection_plan.py').read_text()
 import threading,numpy as np
 from .startup import StagedLanes
 from native_onpolicy_collection_v25.semantic import content_key
 old=T/'outputs/native_onpolicy_collection_v24/fit_005';rr=mapped(read(old/'execution_state.json')['root']);meta=read(sorted((rr/'initial_prefetch').glob('*.json'))[0]);req=read(rr/'semantic/requests'/(meta['content_key']+'.json'))
 with np.load(mapped(req['frame_file']),allow_pickle=False)as z:q0=dict(rgb=tuple(z['cam'+str(i)].copy()for i in range(3)),command_state=z['command_state'].copy(),native_action_count=0)
 assert content_key(q0['rgb'],q0['command_state'],req['instruction'])==meta['content_key']
 rows=[];finished=[];gate=StagedLanes(threading.Event(),rows.append,lanes=8);threads=[threading.Thread(target=lambda n=n:finished.append((n,gate.wait(n))))for n in range(4,8)]
 for thread in threads:thread.start()
 assert not finished;assert gate.release_q0(q0,'CPU_actual_saved_q0',meta['content_key'])
 for thread in threads:thread.join(3);assert not thread.is_alive()
 assert sorted(finished)==[(n,True)for n in range(4,8)]and len(rows)==1 and rows[0]['released_lanes']==[4,5,6,7]

 old=ast.parse((T/'native_onpolicy_collection_v17/host.py').read_text());new=ast.parse((C/'host.py').read_text())
 def terminal(tree):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='terminal')
 from native_process_missing_guard_v2.terminal import terminal_ast_exact
 assert terminal_ast_exact(terminal(new),terminal(old))
 fit=[(task,i)for n in range(2)for task in TASKS for i in range(n*5,(n+1)*5)];val=[(task,i)for task in TASKS for i in range(5)];assert len(fit)==len(set(fit))==50 and len(val)==len(set(val))==25
 source=(C/'collection_plan.py').read_text();assert "7124000 if split=='fit'else 7125000"in source and "simulation_lanes=8,cases_per_batch=25"in source and "report['coverage_passed']"in source and "mechanism/'actual_host_wait_exit.json'"in source
 from .contracts import check
 check()
 teacher=T/'outputs/native_onpolicy_arm_teacher_v1/CPU_legacy40_packet_parity.json';proof=read(teacher);assert proof['passed']and proof['actual_qualified_legacy_cases']==40
 for name,digest in proof['source_sha256'].items():assert sha(mapped(name))==digest
 from native_onpolicy_calibration_data_v17.dataset import complete_data
 try:complete_data()
 except ValueError as error:assert 'Full fixed325' in str(error)
 else:raise AssertionError('Incomplete population accepted for fitting')
 # Actual failed C27 last FIT7 must remain outside all scientific datasets.
 from native_onpolicy_calibration_data_v17.reader import Reader
 failed=T/'outputs/native_onpolicy_collection_v27/fit_007';bad=mapped(read(failed/'execution_state.json')['root']);assert read(failed/'actual_host_wait_exit.json')['exit_code']==1
 try:Reader().qualified(bad)
 except (AssertionError,FileNotFoundError,ValueError):pass
 else:raise AssertionError('Failed C27 cold batch accepted')
 files=[T/'outputs/native_q0_handoff_atomic_v1/CPU_actual_lock_handoff_checks_v1.json',T/'outputs/native_q0_handoff_atomic_v1/CPU_actual_host_exit_v1.json',T/'outputs/native_onpolicy_collection_v38/CPU_collection_contract_checks_v1.json',T/'outputs/native_onpolicy_collection_v38/CPU_collection_actual_host_exit_v1.json',T/'outputs/native_server_rule4_v10/CPU_actual_C27_observer_history_checks_v1.json',T/'outputs/native_server_rule4_v10/CPU_actual_host_exit_v1.json',failed/'actual_host_wait_exit.json',T/'outputs/native_onpolicy_collection_v26/CPU_collection_contract_checks_v3.json',T/'outputs/native_onpolicy_collection_v26/CPU_collection_actual_host_exit_v3.json',T/'outputs/native_onpolicy_calibration_data_v17/CPU_reader_regression_v2.json',T/'outputs/native_onpolicy_calibration_data_v17/CPU_actual_host_exit_v2.json',T/'outputs/native_onpolicy_collection_v25/CPU_collection_contract_checks_v3.json',T/'outputs/native_onpolicy_collection_v25/CPU_collection_actual_host_exit_v3.json',teacher,T/'outputs/native_initial_arm_mechanism_v4/CPU_contract_checks_v1.json',T/'outputs/native_initial_arm_mechanism_v4/CPU_actual_host_exit_v1.json',T/'outputs/native_arm_phase_feedback_v1/CPU_integration_checks.json',T/'outputs/native_arm_phase_feedback_v1/CPU_integration_actual_host_exit.json']
 write(T/'outputs/native_onpolicy_arm_collection_v35/CPU_contract_checks_v18.json',dict(passed=True,exact_C17_except_teacher_allocator_q0_admission_and_reversible_once_q0_wait=True,original_average_limits_with_explicit_idle_zero_bound_exception=True,fixed_FIT50_VAL25_prospective_population=True,native_mechanism_coverage_and_controls_required_before_freeze=True,actual_legacy40_packet_parity_verified=True,complete325_distinct_support_gate_still_blocks_optimizer=True,no_GPU_or_optimizer=True,actual_eight_lane_q0_admission_threads_passed=True,source_sha256={str(p):sha(p)for p in C.rglob('*')if p.is_file()and p.suffix in('.py','.sh','.md')},input_sha256={str(p):sha(p)for p in files}))
 print('ARM_FIXED75_CPU_SCOPE_AND_INCOMPLETE325_FIT_GATE_PASS',flush=True)
if __name__=='__main__':main()
