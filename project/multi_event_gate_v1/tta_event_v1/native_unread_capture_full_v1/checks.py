"""Real saved native controls plus missing/differing-feedback refusal; CPU only."""
from pathlib import Path
import ast,copy,uuid,types
import numpy as np
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped,verify
from .scope import restore_runtime,INSTALL,HOST_AUDIT

def main():
 import torch
 torch.set_num_threads(1)
 old=T/'native_onpolicy_arm_collection_v26';new=T/'native_onpolicy_arm_collection_v29'
 out=T/'outputs/native_onpolicy_arm_collection_v29';out.mkdir(parents=True,exist_ok=True)
 predecessor=T/'outputs/native_onpolicy_arm_collection_v26/CPU_contract_checks_v2.json'
 pv=read(predecessor);verify(pv);assert pv['passed']
 for name,digest in pv['source_sha256'].items():assert sha(mapped(name))==digest
 expected=lambda name:(old/name).read_text().replace(old.name,new.name)
 exceptions={'AGENTS.md','operational_checks.py','runtime.py','preload.py','host.py','closed_audit.py','collection_plan.py'}
 for p in old.rglob('*'):
  if not p.is_file()or p.suffix not in('.py','.sh','.md'):continue
  q=new/p.relative_to(old);ast.parse(q.read_text())if q.suffix=='.py'else None
  if str(p.relative_to(old))not in exceptions:assert q.read_text()==expected(p.relative_to(old)),str(q)
 assert restore_runtime((new/'runtime.py').read_text())==expected('runtime.py')
 assert (new/'preload.py').read_text().replace(INSTALL,'')==expected('preload.py')
 host=(new/'host.py').read_text().replace(HOST_AUDIT,'').replace(",root/'CPU_unread_capture_whole_reference_audit.json'",'')
 assert host==expected('host.py'),'Original GPU/account/CPU/exit guards changed'
 closed=(new/'closed_audit.py').read_text();start=closed.index(" assert spec['unread_capture_whole_reference_each_case']");end=closed.index(' for p in controls:',start)
 source=expected('closed_audit.py');original=source[source.index(' cold_controls='):source.index(' for p in controls:')]
 assert closed[:start]+original+closed[end:]==source
 plan=(new/'collection_plan.py').read_text();start=plan.index('def VAL_budget_authorization():');end=plan.index('\ndef freeze_stage():',start)
 original=expected('collection_plan.py');a=original.index('def VAL_budget_authorization():');b=original.index('\ndef freeze_stage():',a)
 plan=plan[:start]+original[a:b]+plan[end:]
 start=plan.index("  parity_file=T/");end=plan.index("  v['future_preparation_each_task']=0",start);plan=plan[:start]+plan[end:]
 plan=plan.replace("  from native_unread_capture_full_v1.budget import require_planned_full_batch\n  require_planned_full_batch(v)\n",'').replace('all_failed_seconds_retained=12409.593','all_failed_seconds_retained=11862.916')
 plan=plan.replace('native_onpolicy_calibration_data_v25.reader','native_onpolicy_calibration_data_v24.reader').replace('outputs/native_onpolicy_arm_collection_v29/CPU_contract_checks_v3.json','outputs/native_onpolicy_arm_collection_v29/CPU_contract_checks_v2.json').replace('outputs/native_onpolicy_arm_collection_v29/CPU_actual_host_exit_v3.json','outputs/native_onpolicy_arm_collection_v29/CPU_actual_host_exit_v2.json')
 assert plan==original,'Unreviewed population/budget/source/model change'
 parity_file=T/'outputs/native_onpolicy_full_ops_v2/render_parity_v1/software_parity_report.json'
 parity=read(parity_file);verify(parity)
 assert parity['software_parity_passed']and parity['own_GPU_released']and not parity['scientific_qualification']
 assert abs(parity['cumulative_VAL_seconds']-12409.593)<1e-6
 from .budget import authorization,require_planned_full_batch,USED
 _,budget=authorization(T);assert budget['approved_total_seconds']==14400
 spec=dict(dataset_split='adapter_validation',stage_total_GPU_budget_seconds=14400,inherited_stage_GPU_seconds=USED)
 try:require_planned_full_batch(spec)
 except AssertionError:pass
 else:raise AssertionError('Insufficient current finite budget accepted')
 require_planned_full_batch(dict(spec,stage_total_GPU_budget_seconds=18000))
 # This numeric test creates no authorization file, study, CUDA permit or process.
 reader_folder=T/'native_onpolicy_calibration_data_v25'
 from native_onpolicy_calibration_data_v25.reader import Reader
 original_stage=T/'outputs/native_onpolicy_collection_v38/fit_007'
 original_root=mapped(read(original_stage/'execution_state.json')['root'])
 checked,cases,recovered=Reader().qualified(original_root);assert len(cases)==25 and recovered['passed']
 probe=mapped(parity['root'])
 try:Reader().qualified(probe)
 except (AssertionError,FileNotFoundError,ValueError):pass
 else:raise AssertionError('Short software probe accepted as a scientific batch')
 # Exercise the actual new Reader source scope, including all prior exact unwind.
 reader_text=(reader_folder/'reader.py').read_text();tree=ast.parse(reader_text)
 method=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Reader').body
 method=next(n for n in method if isinstance(n,ast.FunctionDef)and n.name=='qualified')
 gate=next(n for n in method.body if isinstance(n,ast.If)and ast.unparse(n.test)=="spec.get('initial_ARM_path_feedback')")
 # The whole-pair audit is deliberately absent: the actual new reader MUST refuse.
 fixture=out/'CPU_fixture'/uuid.uuid4().hex;fixture.mkdir(parents=True)
 source_spec=dict(initial_ARM_path_feedback=True,unread_capture_whole_reference_each_case=True,
  every_original_actual_get_obs_update_random_draw_preserved=True,
  source_sha256={str(new/'runtime.py'):sha(new/'runtime.py'),**{str(p):sha(p)for p in [T/'native_unread_capture_full_v1/scope.py',T/'native_unread_capture_full_v1/audit.py',T/'native_discard_capture_v1/defer.py']}})
 code=compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(gate)],type_ignores=[])),str(reader_folder/'reader.py'),'exec')
 try:exec(code,dict(T=T,Path=Path,mapped=mapped,sha=sha,spec=source_spec,root=fixture,read=read,verify=verify))
 except FileNotFoundError as e:assert 'CPU_unread_capture_whole_reference_audit.json'in str(e)
 else:raise AssertionError('Missing whole25 reference proof accepted')
 # Real original whole pair from FAILED ARM17 is a software-only regression input.
 # It never becomes an accepted TRAIN scene or a current scientific success.
 previous=T/'outputs/native_onpolicy_arm_collection_v17/val_000'
 previous_root=mapped(read(previous/'execution_state.json')['root'])
 old_pair=next((previous_root/'cases').glob('*/q0_handoff_whole_reference_audit.json')).parent
 oldspec=read(previous_root/'protocol.json');verify(oldspec)
 def pair_folder(name):
  folder=fixture/name;folder.mkdir()
  for branch in ['M_event_full','handoff_reference']:
   target=folder/branch;target.mkdir()
   for file in(old_pair/branch).iterdir():
    (target/file.name).symlink_to(file,target_is_directory=file.is_dir())
  return folder
 from .audit import compare
 positive=pair_folder('original_saved_pair');compare(positive,oldspec)
 assert read(positive/'unread_capture_pair_audit.json')['all_arrived_feedback_RGB_clocks_commands_byte_exact']
 negative=pair_folder('changed_arrived_feedback');changed=negative/'M_event_full/execution000_arrivals.npz'
 with np.load(changed,allow_pickle=False)as z:values={name:z[name].copy()for name in z.files}
 changed.unlink();assert values['head_rgb'].size>0
 values['head_rgb'].flat[0]^=np.uint8(1)
 with changed.open('xb')as f:np.savez_compressed(f,**values)
 receipt=negative/'M_event_full/execution000.json';receipt_value=read(receipt);receipt.unlink();receipt_value['dense_sha256']=sha(changed);write(receipt,receipt_value)
 try:compare(negative,oldspec)
 except AssertionError as error:assert 'Arrived feedback RGB/clock/state mismatch'in str(error)
 else:raise AssertionError('Mid-chunk changed feedback accepted')
 assert not torch.cuda.is_initialized()
 inputs=[predecessor,predecessor.with_name('CPU_actual_host_exit_v2.json'),parity_file,original_root/'atomic_full_recovery_qualification.json',previous_root/'protocol.json',old_pair/'q0_handoff_whole_reference_audit.json']
 reader_out=T/'outputs/native_onpolicy_calibration_data_v25';reader_out.mkdir(parents=True,exist_ok=True)
 reader_proof=reader_out/'CPU_reader_regression_v3.json'
 write(reader_proof,dict(passed=True,actual_qualified_C38_25_retained=True,software_probe28_refused=True,
  missing_whole25_audit_refused=True,all325_distinct8FIT_nonemptyVAL_and_old100_unchanged=True,
  no_GPU_model_optimizer_or_fixture_scientific_qualification=True,
  source_sha256={str(p):sha(p)for p in reader_folder.iterdir()if p.is_file()},input_sha256={str(p):sha(p)for p in inputs}))
 sources=[p for directory in [new,T/'native_unread_capture_full_v1']for p in directory.rglob('*')if p.is_file()and p.suffix in('.py','.sh','.md')]
 write(out/'CPU_contract_checks_v3.json',dict(passed=True,original26_guards_math_episodes_and_population_unchanged=True,
  declared_render_omission_requires25_whole_original_pairs=True,real_saved_original_pair_arrivals_exact=True,
  single_changed_arrived_RGB_pixel_refused=True,failed_old_pair_is_software_input_only=True,
  insufficient_current14400_budget_refused=True,proposed18000_numeric_test_is_NOT_authorization=True,
  original_qualified_data_retained=True,software_probe28_NOT_qualified=True,no_GPU_or_optimizer=True,
  source_sha256={str(p):sha(p)for p in sources},input_sha256={str(p):sha(p)for p in [*inputs,reader_proof]}))
 print('WHOLE25_PAIR_AND_ARRIVED_RGB_CPU_CHECKS_PASS_GPU_BUDGET_REVIEW_STILL_REQUIRED',flush=True)

if __name__=='__main__':main()
