import ast,copy,threading,textwrap,uuid
from pathlib import Path
from types import SimpleNamespace
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped,verify
from .carry import install,audit
from .scope import transformed,restore_runtime,PLAN

def main():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 old=T/'native_onpolicy_arm_collection_v34';new=T/'native_onpolicy_arm_collection_v35'
 for p in old.rglob('*'):
  if p.is_file()and p.suffix in('.py','.md','.sh'):
   q=new/p.relative_to(old);assert q.read_text()==transformed(p.name,p.read_text()),str(q)
   if q.suffix=='.py':ast.parse(q.read_text())
 assert restore_runtime((new/'runtime.py').read_text())==(old/'runtime.py').read_text().replace(old.name,new.name)
 proof=T/'outputs/native_fixed_FIT_inputs_v1/verified_old_FIT0_prefix_v1.json';value=read(proof);verify(value)
 tasks=['beat_block_hammer','turn_switch','adjust_bottle','grab_roller','open_laptop']
 folder=T/'outputs/native_carried_next_FIT_v1/CPU_fixture'/uuid.uuid4().hex;future=folder/'next_preparation';future.mkdir(parents=True)
 host_proof=Path('/data/qiufw/cx_migration/cx_new_direction_guide_TTA')/proof.relative_to(T.parents[1])
 spec=dict(dataset_split='adapter_validation',future_preparation_population=dict(split='fit',seed_start=7124000,first_index=0),future_preparation_each_task=1,carried_FIT0_input_proof=str(host_proof))
 fresh_calls=[]
 def next_case(name):
  fresh_calls.append(name);assert name=='open_laptop'
  item=dict(task=name,seed=7124000,episode_index=0,case_id=name+':7124000',CPU_fixture_only=True)
  write(future/'accepted'/(item['case_id']+'.json'),item);return item
 schedule=SimpleNamespace(next=next_case,root=future,state={t:dict(next_seed=7124000,accepted=0,rows=[])for t in tasks})
 install(schedule,spec);items=[schedule.next(t)for t in tasks];assert fresh_calls==['open_laptop']
 assert all(item['episode_index']==0 for item in items)
 assert len(list((future/'carried_inputs').glob('*.json')))==4
 try:schedule.next('turn_switch')
 except AssertionError:pass
 else:raise AssertionError('Duplicate carried input accepted')
 write(future/'preparation_complete.json',dict(complete=True,expert_valid_inputs=5,planned_maximum=5,first_index=0,last_exclusive=1,CPU_fixture_only=True))
 summary=audit(folder,spec);assert summary['passed']and summary['original_negative_validity_rows_preserved']
 # Exercise the actual new Reader block and its atomic/hash requirements.
 reader_file=T/'native_onpolicy_calibration_data_v28/reader.py';tree=ast.parse(reader_file.read_text())
 method=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='qualified')
 block=next(n for n in ast.walk(method)if isinstance(n,ast.If)and ast.unparse(n.test)=="spec.get('carried_next_FIT0_scope')")
 code=compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(block)],type_ignores=[])),str(reader_file),'exec')
 gate_root=folder/'reader_gate_fixture';gate_root.mkdir();gate_file=gate_root/'CPU_carried_FIT0_inputs_audit.json'
 bound_sources=[T/'native_carried_next_FIT_v1/scope.py',T/'native_carried_next_FIT_v1/carry.py']
 base_spec=dict(carried_next_FIT0_scope=True,carried_FIT0_input_proof=str(host_proof),source_sha256={str(p):sha(p)for p in bound_sources})
 def gate(candidate,value,bind=True):
  gate_file.write_text(__import__('json').dumps(value));namespace=dict(T=T,Path=Path,spec=candidate,root=gate_root,read=read,sha=sha,mapped=mapped,verify=verify,runtime=(new/'runtime.py').read_text(),qual=dict(input_sha256={str(gate_file):sha(gate_file)}if bind else{}),self=SimpleNamespace(inputs={}))
  exec(code,namespace);assert namespace['runtime']==(old/'runtime.py').read_text().replace(old.name,new.name)
 gate(base_spec,summary);negatives=[]
 for name in ['wrong_carried_count','wrong_required_count','wrong_fresh_task','missing_negative_rows','failed_masks_allowed','missing_source','unbound_atomic']:
  candidate=copy.deepcopy(base_spec);v=copy.deepcopy(summary)
  if name=='wrong_carried_count':v['carried_pure_inputs']=3
  if name=='wrong_required_count':v['all_required_inputs']=4
  if name=='wrong_fresh_task':v['fresh_required_task']='turn_switch'
  if name=='missing_negative_rows':v['original_negative_validity_rows_preserved']=False
  if name=='failed_masks_allowed':v['no_failed_policy_masks_or_labels_imported']=False
  if name=='missing_source':candidate['source_sha256'].pop(str(bound_sources[0]))
  try:gate(candidate,v,bind=name!='unbound_atomic')
  except AssertionError:negatives.append(name)
  else:raise AssertionError(name)
 untouched=SimpleNamespace(next=next_case);assert install(untouched,{})is None and untouched.next is next_case
 bad=dict(spec,future_preparation_each_task=2)
 try:install(schedule,bad)
 except AssertionError:pass
 else:raise AssertionError('Extra fixed work accepted')
 # Production plan's final assignments must activate exactly VAL1/FIT0-1/FIT1-0.
 v=dict(spec,input_sha256={},source_sha256={},cumulative_budget_amendment={})
 exec(textwrap.dedent(PLAN),dict(T=T,read=read,sha=sha,verify=verify,split='val',number=0,v=v))
 assert v['future_preparation_each_task']==1 and abs(v['inherited_stage_GPU_seconds']-13747.181)<1e-6
 tree=ast.parse((new/'collection_plan.py').read_text());assignments=[n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(ast.unparse(t)=="v['future_preparation_each_task']"for t in n.targets)]
 assert ast.unparse(assignments[-1].value)=="1 if split == 'val' or number == 0 else 0"
 from native_onpolicy_arm_collection_v35.dispatch_checks import run
 routing=run();assert routing['passed']and routing['cases']==25
 from native_onpolicy_calibration_data_v28.reader import Reader
 root=mapped(read(T/'outputs/native_onpolicy_collection_v38/fit_007/execution_state.json')['root']);_,cases,report=Reader().qualified(root);assert len(cases)==25 and report['passed']
 failed=mapped(read(T/'outputs/native_onpolicy_arm_collection_v34/val_000/execution_state.json')['root'])
 try:Reader().qualified(failed)
 except (AssertionError,FileNotFoundError,ValueError):pass
 else:raise AssertionError('Failed34 accepted')
 sources=[p for d in[new,Path(__file__).parent,T/'native_onpolicy_calibration_data_v28']for p in d.rglob('*')if p.is_file()and p.suffix in('.py','.md','.sh')]
 baseline=T/'outputs/native_onpolicy_arm_collection_v34/CPU_contract_checks_v2.json';verify(read(baseline));assert read(baseline)['passed']
 result=dict(passed=True,original34_runtime_episode_math_model_queries_guards_preserved=True,
  four_real_pure_inputs_hashes_and_three_negative_switch_decisions_preserved=True,
  CPU_fixture_fresh_screen_called_only_for_missing_laptop=True,duplicate_and_extra_work_refused=True,
  host_container_input_path_mapping_verified=True,production_reader_gate_negatives=negatives,
  final_effective_future_counts_VAL1_FIT0_1_FIT1_0=True,actual25_case_RNG_routing=routing,
  old_qualified_C38_25_accepted_failed34_refused=True,original325_support_old100_unchanged=True,
  fixtures_NOT_native_or_scientific_qualification=True,source_sha256={str(p):sha(p)for p in sources},
  input_sha256={str(p):sha(p)for p in[proof,baseline]})
 write(T/'outputs/native_onpolicy_arm_collection_v35/CPU_contract_checks_v2.json',result)
 write(T/'outputs/native_onpolicy_calibration_data_v28/CPU_reader_regression_v2.json',result)
 print('REQUIRED_ONLY_NEXT_FIT_INPUT_REUSE_AND_ORIGINAL25_RNG_CPU_PASS',flush=True)
if __name__=='__main__':main()
