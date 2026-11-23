"""Prepare an isolated mixed-prefix continuation; never launch here."""
from pathlib import Path
from native_policy_execution_v1.common import T
M={'native_onpolicy_arm_collection_v73':'native_onpolicy_arm_collection_v74',
   'native_prepared_FIT_collection_v28':'native_prepared_FIT_collection_v29',
   'native_onpolicy_calibration_data_v56':'native_onpolicy_calibration_data_v57'}
NOTE='\nProspective74 continues qualified73 FIT0 without rerun. FIT1 uses real three-task cached prefix and original fresh preparation for all remaining cases; no omitted cases or numeric/teacher changes. Original failed73 FIT1 was CPU-only. FIT0 cannot launch in74. Original budgets,325/support/old100/fixed1000 remain. Freeze after CPU checks.\n'
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text
def transform_extra(name,text):
 if name=='host.py':text=text.replace('from native_postexit_compat_v1.tail import qualify','from native_postexit_compat_v2.tail import qualify')
 if name=='known_preparation.py':
  old="'native_onpolicy_arm_collection_v17','native_onpolicy_arm_collection_v74'"
  assert text.count(old)==1;text=text.replace(old,"'native_onpolicy_arm_collection_v17','native_onpolicy_arm_collection_v73','native_onpolicy_arm_collection_v74'")
 if name=='collection_plan.py':
  anchor='def freeze_stage():\n';assert text.count(anchor)==1;text=text.replace(anchor,anchor+" assert COHORT in('fit_001','val_000'),'Reuse qualified73 FIT0; never recollect or reset FIT0'\n")
  old="  if split=='fit':assert cached['cached_valid_cases']>=5,'Required genuine next-FIT preparation not complete; no cold failed-workload retry'\n"
  assert text.count(old)==1;text=text.replace(old,"  if split=='fit':\n   from native_mixed_FIT_preparation_v1.gate import require\n   require(v,cached,initial)\n")
  text=text.replace("T/'native_postexit_compat_v1'","T/'native_postexit_compat_v2'")
  anchor="  v['identity']=identity(v);verify(v);write(O/'study.json',v);return v\n"
  assert text.count(anchor)==1
  extra="  for file in[T/'outputs/native_mixed_FIT_preparation_v1/CPU_contract_checks_v3.json',T/'outputs/native_mixed_FIT_preparation_v1/actual_CPU_exit_v3.json']:\n   x=read(file);verify(x)\n   if 'exit_code'in x:assert x['exit_code']==0 and x['actual_Popen_wait_return']\n   else:assert x['passed']\n   v['input_sha256'][str(file)]=sha(file)\n  v['source_sha256'].update({str(p):sha(p)for p in(T/'native_mixed_FIT_preparation_v1').glob('*')if p.is_file()and p.suffix in('.py','.md')})\n"
  text=text.replace(anchor,extra+anchor)
 if name=='prefix_stage.py':
  old=" folder=T/'outputs/native_onpolicy_arm_collection_v74'/(split+'_'+str(number).zfill(3));"
  assert text.count(old)==1;text=text.replace(old," folder=T/'outputs'/('native_onpolicy_arm_collection_v73'if split=='fit'and number==0 else'native_onpolicy_arm_collection_v74')/(split+'_'+str(number).zfill(3));")
 if name=='AGENTS.md':text+=NOTE
 return text.replace('outputs/native_onpolicy_arm_collection_v74/CPU_contract_checks_v2.json','outputs/native_onpolicy_arm_collection_v74/CPU_contract_checks_v3.json').replace('outputs/native_onpolicy_arm_collection_v74/CPU_actual_host_exit_v2.json','outputs/native_onpolicy_arm_collection_v74/CPU_actual_host_exit_v3.json')
def main():
 for old,new in M.items():
  source,dest=T/old,T/new;assert not dest.exists()
  for p in source.rglob('*'):
   if p.is_file()and p.suffix in('.py','.sh','.md'):
    target=dest/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
    text=rename(p.read_text())
    if old=='native_onpolicy_arm_collection_v73':text=transform_extra(p.name,text)
    target.write_text(text);target.chmod(p.stat().st_mode&0o777)
 scope=T/'native_prepared_FIT_collection_v29/scope.py'
 scope.write_text("from native_prepared_FIT_collection_v28.scope import transform as previous_transform,restore_runtime\nfrom native_mixed_FIT_preparation_v1.prepare import rename,transform_extra\ndef transform(name,text):return transform_extra(name,rename(previous_transform(name,text)))\n")
 checks=T/'native_prepared_FIT_collection_v29/checks.py';text=checks.read_text().replace("assert cache['cached_valid_cases']==14","assert cache['cached_valid_cases']==25").replace('actual_original_known_preparation_fourteen_pure_inputs_loaded=True','actual_original_known_preparation_qualified25_pure_inputs_loaded=True')
 text=text.replace("folders=[new,Path(__file__).parent,", "folders=[T/'native_mixed_FIT_preparation_v1',T/'native_postexit_compat_v2',new,Path(__file__).parent,")
 checks.write_text(text)
 dest=T/'native_postexit_compat_v2';assert not dest.exists();dest.mkdir()
 for name in['tail.py','AGENTS.md']:
  text=(T/'native_postexit_compat_v1'/name).read_text().replace('native_postexit_compat_v1/tail.py','native_postexit_compat_v2/tail.py').replace("spec['version'] == 'native_onpolicy_arm_collection_v73'","spec['version'] == 'native_onpolicy_arm_collection_v74'")
  (dest/name).write_text(text)
 reader=T/'native_onpolicy_calibration_data_v57/reader.py';text=reader.read_text();old=' def qualified(self,root):\n';assert text.count(old)==1
 text=text.replace(old," def qualified(self,root):\n  if read(Path(root)/'protocol.json').get('version')=='native_onpolicy_arm_collection_v73':\n   from native_onpolicy_calibration_data_v56.reader import Reader as Previous\n   previous=Previous();result=previous.qualified(root);self.inputs.update(previous.inputs);return result\n  return self._qualified_current(root)\n def _qualified_current(self,root):\n")
 reader.write_text(text)
 source,dest=T/'native_onpolicy_calibration_fit_v3',T/'native_onpolicy_calibration_fit_v4';assert not dest.exists()
 for p in source.rglob('*'):
  if p.is_file()and p.suffix in('.py','.sh','.md'):
   q=dest/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True);text=p.read_text().replace('native_onpolicy_calibration_fit_v3','native_onpolicy_calibration_fit_v4').replace('native_onpolicy_calibration_data_v56','native_onpolicy_calibration_data_v57')
   if p.name=='host.py':
    text=text.replace(" sys.path.insert(0,str(P/'deployment'));", " assert not(T/'outputs/native_onpolicy_calibration_fit_v3/runs').exists()and not list((T/'outputs/native_onpolicy_calibration_fit_v3').glob('host_run_*')),'Prior v3 fitting budget must not be reset'\n sys.path.insert(0,str(P/'deployment'));")
   if p.name=='train.py':text=text.replace("'native_onpolicy_calibration_fit_v2','native_onpolicy_calibration_fit_v4'","'native_onpolicy_calibration_fit_v2','native_onpolicy_calibration_fit_v3','native_onpolicy_calibration_fit_v4'")
   q.write_text(text);q.chmod(p.stat().st_mode&0o777)
 source,dest=T/'native_collection73_supervision_v1',T/'native_collection74_supervision_v1';assert not dest.exists()
 for p in source.glob('*'):
  if p.suffix not in('.py','.md'):continue
  text=p.read_text().replace('native_collection73_supervision_v1','native_collection74_supervision_v1').replace('native_onpolicy_arm_collection_v73','native_onpolicy_arm_collection_v74').replace('native_onpolicy_calibration_data_v56','native_onpolicy_calibration_data_v57').replace('native_onpolicy_calibration_fit_v3','native_onpolicy_calibration_fit_v4')
  if p.name=='run.py':
   text=text.replace("fit_000/study.json","fit_001/study.json").replace("assert spec['stage_total_GPU_budget_seconds'] == 14400 and spec['inherited_stage_GPU_seconds'] == 9679.983","assert spec['stage_total_GPU_budget_seconds'] == 7200 and spec['inherited_stage_GPU_seconds'] == 0 and spec['stage']=='fit_001'")
   text=text.replace("'--split', 'fit')","'--split', 'fit','--start-stage','1')")
   text=text.replace("assert len(cases) == 25 and audit['passed'] and spec['version'] == 'native_onpolicy_arm_collection_v74'","assert len(cases) == 25 and audit['passed'] and spec['version']in('native_onpolicy_arm_collection_v73','native_onpolicy_arm_collection_v74')")
  dest.mkdir(exist_ok=True);(dest/p.name).write_text(text)
 print('MIXED74_READER57_FIT4_PREPARED_NOT_FROZEN')
if __name__=='__main__':main()
