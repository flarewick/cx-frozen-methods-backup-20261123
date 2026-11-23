"""Fresh75 uses genuine completed preparation, unchanged runtime and limits."""
from pathlib import Path
from native_policy_execution_v1.common import T
M={'native_onpolicy_arm_collection_v74':'native_onpolicy_arm_collection_v75',
   'native_prepared_FIT_collection_v29':'native_prepared_FIT_collection_v30',
   'native_onpolicy_calibration_data_v57':'native_onpolicy_calibration_data_v58'}
NOTE='\nProspective75 requires>=8 independently validated passive initial inputs spanning all5 tasks before GPU, preserves actual failedFIT1 seconds283.246/7200, qualified73 FIT0 and every fixed case. No failed masks/policy/recovery labels are reused. Runtime/teacher/math/guards unchanged. No75 FIT0. CPU before first freeze.\n'
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text
def extra(name,text):
 if name=='host.py':text=text.replace('from native_postexit_compat_v2.tail import qualify','from native_postexit_compat_v3.tail import qualify')
 if name=='collection_plan.py':
  text=text.replace('from native_mixed_FIT_preparation_v1.gate import require','from native_fixed_FIT_inputs_v5.verify import require_ready as require').replace("T/'native_postexit_compat_v2'","T/'native_postexit_compat_v3'")
  anchor="  v['identity']=identity(v);verify(v);write(O/'study.json',v);return v\n";assert text.count(anchor)==1
  text=text.replace(anchor,'  from native_fixed_FIT_inputs_v5.verify import carry_failed_FIT1\n  carry_failed_FIT1(v)\n'+anchor)
 if name=='known_preparation.py':
  anchor=" cache={};requests={};keys={};diagnostics={};refs={};canonical=[]\n";assert text.count(anchor)==1
  text=text.replace(anchor,' from native_fixed_FIT_inputs_v5.verify import extra_roots as actual_eight_roots\n extra8,inputs8=actual_eight_roots(seed_start);roots.extend(extra8);inputs.update(inputs8)\n'+anchor)
 if name=='AGENTS.md':text+=NOTE
 return text
def main():
 for old,new in M.items():
  b,n=T/old,T/new;assert not n.exists()
  for p in b.rglob('*'):
   if p.is_file()and p.suffix in('.py','.sh','.md'):
    q=n/p.relative_to(b);q.parent.mkdir(parents=True,exist_ok=True);s=rename(p.read_text())
    if old=='native_onpolicy_arm_collection_v74':s=extra(p.name,s)
    q.write_text(s);q.chmod(p.stat().st_mode&0o777)
 (T/'native_prepared_FIT_collection_v30/scope.py').write_text('from native_prepared_FIT_collection_v29.scope import transform as previous_transform,restore_runtime\nfrom native_fixed_FIT_inputs_v5.prepare import rename,extra\ndef transform(name,text):return extra(name,rename(previous_transform(name,text)))\n')
 p=T/'native_prepared_FIT_collection_v30/checks.py';s=p.read_text().replace("T/'native_postexit_compat_v2'","T/'native_postexit_compat_v3'").replace('folders=[',"folders=[T/'native_fixed_FIT_inputs_v5',",1);p.write_text(s)
 n=T/'native_postexit_compat_v3';assert not n.exists();n.mkdir()
 for name in['tail.py','AGENTS.md']:
  s=(T/'native_postexit_compat_v2'/name).read_text().replace('native_postexit_compat_v2/tail.py','native_postexit_compat_v3/tail.py').replace("spec['version'] == 'native_onpolicy_arm_collection_v74'","spec['version'] == 'native_onpolicy_arm_collection_v75'")
  (n/name).write_text(s)
 p=T/'native_onpolicy_calibration_data_v58/reader.py';s=p.read_text();anchor="  if spec.get('version')=='native_onpolicy_arm_collection_v75':\n";assert s.count(anchor)==1
 s=s.replace(anchor,anchor+"   if spec['stage']=='fit_001':\n    assert spec['stage_total_GPU_budget_seconds']==7200 and spec['inherited_stage_GPU_seconds']==283.246\n    assert spec['actual_failed_FIT1_GPU_seconds_retained']and spec['eight_actual_prepared_inputs_all_five_tasks_before_GPU']\n")
 p.write_text(s)
 for old,new,dataold,datanew in [('native_onpolicy_calibration_fit_v4','native_onpolicy_calibration_fit_v5','native_onpolicy_calibration_data_v57','native_onpolicy_calibration_data_v58'),('native_collection74_supervision_v1','native_collection75_supervision_v1','native_onpolicy_calibration_data_v57','native_onpolicy_calibration_data_v58')]:
  b,n=T/old,T/new;assert not n.exists()
  for p in b.rglob('*'):
   if p.is_file()and p.suffix in('.py','.sh','.md'):
    q=n/p.relative_to(b);q.parent.mkdir(parents=True,exist_ok=True);s=p.read_text().replace(old,new).replace(dataold,datanew)
    if old.startswith('native_collection'):
     s=s.replace('native_onpolicy_arm_collection_v74','native_onpolicy_arm_collection_v75').replace('native_onpolicy_calibration_fit_v4','native_onpolicy_calibration_fit_v5').replace("spec['inherited_stage_GPU_seconds'] == 0","spec['inherited_stage_GPU_seconds'] == 283.246")
    elif p.name=='host.py':s=s.replace(" sys.path.insert(0,str(P/'deployment'));"," assert not(T/'outputs/native_onpolicy_calibration_fit_v4/runs').exists()and not list((T/'outputs/native_onpolicy_calibration_fit_v4').glob('host_run_*')),'Prior v4 fitting budget must not be reset'\n sys.path.insert(0,str(P/'deployment'));")
    elif p.name=='train.py':s=s.replace("'native_onpolicy_calibration_fit_v3','native_onpolicy_calibration_fit_v5'","'native_onpolicy_calibration_fit_v3','native_onpolicy_calibration_fit_v4','native_onpolicy_calibration_fit_v5'")
    q.write_text(s);q.chmod(p.stat().st_mode&0o777)
 print('FRESH75_EIGHT_READY_PREFIX_READER58_FIT5_PREPARED_NOT_FROZEN')
if __name__=='__main__':main()
