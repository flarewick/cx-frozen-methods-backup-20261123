from native_fit_support_review_v1.support import minimum_fit,evidence as support_evidence
"""Publish an audited325 manifest only after the actual CPU builder exited0."""
from pathlib import Path
import os,sys,subprocess,time
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import verify
from .dataset import BASE,ARM_BASE
O=T/'outputs/native_onpolicy_calibration_data_v60'
def main():
 for split in('fit','val'):
  assert any((T/'outputs'/v/(split+'_complete.json')).exists()for v in BASE),'Original fixed250 incomplete; no builder started'
  assert any((T/'outputs'/v/(split+'_complete.json')).exists()for v in ARM_BASE),'Additional fixed75 incomplete; no builder started'
 assert not(O/'build_host_exit.json').exists()and not(O/'qualified_dataset.json').exists(),'Inspect prior actual build; no silent retry'
 P=T.parents[1];sys.path.insert(0,str(P/'deployment'));import user_container as container
 canonical=Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA');script=canonical/(T/'native_onpolicy_collection_v21/scripts/cpu.sh').relative_to(P);O.mkdir(parents=True,exist_ok=True)
 with(O/'build_host.log').open('x')as log:
  process=subprocess.Popen(container.command(['bash',str(script),'-m','native_onpolicy_calibration_data_v60.dataset'],mount_project=True),stdout=log,stderr=subprocess.STDOUT);code=process.wait()
 actual=O/'build_host_exit.json';write(actual,dict(exit_code=code,actual_Popen_wait_return=True,pid=process.pid,proc_absent_after_wait=not Path('/proc',str(process.pid)).exists()));assert code==0,'Failed builder retained; no automatic retry'
 pending=O/'qualified_dataset_pending_actual_build_exit.json';value=read(pending);verify(value)
 assert value['complete']and value['identities']==325 and value['FIT']==250 and value['VAL']==75 and value['optimizer_steps']==0 and value['actual_build_host_exit_pending']
 assert all(g['adapter_fit']['eligible_distinct_cases']>=minimum_fit() and g['adapter_validation']['eligible_distinct_cases']>=1 for g in value['groups'].values())
 value.update(actual_build_host_exit_pending=False,actual_build_host_exit0=True,build_host_exit_file=str(actual),build_host_exit_sha256=sha(actual));value['input_sha256'].update({str(pending):sha(pending),str(actual):sha(actual)})
 write(O/'qualified_dataset.json',value);print('ACTUAL_CPU325_BUILD_EXIT0_QUALIFIED',flush=True)
if __name__=='__main__':main()
