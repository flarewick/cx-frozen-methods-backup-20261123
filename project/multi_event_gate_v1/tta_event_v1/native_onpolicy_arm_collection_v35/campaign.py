"""Sequential complete stages, no failed-stage retry and actual host waits."""
from pathlib import Path
import argparse,os,subprocess,sys
from .common import T,read,sha,mapped
from native_policy_execution_v1.common import write
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=('fit','val'),required=True);p.add_argument('--start-stage',type=int,default=0);a=p.parse_args();base=T/'outputs/native_onpolicy_arm_collection_v35';stages=[]
 assert 0<=a.start_stage<(2 if a.split=='fit'else 1)
 for n in range(a.start_stage):
  from .prefix_stage import qualified
  name=a.split+'_'+str(n).zfill(3);o=qualified(a.split,n);r=mapped(read(o/'execution_state.json')['root']);assert read(o/'actual_host_wait_exit.json')['exit_code']==0 and read(r/'atomic_full_recovery_qualification.json')['passed'];stages.append(dict(stage=name,root=str(r),actual_host_exit_file=str(o/'actual_host_wait_exit.json'),actual_host_exit_sha256=sha(o/'actual_host_wait_exit.json')))
 for n in range(a.start_stage,2 if a.split=='fit'else 1):
  name=a.split+'_'+str(n).zfill(3);o=base/name;assert not(o/'actual_host_wait_exit.json').exists(),'Inspect completed/failed stage instead of silently rerunning campaign'
  process=subprocess.Popen([sys.executable,'-B','-u','-m','native_onpolicy_arm_collection_v35.launch'],env=dict(os.environ,CX_ONPOLICY_STAGE=name));code=process.wait();assert code==0,'Stage failed; no unchanged automatic retry'
  actual=read(o/'actual_host_wait_exit.json');r=mapped(read(o/'execution_state.json')['root']);assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']and read(r/'atomic_full_recovery_qualification.json')['passed']
  stages.append(dict(stage=name,root=str(r),actual_host_exit_file=str(o/'actual_host_wait_exit.json'),actual_host_exit_sha256=sha(o/'actual_host_wait_exit.json')));print('COMPLETE_COLLECTION_STAGE',name,flush=True)
 write(base/(a.split+'_complete.json'),dict(complete=True,split=a.split,stages=stages,cases_each_task=10 if a.split=='fit'else 5,optimizer_steps=0,no_TTA_or_formal100_started=True))
if __name__=='__main__':main()
