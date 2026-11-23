"""Fail-closed startup-recovery decisions; no service or GPU is started."""
import copy
from .preparation_recovery import decision
from .controller import T,O,read,save
def main():
 old=T/'outputs/native_cross_task100_v11/batches/004/runs/94f5e547661349e4adcdffe374413880';rows=[__import__('json').loads(x)for x in(old/'attempt/resources.jsonl').read_text().splitlines()]
 new=dict(completed_expert_rows=13,cached_valid_cases=8,primary_policy_results_imported=False,old_masks_or_teacher_outputs_imported=False)
 assert decision(rows,'',None,new,0)[0]
 cases=[]
 def reject(name,r=rows,log='',prior=None,n=new,count=0,alive=False):
  okay,reason=decision(r,log,prior,n,count,alive);assert not okay,name;cases.append(dict(case=name,reason=reason))
 reject('same_preparation',prior=new);reject('recovery_limit',count=2);reject('live_old_GPU_PID',alive=True);reject('software_error',log='AssertionError: software mismatch')
 samples=[i for i,r in enumerate(rows)if r.get('gpu_pids')and'seconds'in r];i=samples[-1];pid=str(rows[i]['gpu_pids'][0])
 for name,field,value in [('late_rolling_failure','seconds',500),('device_VRAM','memory_mib',24564),('account_limit','account_gpu_count',4)]:
  changed=copy.deepcopy(rows);changed[i][field]=value;reject(name,r=changed)
 for name,field,value in [('None_utilization','process_mean_utilization',None),('CPU_limit','cpu',1601),('process_VRAM','process_memory_mib',24564),('RSS_limit','rss_bytes',129*2**30)]:
  changed=copy.deepcopy(rows);changed[i][field][pid]=value;reject(name,r=changed)
 changed=copy.deepcopy(rows);changed[-1]['own_gpu_released']=False;reject('release_not_proved',r=changed)
 reject('primary_outcomes',n=dict(new,primary_policy_results_imported=True));reject('old_masks',n=dict(new,old_masks_or_teacher_outputs_imported=True))
 save(O/'CPU_preparation_recovery_checks.json',dict(passed=True,actual_startup_failure_with_more_preparation_can_recover=True,unchanged_unknown_late_VRAM_CPU_RSS_account_software_livePID_release_failures_rejected=cases,maximum_two_preparation_recoveries_per_logical_batch=True,no_service_calls=True,no_GPU_calls=True));print('STRICT_PREPARATION_RECOVERY_CPU_PASS',flush=True)
if __name__=='__main__':main()
