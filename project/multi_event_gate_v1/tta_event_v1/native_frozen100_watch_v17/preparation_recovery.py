"""Only startup low-utilization failures with strictly more verified real preparation."""
import math,time,json,subprocess
from pathlib import Path
from native_frozen100_v17.common import COHORT,O as CAMPAIGN_OUTPUT
def numeric(value):return type(value)in(int,float)and math.isfinite(value)
def decision(rows,log,old_cache,new_cache,count,former_pid_alive=False):
 if count>=2:return False,'Two preparation recoveries already used in this logical batch'
 if former_pid_alive:return False,'Former GPU PID still alive'
 if not rows or not rows[-1].get('own_gpu_released')or rows[-1].get('remaining_own_gpu_pids'):return False,'GPU release not proved'
 stops=[r['stopped']for r in rows if'stopped'in r]
 legal=('Sustained GPU utilization/VRAM threshold would not be met','Per-process cumulative/rolling GPU utilization is unverified or below 34%')
 if len(stops)!=1 or stops[0]not in legal:return False,'Not a supported startup utilization failure'
 if any(s in log for s in('CUDA out of memory','OutOfMemoryError','AssertionError','ValueError:','TypeError:','FileNotFoundError:','MemoryError:','RuntimeError:','Nonfinite','source changed')):return False,'Software or numerical error present'
 samples=[r for r in rows if r.get('gpu_pids')and'seconds'in r]
 if not samples:return False,'No actual GPU owner sample'
 row=samples[-1]
 if not numeric(row['seconds'])or not 280<=row['seconds']<=310:return False,'Failure not in first startup resource check'
 if len(row['gpu_pids'])!=1:return False,'Not exactly one GPU owner'
 if not numeric(row.get('memory_mib'))or not 4096<=row['memory_mib']<=24563:return False,'Device VRAM violation'
 if not numeric(row.get('account_gpu_count'))or row['account_gpu_count']>3:return False,'Account GPU violation'
 metrics=[row.get(k)for k in('mean_utilization','time_weighted_mean_utilization','rolling_300s_utilization')]
 for pid in row['gpu_pids']:
  key=str(pid)
  for name in('process_utilization','process_mean_utilization','process_rolling_300s_utilization'):
   metrics.append(row.get(name,{}).get(key))
  cpu=row.get('cpu',{}).get(key);mem=row.get('process_memory_mib',{}).get(key);rss=row.get('rss_bytes',{}).get(key)
  if not numeric(cpu)or not 40<=cpu<=1600 or not numeric(mem)or not 4096<=mem<=24563 or not numeric(rss)or rss>128*2**30:return False,'CPU/process VRAM/RSS violation or missing metric'
 if not all(numeric(v)for v in metrics):return False,'Unknown utilization is never waived'
 # An instant dip alone does not permit a retry; actual averages must fail.
 if min(metrics[:3]+metrics[4:])>=34:return False,'No actual low cumulative or rolling utilization'
 old=old_cache or{};new=new_cache or{}
 if new.get('primary_policy_results_imported',True)or new.get('old_masks_or_teacher_outputs_imported',True):return False,'Illegal preparation inputs'
 if new.get('completed_expert_rows',0)<=old.get('completed_expert_rows',0):return False,'No new completed real expert preparation; unchanged retry forbidden'
 if new.get('cached_valid_cases',0)<old.get('cached_valid_cases',0):return False,'Prepared case coverage regressed'
 return True,'Startup bottleneck diagnosed; strictly more verified completed preparation removes repeated GPU screening'
def run(previous,state,control,*,T,O,UNIT,ENV,read,save,event,unit,live_identity):
 campaign=CAMPAIGN_OUTPUT/'campaign_failure.json'
 if not campaign.exists():return False
 failure=read(campaign)
 if failure.get('run_id')!=previous.name:return False
 if any(name in failure.get('traceback','')for name in('AssertionError','ValueError:','TypeError:','FileNotFoundError:','MemoryError:')):return False
 last_exception=failure.get('traceback','').strip().splitlines()[-1]if failure.get('traceback','').strip()else ''
 if last_exception not in('SystemExit: 1','RuntimeError: Sustained GPU utilization/VRAM threshold would not be met','RuntimeError: Per-process cumulative/rolling GPU utilization is unverified or below 34%'):return False
 rows=[json.loads(x)for x in(previous/'attempt/resources.jsonl').read_text().splitlines()];log=(previous/'GPU_host.log').read_text()if(previous/'GPU_host.log').exists()else'';spec=read(previous/'protocol.json');batch=previous.parents[1].name
 count=control.get('preparation_recoveries',{}).get(batch,0)
 alive=any(live_identity(pid)is not None for r in rows for pid in r.get('gpu_pids',[]))
 # Cheap conservative precheck before hashing any new preparation.
 trial=dict(completed_expert_rows=(spec.get('cached_preparation')or{}).get('completed_expert_rows',0)+1,cached_valid_cases=spec['cases_per_batch'],primary_policy_results_imported=False,old_masks_or_teacher_outputs_imported=False)
 okay,reason=decision(rows,log,spec.get('cached_preparation'),trial,count,alive)
 if not okay:return False
 from native_frozen100_v17.preparation import collect
 from native_frozen100_v17.common import verify,sha
 from native_fast_inventory_v1.inventory import inventory
 study=read(CAMPAIGN_OUTPUT/'study.json');verify(study)
 try:prepared=collect(study,spec['batch_index'],spec['initial_screen_state'])
 except Exception as exc:
  save(O/'preparation_recovery_rejected.json',dict(time=time.time(),reason='Preparation validation failed',exception=str(exc),attempt=str(previous)));return False
 okay,reason=decision(rows,log,spec.get('cached_preparation'),prepared,count,alive)
 if not okay:
  save(O/'preparation_recovery_rejected.json',dict(time=time.time(),reason=reason,attempt=str(previous)));return False
 fleet=inventory()
 if fleet['own_gpu_count']>=3 or not fleet['available_indices']:return False
 current=unit(UNIT)
 if current.get('ActiveState')in('active','activating','deactivating')or current.get('InvocationID')!=state.get('InvocationID'):return False
 # Freeze the newly available CPU-only preparation proof before requesting a start.
 proof=dict(time=time.time(),attempt=str(previous),reason=reason,logical_batch=batch,old_completed_expert_rows=(spec.get('cached_preparation')or{}).get('completed_expert_rows',0),new_completed_expert_rows=prepared['completed_expert_rows'],new_cached_valid_cases=prepared['cached_valid_cases'],new_preparation_input_sha256=prepared['input_sha256'],old_resource_sha256=sha(previous/'attempt/resources.jsonl'),whole_primary_branches_must_rerun=True,resource_thresholds_and_GPU_clock_unchanged=True,actual_new_GPU_admission_still_required=True,recovery_number=count+1)
 save(O/('verified_preparation_recovery_'+batch+'_'+str(count+1)+'.json'),proof)
 control.setdefault('preparation_recoveries',{})[batch]=count+1;control['handled_invocations'].append(state.get('InvocationID',''));control['was_active']=False;save(O/'control.json',control)
 blocked=O/'blocked.json'
 if blocked.exists():
  target=O/'past_blocks'/('before_preparation_'+batch+'_'+str(time.time_ns())+'.json');target.parent.mkdir(parents=True,exist_ok=True);blocked.rename(target)
 result=subprocess.run(['systemctl','--user','start','--no-block',UNIT],env=ENV,capture_output=True,text=True,timeout=10)
 if result.returncode:raise RuntimeError(result.stderr)
 event('verified_new_preparation_recovery_started',**{k:v for k,v in proof.items()if k!='time'});save(O/'last_action.json',dict(time=time.time(),action='started_with_strictly_more_verified_preparation',**{k:v for k,v in proof.items()if k!='time'}));return True
