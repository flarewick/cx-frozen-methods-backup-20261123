"""Recover a proved external interruption; never waive active resource checks."""
from pathlib import Path
from native_frozen100_v24.common import COHORT,O as CAMPAIGN_OUTPUT
import hashlib,json,os,math
REASON='GPU occupied before our verified arrival or arrival order unverified; stopping only our job'
def numeric(v):return type(v)in(int,float)and math.isfinite(v)
def classify(folder,*,read,live_identity):
 try:
  rows=[json.loads(x)for x in(folder/'attempt/resources.jsonl').read_text().splitlines()]
  assert [x['stopped']for x in rows if'stopped'in x]==[REASON]
  assert rows[-1].get('own_gpu_released')is True and not rows[-1].get('remaining_own_gpu_pids')
  paths=[folder/'protocol.json',folder/'attempt/launch.json',folder/'attempt/barrier/normal_child_exit.json',folder/'attempt/barrier/owned_pid_permit.json',folder/'attempt/resources.jsonl',folder/'GPU_host.log']
  spec=read(paths[0]);launch=read(paths[1]);child=read(paths[2]);permit=read(paths[3]);signals=[(p,read(p))for p in folder.glob('signal_received_*.json')]
  assert signals and all(s['signal']==2 for p,s in signals)
  assert child['exit_code']==1 and child['normal_exit']is False and child['proc_absent_after_wait']is True and child['cpu_only_fixture']is False
  assert child['child_inner_pid']==child['waited_child_inner_pid']
  assert child['attempt_nonce']==launch['nonce']and child['protocol_identity']==spec['identity']
  assert child['gpu_uuid']==launch['gpu']['uuid']==spec['required_gpu_uuid']==permit['gpu_uuid']
  ticks=os.sysconf('SC_CLK_TCK');assert all(s['inner_pid']==child['child_inner_pid']and child['child_started_ticks']*1e9/ticks<=s['monotonic_ns']<=child['waited_monotonic_ns']for p,s in signals)
  log=(folder/'GPU_host.log').read_text()
  assert 'KeyboardInterrupt: Recorded external SIGINT'in log
  assert not any(s in log for s in('CUDA out of memory','OutOfMemoryError','AssertionError','ValueError:','TypeError:','FileNotFoundError:','MemoryError:','Nonfinite','source changed'))
  assert not any('RuntimeError:'in line and line!='RuntimeError: Traceback (most recent call last):'for line in log.splitlines())
  prior=set();foreign=set();first_foreign=None
  for row in rows:
   if'seconds'not in row:continue
   assert numeric(row.get('memory_mib'))and row['memory_mib']<=24563
   assert numeric(row.get('account_gpu_count'))and row['account_gpu_count']<=3
   assert all(numeric(v)and v<=1600 for v in row.get('cpu',{}).values())
   assert all(numeric(v)and v<=128*2**30 for v in row.get('rss_bytes',{}).values())
   seen=set(row.get('raw_foreign_pids',[]))
   if seen:
    assert not row.get('gpu_pids')and seen<=prior and seen<=set(row.get('normal_child_previously_metric_verified_pids',[]))
    foreign|=seen;first_foreign=first_foreign or row
   else:
    assert not foreign,'Former GPU owner reappeared after termination'
    prior.update(set(row.get('gpu_pids',[]))&set(row.get('normal_child_previously_metric_verified_pids',[])))
   if row['seconds']>=280 and row.get('gpu_pids'):
    assert min(row['mean_utilization'],row['time_weighted_mean_utilization'],row['rolling_300s_utilization'])>=34 and row['memory_mib']>=4096
    for pid in row['gpu_pids']:
     k=str(pid);assert numeric(row['process_utilization'][k])and min(row['process_mean_utilization'][k],row['process_rolling_300s_utilization'][k])>=34 and row['process_memory_mib'][k]>=4096 and 40<=row['cpu'][k]<=1600
  assert foreign=={permit['host_GPU_PID']}and foreign<=prior
  assert all(live_identity(pid)is None for pid in prior|foreign)
  # Child wait proof was already on disk when the residue telemetry completed.
  assert paths[2].stat().st_mtime_ns<=min(s['unix_ns']for s in first_foreign['process_metric_samples'])
  from native_frozen100_v24.common import verify
  verify(spec)
  def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
  return dict(passed=True,attempt=str(folder),recognized_SIGINT_bound_to_actual_reaped_child=True,actual_child_exit_code=1,old_partial_never_qualified=True,only_former_metric_identity_verified_own_PID_in_foreign_list=sorted(foreign),all_former_GPU_PIDs_absent=True,actual_GPU_release_proved=True,all_other_resource_failures_excluded=True,source_inputs_verified=True,whole_primary_batch_must_rerun=True,actual_new_admission_required=True,original_external_interruption_counter_limit_still_applies=True,input_sha256={str(p):sha(p)for p in paths+[p for p,s in signals]})
 except (AssertionError,KeyError,TypeError,ValueError,FileNotFoundError):return None
