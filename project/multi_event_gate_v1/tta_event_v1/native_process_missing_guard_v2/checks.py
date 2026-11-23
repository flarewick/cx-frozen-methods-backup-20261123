"""Real C20 raw idle replies and adversarial cases; no GPU work or data promotion."""
from pathlib import Path
import copy,json,sys,random,ast
from native_policy_execution_v1.common import T,read,write,sha
from .empty import admissible_missing,zero_previous
from .rewrite import transform
sys.path.insert(0,str(T.parents[1]/'deployment'))
from gpu_utilization_window import UtilizationWindow
O=T/'outputs/native_process_missing_guard_v2'
def main():
 source=T/'native_process_sampler_v13/strict_guard.py';policy=read(T/'native_process_sampler_v13/policy.json');assert sha(source)==policy['terminal_guard_source_sha256'];compile(transform(source.read_text()),str(source),'exec')
 parent=T/'outputs/native_onpolicy_collection_v20/fit_004';root=Path(read(parent/'execution_state.json')['root']);actual=read(parent/'actual_host_wait_exit.json');assert actual['exit_code']==1 and actual['proc_absent_after_wait']
 resources=root/'attempt/resources.jsonl';rows=[json.loads(x)for x in resources.read_text().splitlines()];target=next(r for r in reversed(rows)if r.get('process_utilization_initial_missing_pids')and r.get('seconds',0)>=280);active=target['gpu_pids'];values={int(p):v for p,v in target['process_utilization'].items()if v is not None};samples=target['process_metric_samples'];uuid=samples[-1]['gpu_uuid'];index=samples[-1]['gpu'];eligible=admissible_missing(active,values,samples,uuid,index);assert set(eligible)==set(active)
 lower=UtilizationWindow();original=UtilizationWindow();sums={};count=0;last=None
 for row in rows:
  if 'seconds'not in row:continue
  count+=1;live=row['gpu_pids'];v={int(p):value for p,value in row['process_utilization'].items()if value is not None};allowed=admissible_missing(live,v,row['process_metric_samples'],uuid,index);zero_previous(lower,UtilizationWindow(),allowed)
  for pid in live:
   if pid in v:sums[pid]=sums.get(pid,0)+v[pid]
  ca,ra=lower.add(row['seconds'],0,{p:v.get(p,0)for p in live});cb,rb=original.add(row['seconds'],0,{p:v.get(p,0)for p in live})
  for pid in live:assert ca[pid]<=cb[pid]+1e-12 and ra[pid]<=rb[pid]+1e-12
  if row is target:last={str(pid):dict(cumulative_lower_bound=min(sums.get(pid,0)/count,ca[pid]),rolling_lower_bound=ra[pid],original_cumulative=cb[pid])for pid in live}
 assert last and all(v['cumulative_lower_bound']>=34 and v['rolling_lower_bound']>=34 for v in last.values())
 assert all(target['process_utilization'][str(p)]is None for p in active),'Never replace None with a measured zero'
 rejected=0
 for kind in('stale_clock','transport_timeout','previous_frame','wrong_uuid','API_error'):
  bad=copy.deepcopy(samples)
  if kind=='stale_clock':bad[-1]['raw_NVML']['frames'][-1]['host_timestamp_us']+=10000000
  elif kind=='transport_timeout':bad[-1]['transport']['request_deadline_seconds']=4
  elif kind=='previous_frame':bad[-1]['previous_call_data_used']=True
  elif kind=='wrong_uuid':bad[-1]['raw_NVML']['gpu_uuid']='GPU-other'
  else:bad[-1]['raw_NVML']['frames'][-1]['API_status']='GPU_IS_LOST'
  try:accepted=admissible_missing(active,values,bad,uuid,index)
  except (ValueError,KeyError):accepted={}
  assert not accepted,kind;rejected+=1
 # Across arbitrary readings, changing an unknown interval to0 can never raise cumulative/rolling.
 rng=random.Random(17)
 for run in range(40):
  a=UtilizationWindow();b=UtilizationWindow();t=0.
  for step in range(150):
   t+=rng.uniform(.2,8);value=rng.uniform(0,100);missing=rng.random()<.2
   if missing:a.previous[1]=0.
   ca,ra=a.add(t,0,{1:0 if missing else value});cb,rb=b.add(t,0,{1:0 if missing else value});assert ca[1]<=cb[1]+1e-10 and ra[1]<=rb[1]+1e-10
 low=UtilizationWindow()
 for n in range(1,302):ca,ra=low.add(n,0,{1:0})
 assert ca[1]==ra[1]==0 and ca[1]<34
 preserved=[]
 for version in('native_onpolicy_collection_v18','native_onpolicy_collection_v19'):
  d=T/'outputs'/version/'fit_004';r=Path(read(d/'execution_state.json')['root']);rr=[json.loads(x)for x in(r/'attempt/resources.jsonl').read_text().splitlines()];last_live=next(x for x in reversed(rr)if x.get('gpu_pids'));assert last_live['seconds']>=280 and min(last_live['time_weighted_mean_utilization'],*last_live['process_mean_utilization'].values())<34;preserved.append(version)
 c23=T/'outputs/native_onpolicy_collection_v23/fit_004';c23root=Path(read(c23/'execution_state.json')['root']);failure_metric=c23root/'last_successfully_collected_metric_at_failure.json';metric=read(failure_metric)['metric']
 from native_process_sampler_v13.metrics import select
 c23rows=[json.loads(x)for x in(c23root/'attempt/resources.jsonl').read_text().splitlines()];failed=next(x for x in reversed(c23rows)if x.get('process_utilization_initial_missing_pids'));actual_samples=failed['process_metric_samples'];assert len(actual_samples)==3
 actual_pid=3106568;assert failed['process_utilization'][str(actual_pid)]is None
 fresh=admissible_missing([actual_pid],{},actual_samples,metric['gpu_uuid'],metric['gpu']);assert actual_pid in fresh
 from native_process_idle_guard_v1.empty import admissible_missing as prior
 assert not prior([actual_pid],{},actual_samples,metric['gpu_uuid'],metric['gpu'])
 data=actual_samples[0]['raw_NVML'];begin=data['frames'][0]['driver_timestamp_us'];end=data['frames'][-1]['driver_timestamp_us'];assert any(v['value']>0 for v in data['frames'][-1]['device_samples']['SM']if begin<v['timestamp_us']<=end)
 assert metric['previous_call_data_used']is False and metric['device_wide_substitution']is False and metric['missing_records_inferred_zero']is False
 c23resources=c23root/'attempt/resources.jsonl';inputs_extra=[failure_metric,c23resources,c23/'actual_host_wait_exit.json']
 # Exact production windows are preserved; missing readings stay unavailable, never receive device utilization.
 assert(Path(__file__).parent/'rewrite.py').read_text()==(T/'native_process_idle_guard_v1/rewrite.py').read_text()
 proof=dict(passed=True,actual_C20_final_idle_empty_reply_verified=True,actual_C23_nonzero_dense_clock_empty_reply_admitted_as_unknown_zero_lower_bound=True,actual_C23_old_predicate_rejection_reproduced=True,actual_C20_post_zero_lower_bound=last,instantaneous_None_preserved=True,CPU_random_window_sequences=40,unknowns_can_never_increase_original_averages=True,invalid_telemetry_cases_rejected=rejected,all_missing_startup_still_fails34=True,actual_C18_C19_low_mean_failures_still_fail=preserved,original_AST_exact_except_enumerated_idle_lower_bound_edits=True,original_34_280_memory_CPU_RSS_account_arrival_deadline_checks_retained=True,no_GPU_work=True,no_failed_scientific_data_promoted=True,source_sha256={str(f):sha(f)for f in Path(__file__).parent.glob('*')if f.is_file()},input_sha256={str(f):sha(f)for f in(source,T/'native_process_sampler_v13/metrics.py',T/'native_account_stream_v1/guard.py',T.parents[1]/'deployment/gpu_utilization_window.py',resources,parent/'actual_host_wait_exit.json',*inputs_extra)})
 write(O/'CPU_actual_C20_idle_lower_bound_checks.json',proof);print('ACTUAL_C20_IDLE_FRAME_ZERO_LOWER_BOUND_AND_ALL_ORIGINAL_LIMITS_PASS',flush=True)
if __name__=='__main__':main()
