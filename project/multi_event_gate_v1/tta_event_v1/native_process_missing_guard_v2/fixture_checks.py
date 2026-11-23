"""CPU-only telemetry audit roundtrip. Never a native data qualification."""
from pathlib import Path
import copy,json,sys
from native_policy_execution_v1.common import T,read,write,sha
from .empty import admissible_missing,zero_previous
from .audit import audit
sys.path.insert(0,str(T.parents[1]/'deployment'))
from gpu_utilization_window import UtilizationWindow
O=T/'outputs/native_process_missing_guard_v2'
def main():
 d=T/'outputs/native_onpolicy_collection_v20/fit_004';r=Path(read(d/'execution_state.json')['root']);source=r/'attempt/resources.jsonl';original=[json.loads(x)for x in source.read_text().splitlines()];spec=read(r/'protocol.json');root=O/'CPU_telemetry_roundtrip_fixture';(root/'attempt').mkdir(parents=True,exist_ok=False)
 protocol=dict(CPU_fixture_only=True,scope='CPU_only_resource_window_roundtrip_no_native_qualification',conservative_idle_missing_lower_bound_guard=True,required_gpu_uuid=spec['required_gpu_uuid']);write(root/'protocol.json',protocol)
 window=UtilizationWindow();recent=UtilizationWindow(60);sums={};count=0;rows=[]
 for row in original:
  if 'seconds'not in row:continue
  row=copy.deepcopy(row);count+=1;active=row['gpu_pids'];values={int(p):v for p,v in row['process_utilization'].items()if v is not None};samples=row['process_metric_samples'];allowed=admissible_missing(active,values,samples,protocol['required_gpu_uuid'],samples[-1]['gpu'])
  row['missing_process_zero_lower_bound']={str(k):v for k,v in allowed.items()};row['zeroed_missing_interval']=zero_previous(window,recent,allowed);elapsed=row['seconds'];row['lower_bound_exact_elapsed']=elapsed
  for pid in active:
   if pid in values:sums[pid]=sums.get(pid,0)+values[pid]
  ca,ra=window.add(elapsed,0,{p:values.get(p,0)for p in active});_,rr=recent.add(elapsed,0,{p:values.get(p,0)for p in active})
  row['process_mean_utilization']={str(p):min(sums.get(p,0)/count,ca.get(p,0))for p in active};row['process_rolling_300s_utilization']={str(p):ra.get(p,0)for p in active};row['process_recent_60s_utilization']={str(p):rr.get(p,0)for p in active};rows.append(row)
 rows.append(dict(CPU_fixture_only=True,original_failed_job_remains_failed=True,own_gpu_released=True,remaining_own_gpu_pids=[]))
 with(root/'attempt/resources.jsonl').open('x')as f:
  for row in rows:f.write(json.dumps(row)+'\n')
 checked=audit(root,protocol);assert checked['passed']and checked['accepted_fresh_empty_intervals']>=1
 bad=O/'CPU_tampered_telemetry_fixture';(bad/'attempt').mkdir(parents=True,exist_ok=False);write(bad/'protocol.json',protocol);tampered=copy.deepcopy(rows);next(row for row in tampered if row.get('gpu_pids'))['process_mean_utilization'][str(next(row for row in tampered if row.get('gpu_pids'))['gpu_pids'][0])]+=1
 with(bad/'attempt/resources.jsonl').open('x')as f:
  for row in tampered:f.write(json.dumps(row)+'\n')
 try:audit(bad,protocol)
 except AssertionError:pass
 else:raise AssertionError('Inflated telemetry passed exact re-audit')
 assert not(root/'atomic_full_recovery_qualification.json').exists()and not(r/'atomic_full_recovery_qualification.json').exists()
 write(O/'CPU_actual_resource_audit_roundtrip_checks.json',dict(passed=True,actual_C20_raw_frames_recomputed_with_CPU_fixture_timing=True,CPU_fixture_only=True,exact_process_window_auditor_roundtrip=True,tampered_average_rejected=True,no_failed_C20_data_promoted=True,no_GPU_work=True,input_sha256={str(p):sha(p)for p in(source,root/'protocol.json',root/'attempt/resources.jsonl',root/'CPU_idle_missing_lower_bound_audit.json',bad/'attempt/resources.jsonl')},source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()}));print('CPU_ACTUAL_C20_RAW_FRAME_AUDIT_ROUNDTRIP_AND_TAMPER_REJECTION_PASS',flush=True)
if __name__=='__main__':main()
