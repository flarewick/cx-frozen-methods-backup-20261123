"""Compare incomplete traces against independently known complete period truth."""
from pathlib import Path
import random,json,hashlib
from .device_window import DeviceWindow
T=Path(__file__).resolve().parents[1];O=T/'outputs/native_process_sampler_v13';EPOCH=1700000000000000

def packet(values,drop=()):
 rows=[dict(timestamp_us=EPOCH+i*200000,value=int(value))for i,value in enumerate(values,1)if i not in drop]
 return dict(gpu_uuid='GPU-CPU-truth',frames=[dict(device_samples={name:[dict(r)for r in rows]for name in('SM','bandwidth')})])
def full_period_truth(values,begin,end):
 return sum(value*max(0,min(EPOCH+i*200000,end)-max(EPOCH+(i-1)*200000,begin))for i,value in enumerate(values,1))
def run(values,drop=()):
 w=DeviceWindow(EPOCH,'GPU-CPU-truth');end=EPOCH+len(values)*200000+100000
 r=w.update(packet(values,drop),end,(end-EPOCH)/1e6,steady_required=True);return w,r,end

def main():
 w,r,end=run([80]*200,drop={100})
 assert r['untrusted_internal_us']==400000 and r['sample_mean']==80 and r['cumulative_lower_bound']>=34
 assert len(w.lookup)==199 and set(w.lookup)=={EPOCH+i*200000 for i in range(1,201)if i!=100}and r['no_synthetic_zero_records']and r['internal_unknown_time_never_credited']
 assert r['cumulative_lower_bound']*float(end-EPOCH)<=full_period_truth([80]*200,EPOCH,end)+1e-6
 # A superficially35% point mean must FAIL once enough unknown time is retained.
 _,bad,_=run([35]*200,drop=set(range(50,80)))
 assert bad['sample_mean']==35 and bad['cumulative_lower_bound']<34 and bad['rolling300_lower_bound']<34
 rng=random.Random(17);trials=[]
 for n in range(100):
  values=[rng.randrange(101)for _ in range(1800)];drop={i for i in range(2,1799)if rng.random()<.08};obj,result,now=run(values,drop)
  for seconds,key in[(None,'cumulative_lower_bound'),(300,'rolling300_lower_bound'),(60,'rolling60_lower_bound')]:
   begin=EPOCH if seconds is None else max(EPOCH,now-int(seconds*1e6));truth=full_period_truth(values,begin,now)/(now-begin)
   assert result[key]<=truth+1e-10,('Incomplete trace overestimated complete ground truth',n,key,result[key],truth)
  assert obj.start==EPOCH and result['maximum_credited_native_interval_us']==250000
  trials.append(dict(trial=n,dropped_points=len(drop),untrusted_us=result['untrusted_internal_us'],bounded_mean=result['cumulative_lower_bound']))
 # Actual later-published point is inserted; no invented sample/cumulative reset.
 now=EPOCH+201*200000+100000
 later=w.update(packet([80]*201),now,(now-EPOCH)/1e6,steady_required=True)
 assert later['untrusted_internal_us']==0 and w.start==EPOCH and later['backfilled_actual_records']>=1 and w.retrospective(end)['cumulative_lower_bound']>=r['cumulative_lower_bound']
 # Unknown actual component, stale tail, and UUID mismatch remain hard errors.
 for mode in('unknown','stale','UUID'):
  data=packet([80]*20);now=EPOCH+4100000
  if mode=='unknown':data['frames'][0]['device_samples']['SM'][4]['value']=None
  if mode=='stale':now+=1000000
  if mode=='UUID':data['gpu_uuid']='GPU-wrong'
  try:DeviceWindow(EPOCH,'GPU-CPU-truth').update(data,now,(now-EPOCH)/1e6)
  except ValueError:pass
  else:raise AssertionError('Invalid metadata waived: '+mode)
 O.mkdir(parents=True,exist_ok=True)
 proof=dict(passed=True,actual_CPU_ground_truth_trials=trials,unknown_time_no_credit_but_full_denominator=True,no_synthetic_raw_zero_samples=True,positive_case80_percent_lower_bound_still_above34=True,negative_case35_point_mean_but_lower_bound_below34_rejected=True,cumulative_rolling300_rolling60_never_exceeded_complete_known_period_truth=True,late_actual_backfill_no_clock_reset=True,unknown_value_stale_tail_UUID_still_hard_errors=True,maximum_credited_interval_us=250000,minimum_left_and_right_integrals=True,no_CUDA_or_model_or_GPU_execution=True,source_sha256=hashlib.sha256(Path(__file__).with_name('device_window.py').read_bytes()).hexdigest())
 with(O/'CPU_conservative_checks.json').open('x')as f:json.dump(proof,f,indent=2);f.write('\n')
 print('CONSERVATIVE_GAP_100_GROUND_TRUTH_TRIALS_PASS')
if __name__=='__main__':main()
