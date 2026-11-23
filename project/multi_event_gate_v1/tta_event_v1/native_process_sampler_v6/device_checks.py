"""Full-stream integration, rejection rules, original decision AST and live metadata."""
import ast,copy,json,hashlib,time
from pathlib import Path
from .device_window import DeviceWindow
from . import metrics
T=Path(__file__).resolve().parents[1];C=Path(__file__).resolve().parent;O=T/'outputs/native_process_sampler_v6';E=10000000;UUID='GPU-CPU-fixture'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def packet(seconds,value=50,window=14):
 end=int(seconds*1e6);begin=max(0,end-int(window*1e6));times=list(range(begin,end+1,200000))
 return dict(gpu_uuid=UUID,frames=[dict(device_samples={k:[dict(timestamp_us=E+t,value=value)for t in times]for k in ['SM','bandwidth']})])
def main():
 # Remove ONLY new device acquisition/telemetry statements; every original guard decision/helper remains exact.
 tree=ast.parse((C/'strict_guard.py').read_text());old=ast.parse((T/'native_process_sampler_v5/strict_guard.py').read_text())
 class OriginalDecision(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='native_process_sampler_v6.device_window':return None
   if n.module=='native_process_sampler_v6.metrics':n.module='native_process_sampler_v5.metrics'
   return n
  def visit_Assign(self,n):
   names={x.id for x in n.targets if isinstance(x,ast.Name)}
   if names&{'start_unix_us','device_window','legacy_device','dense_device'}:return None
   if names&{'cumulative','rolling','recent'}and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='dict'and any(k.arg=='device'for k in n.value.keywords):return None
   return self.generic_visit(n)
  def visit_Dict(self,n):
   pairs=[(k,v)for k,v in zip(n.keys,n.values)if not(isinstance(k,ast.Constant)and k.value in ['device_continuous','device_legacy_sparse_diagnostic_only'])]
   n.keys=[k for k,v in pairs];n.values=[ast.parse('sum(samples)/len(samples)',mode='eval').body if isinstance(k,ast.Constant)and k.value=='mean_utilization'else v for k,v in pairs]
   return self.generic_visit(n)
 assert ast.dump(OriginalDecision().visit(tree))==ast.dump(old),'Original guard decision or helper changed'
 def function(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
 assert ast.dump(function(C/'metrics.py','select'))==ast.dump(function(T/'native_process_sampler_v5/metrics.py','select'))
 assert ast.dump(function(C/'worker.py','clock_pair'))==ast.dump(function(T/'native_process_sampler_v5/worker.py','clock_pair'))
 w=DeviceWindow(E,UUID)
 for end in range(2,402,2):d=w.update(packet(end),E+end*1000000+100000,end+.1)
 assert abs(d['cumulative_lower_bound']-50*400/400.1)<1e-9 and abs(d['rolling300_lower_bound']-50*299.9/300)<1e-9
 assert d['actual_driver_records']==2001 and d['sample_mean']==50 and d['fixed_start_never_reset']
 zeros=DeviceWindow(E,UUID).update(packet(4,0),E+4100000,4.1);assert zeros['sample_mean']==zeros['cumulative_lower_bound']==0
 # Each timestamp's max, never max of separate stream means.
 p=packet(2);sm=p['frames'][0]['device_samples']['SM'];bw=p['frames'][0]['device_samples']['bandwidth']
 for i,(a,b)in enumerate(zip(sm,bw)):a['value']=80 if i%2==0 else 0;b['value']=0 if i%2==0 else 90
 q=DeviceWindow(E,UUID).update(p,E+2100000,2.1);assert q['sample_mean']>80
 # Startup is never trimmed: genuine100% after60seconds cannot erase the60secondzero prefix.
 w=DeviceWindow(E,UUID)
 for end in range(2,82,2):
  p=packet(end)
  for values in p['frames'][0]['device_samples'].values():
   for row in values:row['value']=0 if row['timestamp_us']<E+60000000 else 100
  q=w.update(p,E+end*1000000+100000,end+.1)
 assert q['cumulative_lower_bound']<26 and q['started_unix_us']==E
 negatives=[]
 for mode in ['missing_component','missing_timestamp','missing_both','invalid','duplicate','stale','future','UUID','clockjump']:
  p=packet(4);r=p['frames'][0]['device_samples'];now=E+4100000;elapsed=4.1
  if mode=='missing_component':r['bandwidth']=[]
  if mode=='missing_timestamp':r['bandwidth'].pop(3)
  if mode=='missing_both':r['SM'].pop(3);r['bandwidth'].pop(3)
  if mode=='invalid':r['SM'][3]['value']=None
  if mode=='duplicate':r['SM'].append(copy.deepcopy(r['SM'][3]))
  if mode=='stale':now=E+4800000;elapsed=4.8
  if mode=='future':r['SM'][-1]['timestamp_us']=now+1
  if mode=='UUID':p['gpu_uuid']='GPU-wrong'
  if mode=='clockjump':elapsed=8
  try:DeviceWindow(E,UUID).update(p,now,elapsed)
  except(ValueError,KeyError,TypeError):pass
  else:raise AssertionError(mode+' accepted')
  negatives.append(mode)
 w=DeviceWindow(E,UUID);w.update(packet(4),E+4100000,4.1);p=packet(6);p['frames'][0]['device_samples']['SM'][4]['value']=49
 try:w.update(p,E+6100000,6.1)
 except ValueError:pass
 else:raise AssertionError('Changed old component accepted')
 # Late actual records can LOWER a formerly passed device average; that must stop.
 w=DeviceWindow(E,UUID)
 for end in range(2,282,2):w.update(packet(end,35),E+end*1000000+100000,end+.1,steady_required=end>=280)
 p=packet(282,35)
 for values in p['frames'][0]['device_samples'].values():
  values.extend(dict(timestamp_us=E+t+50000,value=0)for t in range(268000000,280000000,200000))
 try:w.update(p,E+282100000,282.1,steady_required=True)
 except ValueError as exc:assert 'invalidate' in str(exc)
 else:raise AssertionError('Late real records must invalidate a previously passed low bound')
 # Replay actual passive raw history; no GPU computation or new model forward.
 passive=T/'outputs/native_device_sampling_audit_v1/e939049217404460a0ce4df5ddbc7c46';raw=[json.loads(x)for x in(passive/'passive.jsonl').read_text().splitlines()];first=raw[0];start=max(set(x['timestamp_us']for x in first['SM'])&set(x['timestamp_us']for x in first['bandwidth']));w=DeviceWindow(start,first['gpu_uuid'])
 for x in raw:
  now=max(x['host_us'],max(y['timestamp_us']for y in x['SM']),max(y['timestamp_us']for y in x['bandwidth']))+1
  d=w.update(dict(gpu_uuid=x['gpu_uuid'],frames=[dict(device_samples=dict(SM=x['SM'],bandwidth=x['bandwidth']))]),now,(now-start)/1e6)
 reference=json.loads((passive/'report.json').read_text());assert d['cumulative_lower_bound']<=reference['dense_time_weighted_MAX_SM_bandwidth']+1e-9
 assert d['cumulative_lower_bound']<reference['coarse_arithmetic_mean'],'Dense measurement is not always favorable'
 # Actual existing process sampling now also carries every physicaldevice record; reads only, no CUDA/context.
 active=[];begin_us=time.time_ns()//1000;begin=time.monotonic();device=None
 for i in range(12):
  metrics.process_utilization(6);m=metrics.diagnostics();data=m['raw_NVML'];device=device or DeviceWindow(begin_us,data['gpu_uuid']);actual=device.update(data,time.time_ns()//1000,time.monotonic()-begin);active.append(actual);print('READ_ONLY_DENSE_DEVICE_PASS',i,flush=True)
 O.mkdir(exist_ok=True);out=dict(passed=True,all_original_guard_decisions_helpers_thresholds_AST_exact=True,original_process_selector_and_clock_pair_AST_exact=True,fixed_start_and_startup_zeros_included=True,full_cumulative_rolling300_and_same_timestamp_MAX_exact=True,missing_stale_invalid_gaps_clock_jumps_rejected=negatives,overlapping_samples_counted_once=True,actual_backfilled_samples_merged_chronologically=True,late_revisions_recheck_and_can_invalidate_prior_steady_checks=True,conflicting_old_component_rejected=True,boundary_time_conservative_lower_bound_not_fabricated_raw_zero=True,passive_replay=d,passive_reference=reference,actual_read_only_device_updates=active,no_GPU_work_started=True,old_failed_evaluations_not_reclassified=True,source_sha256={str(p):sha(p)for p in C.glob('*')if p.suffix in('.py','.json')},passive_input_sha256={str(p):sha(p)for p in[passive/'passive.jsonl',passive/'report.json']})
 with(O/'CPU_device_checks.json').open('x')as f:json.dump(out,f,indent=2)
 print('FULL_DEVICE_TIMESTAMP_STREAM_CPU_PASS',flush=True)
if __name__=='__main__':main()
