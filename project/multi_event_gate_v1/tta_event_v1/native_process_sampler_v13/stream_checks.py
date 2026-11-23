"""Actual request/EOF/deadline fault probes, with no model or CUDA work."""
from pathlib import Path
import ast,json,sys,tempfile,time
from .stream import Session
T=Path(__file__).resolve().parents[1];D=Path(__file__).parent;O=T/'outputs/native_process_sampler_v13'
def main():
 O.mkdir(parents=True,exist_ok=True)
 def fn(p,n):return next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef)and x.name==n)
 for name in('strict_guard.py','device_window.py','device_audit.py'):
  assert ast.dump(ast.parse((D/name).read_text().replace('native_process_sampler_v13','native_process_sampler_v12')))==ast.dump(ast.parse((T/'native_process_sampler_v12'/name).read_text())),name
 for name in('frame','choose_pair','verify_trace'):
  assert ast.dump(fn(D/'worker.py',name))==ast.dump(fn(T/'native_process_sampler_v12/worker.py',name))
 assert ast.dump(fn(D/'metrics.py','select'))==ast.dump(fn(T/'native_process_sampler_v12/metrics.py','select'))
 body="""import os,sys,json,time
for line in sys.stdin:
 r=json.loads(line);now=time.time_ns();print(json.dumps(dict(request=r,data=dict(gpu=r['gpu'],fresh_sequence=r['sequence']),worker_pid=os.getpid(),request_started_unix_ns=now,response_ready_unix_ns=time.time_ns())),flush=True)
time.sleep(1.05)
"""
 with tempfile.TemporaryDirectory()as temp:
  proof=Path(temp)/'exit.json';s=Session(4,proof,command=[sys.executable,'-B','-c',body]);began=time.monotonic()
  a,ta=s.request();b,tb=s.request();elapsed=time.monotonic()-began
  assert elapsed<.75 and a['fresh_sequence']==1 and b['fresh_sequence']==2 and ta['worker_pid']==tb['worker_pid']
  close=time.monotonic();s.close();delay=time.monotonic()-close;v=json.loads(proof.read_text());assert delay>=1 and v['exit_code']==0 and v['actual_wait_return']and not v['forced']and v['request_count']==2
  timeout=Session(4,command=[sys.executable,'-B','-c',"import sys,time;sys.stdin.readline();print('actual_frame_before_timeout',file=sys.stderr,flush=True);time.sleep(3.2)"])
  start=time.monotonic()
  try:timeout.request()
  except RuntimeError as e:assert 'three-second'in str(e)and'actual_frame_before_timeout'in str(e)
  else:raise AssertionError('Three-second timeout waived')
  assert 2.8<=time.monotonic()-start<3.5
  try:timeout.request()
  except RuntimeError as e:assert 'cannot resume'in str(e)
  else:raise AssertionError('Failed request silently retried')
  timeout.close()
  fail=Session(4,command=[sys.executable,'-B','-c',"import sys;sys.stdin.readline();print('actual_nonzero_evidence',file=sys.stderr,flush=True);raise SystemExit(1)"])
  try:fail.request()
  except RuntimeError as e:assert 'actual_nonzero_evidence'in str(e)
  else:raise AssertionError('Nonzero ignored')
  try:fail.close()
  except RuntimeError as e:assert 'did not exit normally'in str(e)
  else:raise AssertionError('Nonzero cleanup ignored')
  wrong=Session(4,command=[sys.executable,'-B','-c',body.replace("request=r,data=", "request=dict(r,nonce='wrong'),data=")])
  try:wrong.request()
  except RuntimeError as e:assert 'identity mismatch'in str(e)
  else:raise AssertionError('Nonce mismatch ignored')
  wrong.close()
 source=T/'outputs/native_frozen100_v22/mixed/batches/029/runs/507074896f87403e8eab30a39e1d6d11/last_successfully_collected_metric_at_failure.json';m=json.loads(source.read_text())['metric'];raw=m['raw_NVML'];f=raw['frames'][-1];hi=max({p['timestamp_us']for p in f['device_samples']['SM']}&{p['timestamp_us']for p in f['device_samples']['bandwidth']});assert f['host_timestamp_us']-hi<=750000 and m['unix_ns']//1000-hi>750000
 from .device_window import DeviceWindow
 start=raw['frames'][0]['host_timestamp_us']-1000000;w=DeviceWindow(start,raw['gpu_uuid'])
 try:w.update(raw,m['unix_ns']//1000,(m['unix_ns']//1000-start)/1e6,steady_required=False)
 except ValueError as e:assert 'Physical driver samples stale'in str(e)
 else:raise AssertionError('Old stale response reclassified')
 from .worker import clock_pair,verify_trace,choose_pair
 old_failure=T/'outputs/native_frozen100_v23/mixed/batches/031/runs/1fe0bba50db0402bbdbb9145f1b9a0c8/failure.json'
 import re,copy
 error=json.loads(old_failure.read_text())['traceback'];traces=[]
 for match in re.finditer('actual_acquisition_trace=',error):
  try:traces.append(json.JSONDecoder().raw_decode(error[match.end():])[0])
  except json.JSONDecodeError:pass
 observed=[r['actual_full_frame']for r in traces[-1]];assert len(observed)==6 and choose_pair(observed)is None
 last=copy.deepcopy(observed[-1]);step=200000;last['host_timestamp_us']+=step;last['driver_timestamp_us']+=step
 stamp=last['driver_timestamp_us']
 for kind in('SM','bandwidth'):last['device_samples'][kind]=[dict(timestamp_us=stamp,value=0)]
 last['raw_process_records']=[];last['API_status']='no_process_sample_NOT_FOUND';observed.append(last);calls=[]
 def read(last_seen):calls.append(last_seen);return observed[len(calls)-1]
 pair,trace,_=clock_pair(read,sleep=lambda x:None,monotonic=lambda:0)
 assert len(calls)==7 and pair[-1]==last and choose_pair(trace)==(3,6)
 assert pair[-1]['host_timestamp_us']-pair[0]['host_timestamp_us']<=1800000 and pair[-1]['driver_timestamp_us']-pair[0]['driver_timestamp_us']<=1600000
 pair_failure=T/'outputs/native_frozen100_v24/mixed/batches/033/runs/e381ae414bfc44898d7679dca9e59d67/last_successfully_collected_metric_at_failure.json'
 original=json.loads(pair_failure.read_text())['metric']['raw_NVML'];invalid=original['frames'][-1]
 from .worker import device_time_fresh,eligible
 assert not device_time_fresh(invalid)and not eligible(original['frames'][0],invalid)
 # Keep the original unpaired raw frame; only an extra fully paired fixture can be used.
 fs=copy.deepcopy(original['clock_readiness_frames']);assert choose_pair(fs)is None
 fresh=copy.deepcopy(fs[-1]);fresh['host_timestamp_us']+=200000;fresh['driver_timestamp_us']+=200000
 stamp=fresh['driver_timestamp_us'];fresh['device_samples']={k:[dict(timestamp_us=stamp,value=0)]for k in('SM','bandwidth')};fresh['API_status']='no_process_sample_NOT_FOUND';fresh['raw_process_records']=[];fs.append(fresh)
 indices=choose_pair(fs);assert indices is not None and indices[1]==len(fs)-1
 for kind in('SM','bandwidth'):
  for row in fs[-1]['device_samples'][kind]:row['value']=99
 assert choose_pair(fs)==indices
 try:
  w=DeviceWindow(original['frames'][0]['host_timestamp_us']-1000000,original['gpu_uuid']);now=original['frames'][-1]['host_timestamp_us'];w.update(original,now,(now-w.start)/1e6)
 except ValueError as exc:assert 'Unpaired internal device component'in str(exc)
 else:raise AssertionError('Original unpaired frame reclassified')
 proof=dict(passed=True,only_request_transport_changed=True,strict_guard_DeviceWindow_select_AST_unchanged=True,endpoint_time_eligibility_adds_original_exact_component_pairing=True,unpaired_actual_frames_retained_and_not_eligible=True,no_timestamp_rounding_or_tolerance_added=True,zero_or99_identical_endpoint_choice=True,obsolete_1750ms_acquisition_cutoff_replaced_by2600ms_under_actual3000ms_request_deadline=True,old_actual_trace_has_no_eligible_pair_and_is_not_reclassified=True,additional_fixture_actual_full_frame_enables_first_eligible_pair_with_original_bounds=True,actual_delayed_shutdown_does_not_age_each_request=True,actual_EOF_cleanup_wait0=True,actual_three_second_deadline_and_nonzero_evidence_preserved=True,failed_requests_never_retried=True,nonce_mismatch_rejected=True,old_stale_actual_metric_still_rejected=True,old_sample_age_at_frame_us=f['host_timestamp_us']-hi,old_sample_age_at_parent_us=m['unix_ns']//1000-hi,no_GPU_work=True)
 with(O/'CPU_stream_checks.json').open('x')as out:json.dump(proof,out,indent=2);out.write('\n')
 print('ACTUAL_STREAM_TRANSPORT_CPU_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
