"""Actual original/ordered ABBA requests, full raw NVML retained."""
from pathlib import Path
import subprocess,sys,json,time,select,re
from native_policy_execution_v1.common import T,write,read,sha
from native_onpolicy_recovery_v3.common import mapped
from native_process_sampler_v13 import worker,metrics
from .order import frame_function

def main():
 out=T/'outputs/native_nvml_query_order_v3';out.mkdir(parents=True,exist_ok=True);records=[]
 for index,mode in enumerate(('original','ordered','ordered','original')):
  process=subprocess.Popen([sys.executable,'-B','-u','-m','native_nvml_query_order_v3.probe',mode],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  assert process.stdout.readline().strip()=='CPU_READY';began=time.monotonic();process.stdin.write('run\n');process.stdin.flush()
  assert select.select([process.stdout],[],[],3)[0],'Original hard3second metadata deadline exceeded'
  value=json.loads(process.stdout.readline());elapsed=time.monotonic()-began;assert elapsed<=3
  code=process.wait(timeout=3);assert code==0;value.update(actual_Popen_wait_return=True,exit_code=code,pid=process.pid,proc_absent_after_wait=not Path('/proc',str(process.pid)).exists(),actual_parent_request_seconds=elapsed)
  if value['passed']:worker.verify_trace(value['data']);metrics.select(value['data'],0)
  write(out/('CPU_actual_NVML_'+str(index)+'_'+mode+'.json'),value);records.append(value)
 assert all(r['passed']for r in records if r['mode']=='ordered'),[(r['mode'],r['passed'],r['seconds'])for r in records]
 assert any(not r['passed']for r in records if r['mode']=='original'),'Controlled old ordering failure was not reproduced'
 # Original selection and accounting source/functions are never modified.
 new=frame_function();assert new is not worker.frame
 from .context import context,OrderedSession
 from native_process_sampler_v13 import stream
 before=stream.Session;one=metrics.one_shot_process_utilization
 with context():assert stream.Session is OrderedSession and metrics.one_shot_process_utilization is not one
 assert stream.Session is before and metrics.one_shot_process_utilization is one
 stage=T/'outputs/native_onpolicy_arm_collection_v8/val_000';root=mapped(read(stage/'execution_state.json')['root']);failure=root/'failure.json';text=read(failure)['traceback'];match=re.search(r'actual_acquisition_trace=(\[)',text);trace,_=json.JSONDecoder().raw_decode(text[match.start(1):]);frames=[r['actual_full_frame']for r in trace if'actual_full_frame'in r]
 assert len(frames)==7 and any(f['host_timestamp_us']-f['driver_timestamp_us']>.75e6 for f in frames)
 assert any(r.get('wall_seconds',0)>1.2 for r in trace)and 'No actual eligible endpoint pair' in text
 from native_nvml_query_order_v2 import order as prior
 import inspect
 before=inspect.getsource(prior.clock_pair);after=inspect.getsource(__import__('native_nvml_query_order_v3.order',fromlist=['clock_pair']).clock_pair)
 assert after.replace("sleep(.025 if first is None or age>.75e6 or not original.device_time_fresh(value) else .2)",'sleep(.025 if first is None else .2)')==before
 files=list(out.glob('CPU_actual_NVML_*.json'))+[failure,stage/'actual_host_wait_exit.json']
 write(out/'CPU_actual_query_order_checks_v1.json',dict(passed=True,actual_ABBA=[dict(mode=r['mode'],passed=r['passed'],seconds=r['seconds'])for r in records],actual_ARM8_stale_final_frame_sleep_consumed_usable_pair_window=True,old_failed_GPU_still_excluded=True,all_original_time_bounds_and_3second_deadline_unchanged=True,no_GPU_or_scientific_qualification=True,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()},input_sha256={str(f):sha(f)for f in files}))
 print('ACTUAL_NVML_LATE_STALE_FRAME_FAST_REACQUISITION_CPU_PASS',[(r['mode'],r['passed'],round(r['seconds'],3))for r in records],flush=True)
if __name__=='__main__':main()
