"""Original-body parity, actual fresh NVML scans and fail-closed deadline."""
import copy,importlib.util,json,os,statistics,sys,time,types
from pathlib import Path
from .worker import collector,SOURCES
from .transport import Session,validate,context
from native_policy_execution_v1.common import T,read,write,sha
def parity():
 class NVML:
  nvmlMemory_v2=2
  def __init__(self):self.calls=[]
  def nvmlInit(self):self.calls.append('init')
  def nvmlShutdown(self):self.calls.append('shutdown')
  def nvmlDeviceGetCount(self):self.calls.append('count');return 7
  def nvmlDeviceGetHandleByIndex(self,i):self.calls.append(('handle',i));return i
  def nvmlDeviceGetUUID(self,h):self.calls.append(('uuid',h));return 'GPU-fixture-'+str(h)
  def nvmlDeviceGetMemoryInfo(self,h,version):self.calls.append(('memory',h,version));return types.SimpleNamespace(used=4*2**30 if h==4 else 0,total=24*2**30)
  def nvmlDeviceGetUtilizationRates(self,h):self.calls.append(('util',h));return types.SimpleNamespace(gpu=60 if h==4 else 0,memory=30 if h==4 else 0)
  def nvmlDeviceGetComputeRunningProcesses(self,h):self.calls.append(('compute',h));return[types.SimpleNamespace(pid=os.getpid(),usedGpuMemory=4*2**30)]if h==4 else[]
  def nvmlDeviceGetGraphicsRunningProcesses(self,h):self.calls.append(('graphics',h));return[types.SimpleNamespace(pid=1,usedGpuMemory=None)]if h==0 else[]
 for kind,path in SOURCES.items():
  spec=importlib.util.spec_from_file_location('_original_account_'+kind,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  old=NVML();module.n=old;expected=module.collect()
  new=NVML();fn=collector(kind);fn.__globals__['n']=new;new.nvmlInit();actual=fn();new.nvmlShutdown()
  assert actual==expected and old.calls==new.calls;validate(actual,kind)
 return dict(passed=True,original_inventory_and_rule4_collection_bodies_exact=True,all_NVML_call_arguments_order_and_original_PID_ownership_logic_exact=True,only_session_initialization_lifetime_changes=True)
def main():
 o=T/'outputs/native_account_stream_v1';o.mkdir(parents=True,exist_ok=True);same=parity()
 session=Session(o/'CPU_account_actual_worker_exit.json');times=[]
 try:
  for i in range(24):
   kind='inventory'if i%2==0 else'rule4';start=time.monotonic();fleet=session.request(kind);times.append(time.monotonic()-start)
   assert len(fleet['cards'])==7
 finally:session.close()
 rows=[json.loads(x)for x in session.log.read_text().splitlines()]
 assert len(rows)==24 and len({r['worker_pid']for r in rows})==1 and[r['request']['sequence']for r in rows]==list(range(1,25))
 assert all(r['parent_requested_unix_ns']<=r['request_started_unix_ns']<=r['response_ready_unix_ns']<=r['parent_received_unix_ns']and r['elapsed_seconds']<3 and not r['previous_call_frames_used']for r in rows)
 hung=Session(o/'CPU_hung_account_worker_exit.json',command=[sys.executable,'-B','-c','import time; time.sleep(30)']);start=time.monotonic()
 try:hung.request('inventory');raise AssertionError('Hung account scan accepted')
 except TimeoutError:pass
 elapsed=time.monotonic()-start;assert 3<=elapsed<3.5 and hung.failed and hung.sequence==1
 try:hung.request('inventory');raise AssertionError('Failed account scan resumed')
 except RuntimeError:pass
 try:hung.close();raise AssertionError('Forced cleanup accepted as normal')
 except RuntimeError:pass
 assert hung.process.poll()is not None and not Path('/proc',str(hung.process.pid)).exists()
 from . import guard,admission
 implementation=guard._implementation();assert implementation.guarded_command.__code__.co_filename==str(T/'native_process_sampler_v13/strict_guard.py')
 assert admission.check_admission.__code__ is admission.original.check_admission.__code__
 from native_server_rule4_v4.admission import Monitor
 with context(o/'CPU_admission_account_actual_exit.json')as admit:
  admit.request('inventory');admit.request('rule4');monitor=Monitor()
  try:
   deadline=time.monotonic()+15
   while monitor.latest is None:
    assert monitor.failed is None and time.monotonic()<deadline;time.sleep(.05)
   assert len(monitor.latest['fleet']['cards'])==7
  finally:monitor.close()
 proof=read(monitor.root/'NVML_account_rule4_actual_exit.json');assert proof['exit_code']==0 and not proof['failed_request']and not proof['forced']and proof['actual_wait_return']and proof['proc_absent_after_wait']
 source={str(p):sha(p)for d in(Path(__file__).parent,T/'native_server_rule4_v4')for p in d.glob('*')if p.is_file()and p.suffix in('.py','.md')}
 write(o/'CPU_account_transport_checks.json',dict(passed=True,original_body_parity=same,actual_readonly_requests=24,first_request_seconds=times[0],persistent_median_seconds=statistics.median(times[1:]),request_sequence_nonce_fresh_time_checks=True,original_guard_only_inventory_import_changed_AST_verified=True,original_admission_function_code_identity=True,original_guard_source_sha256=sha(T/'native_process_sampler_v13/strict_guard.py'),hung_deadline_seconds=elapsed,failed_session_cannot_resume=True,rule4_actual_monitor_exit0=True,no_CUDA_or_GPU_work=True,source_sha256=source,input_sha256={str(f):sha(f)for f in[session.proof,session.log,o/'CPU_admission_account_actual_exit.json',monitor.root/'NVML_account_rule4_actual_exit.json',monitor.root/'NVML_account_rule4_actual_exit.requests.jsonl',monitor.root/'actual_monitor_exit.json']}))
 print('ORIGINAL_ACCOUNT_BODY_PARITY_AND_ACTUAL24_FRESH_SCANS_PASS',times[0],statistics.median(times[1:]),flush=True)
if __name__=='__main__':main()
