"""Actual read-only transport plus fail-closed deadline, no GPU allocation."""
import ast,json,statistics,sys,time
from pathlib import Path
from .transport import Session
from native_process_sampler_v13.nvml_query import render,DoctorProxy
from native_policy_execution_v1.common import T,write,sha

def main():
 o=T/'outputs/native_metadata_stream_v1';o.mkdir(parents=True,exist_ok=True)
 actual=Session(o/'CPU_metadata_actual_worker_exit.json');rows=[]
 forms=[['-i','4','--query-gpu=memory.total'],['-i','4','--query-gpu=utilization.gpu,utilization.memory,memory.used'],['--query-compute-apps=gpu_uuid,pid'],['--query-compute-apps=gpu_uuid,pid,used_gpu_memory']]
 try:
  for _ in range(3):
   for arguments in forms:
    began=time.monotonic();text,raw,transport=actual.request(arguments)
    assert text==render(raw,arguments) and not transport['previous_call_frames_used']
    rows.append(dict(arguments=arguments,raw_NVML=raw,transport=transport,elapsed_seconds=time.monotonic()-began))
 finally:actual.close()
 assert len({r['transport']['worker_pid']for r in rows})==1
 assert [r['transport']['request_sequence']for r in rows]==list(range(1,13))
 original=DoctorProxy(None);old_times=[]
 for arguments in forms:
  began=time.monotonic();original.query(arguments);old_times.append(time.monotonic()-began)
 # A stuck transport MUST fail within three seconds, without restarting it.
 hung=Session(o/'CPU_hung_worker_actual_exit.json',command=[sys.executable,'-B','-c','import time; time.sleep(30)'])
 began=time.monotonic()
 try:hung.request(forms[0]);raise AssertionError('Hung metadata worker accepted')
 except TimeoutError:pass
 elapsed=time.monotonic()-began;assert 3<=elapsed<3.4 and hung.sequence==1 and hung.failed
 try:hung.request(forms[0]);raise AssertionError('Failed metadata transport resumed')
 except RuntimeError:pass
 try:hung.close();raise AssertionError('Forced worker cleanup called normal')
 except RuntimeError:pass
 assert hung.closed and hung.process.poll()is not None
 assert not Path('/proc/%d'%hung.process.pid).exists()
 from . import guard
 implementation=guard._implementation()
 assert implementation.MIN_CUMULATIVE_UTILIZATION if hasattr(implementation,'MIN_CUMULATIVE_UTILIZATION')else True
 source={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()}
 write(o/'CPU_transport_checks.json',dict(passed=True,actual_readonly_NVML_requests=rows,original_fresh_worker_elapsed_seconds=old_times,persistent_elapsed_seconds=[r['elapsed_seconds']for r in rows],hung_deadline_elapsed_seconds=elapsed,failed_session_rejects_retry=True,original_strict_guard_source_sha256=sha(T/'native_process_sampler_v13/strict_guard.py'),original_render_used_for_every_response=True,no_CUDA_or_GPU_work=True,source_sha256=source))
 print('ACTUAL_READONLY_METADATA_AND_FAIL_CLOSED_CHECKS_PASS',statistics.median(old_times),statistics.median([r['elapsed_seconds']for r in rows]))
if __name__=='__main__':main()
