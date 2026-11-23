"""Original one-request metadata check before any scientific CUDA admission."""
from pathlib import Path
import time,uuid
from native_policy_execution_v1.common import write,read,sha
from native_process_sampler_v13.stream import Session
from native_process_sampler_v13.worker import verify_trace
def check(index,expected_uuid,output):
 root=Path(output)/uuid.uuid4().hex;root.mkdir(parents=True);client=Session(index,proof=root/'actual_worker_exit.json')
 result=dict(gpu=index,expected_uuid=expected_uuid,passed=False,metadata_only=True,automatic_retry=False,request_count=1,started_unix_ns=time.time_ns())
 try:
  try:data,transport=client.request();write(root/'raw_NVML.json',data);write(root/'transport.json',transport)
  finally:client.close()
  exit=read(root/'actual_worker_exit.json');assert exit['exit_code']==0 and exit['actual_wait_return']and exit['proc_absent_after_wait']and not exit['failed_request']and not exit['forced']and exit['request_count']==1
  assert data['gpu_uuid']==expected_uuid and verify_trace(data)
  result.update(passed=True,input_sha256={str(p):sha(p)for p in(root/'raw_NVML.json',root/'transport.json',root/'actual_worker_exit.json')})
 except BaseException:
  import traceback
  write(root/'failure.json',dict(traceback=traceback.format_exc()))
 result['finished_unix_ns']=time.time_ns();write(root/'result.json',result);return root/'result.json',result
