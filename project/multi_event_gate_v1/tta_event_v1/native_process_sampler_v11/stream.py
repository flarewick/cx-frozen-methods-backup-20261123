"""Bounded fresh NVML request transport, with actual normal worker termination."""
import json,os,select,subprocess,sys,tempfile,time,uuid
from pathlib import Path


def serve(index):
 from . import worker
 worker.TRACE_TO_STDERR=True
 n=worker.n;n.nvmlInit()
 try:
  handle=n.nvmlDeviceGetHandleByIndex(index)
  for line in sys.stdin:
   request=json.loads(line)
   if set(request)!={'nonce','sequence','gpu'}or request['gpu']!=index:raise ValueError('Invalid sampling request')
   began=time.time_ns();data=worker.collect(index,handle=handle)
   response=dict(request=request,data=data,worker_pid=os.getpid(),request_started_unix_ns=began,response_ready_unix_ns=time.time_ns())
   print(json.dumps(response,allow_nan=False,separators=(',',':')),flush=True)
 finally:n.nvmlShutdown()


class Session:
 def __init__(self,index,proof=None,command=None,deadline=3):
  if type(index)is not int or not 0<=index<=6:raise ValueError('Physical GPU index required')
  if deadline!=3:raise ValueError('Original three-second deadline required')
  self.index=index;self.proof=Path(proof)if proof else None;self.command=command;self.deadline=deadline
  self.nonce=uuid.uuid4().hex;self.sequence=0;self.process=None;self.err=None;self.started_ticks=None;self.failed=False;self.closed=False

 def stderr(self):
  if self.err is None:return ''
  self.err.flush();self.err.seek(0);return self.err.read().decode('utf-8','replace')

 def request(self):
  if self.closed or self.failed:raise RuntimeError('Failed or closed sampling session cannot resume')
  began=time.monotonic();self.sequence+=1
  if self.process is None:
   self.err=tempfile.TemporaryFile()
   command=self.command or[sys.executable,'-B','-m','native_process_sampler_v11.worker',str(self.index),'--serve']
   env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]),PYTHONDONTWRITEBYTECODE='1')
   self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,env=env,start_new_session=True)
   self.started_ticks=int(Path('/proc/%d/stat'%self.process.pid).read_text().split(') ',1)[1].split()[19])
  else:self.err.seek(0);self.err.truncate()
  request=dict(nonce=self.nonce,sequence=self.sequence,gpu=self.index)
  try:
   self.process.stdin.write((json.dumps(request)+'\n').encode());self.process.stdin.flush()
   chunks=[];length=0
   while True:
    remaining=self.deadline-(time.monotonic()-began)
    if remaining<=0:raise TimeoutError('Actual NVML three-second request deadline exceeded')
    ready,_,_=select.select([self.process.stdout],[],[],remaining)
    if not ready:raise TimeoutError('Actual NVML three-second request deadline exceeded')
    chunk=os.read(self.process.stdout.fileno(),65536)
    if not chunk:raise RuntimeError('Actual NVML stream EOF; exit='+str(self.process.poll()))
    chunks.append(chunk);length+=len(chunk)
    if length>16*2**20:raise ValueError('Sampling response exceeds bounded payload')
    if b'\n'in chunk:
     payload=b''.join(chunks)
     if not payload.endswith(b'\n')or payload.count(b'\n')!=1:raise ValueError('Unsolicited or multiple sample responses')
     response=json.loads(payload);break
   if time.monotonic()-began>self.deadline:raise TimeoutError('Actual NVML request parsing exceeded three-second deadline')
   if response['request']!=request or response['worker_pid']!=self.process.pid:raise ValueError('Sampling response identity mismatch')
   data=response['data']
   if data['gpu']!=self.index:raise ValueError('Sampling response GPU mismatch')
   return data,dict(kind='persistent_readonly_request_worker',worker_pid=self.process.pid,worker_started_ticks=self.started_ticks,nonce=self.nonce,request_sequence=self.sequence,request_deadline_seconds=3,request_started_unix_ns=response['request_started_unix_ns'],response_ready_unix_ns=response['response_ready_unix_ns'],parent_received_unix_ns=time.time_ns(),previous_call_frames_used=False)
  except BaseException as exc:
   self.failed=True
   raise RuntimeError(str(exc)+'; actual_stream_stderr='+self.stderr())from exc

 def close(self):
  if self.closed:return
  self.closed=True;forced=False;code=None
  if self.process is not None:
   try:self.process.stdin.close();code=self.process.wait(timeout=3)
   except subprocess.TimeoutExpired:
    forced=True;self.process.terminate()
    try:code=self.process.wait(timeout=3)
    except subprocess.TimeoutExpired:self.process.kill();code=self.process.wait(timeout=3)
   note=dict(exit_code=code,actual_wait_return=True,proc_absent_after_wait=not Path('/proc/%d'%self.process.pid).exists(),forced=forced,worker_pid=self.process.pid,worker_started_ticks=self.started_ticks,nonce=self.nonce,request_count=self.sequence,failed_request=self.failed,no_CUDA_or_GPU_work=True)
   if self.proof:
    with self.proof.open('x')as f:json.dump(note,f,indent=2);f.write('\n')
   self.process.stdout.close();evidence=self.stderr();self.err.close()
   if forced or code!=0 or not note['proc_absent_after_wait']:raise RuntimeError('NVML metadata worker did not exit normally: '+json.dumps(note)+'; stderr='+evidence)
