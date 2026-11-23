"""Killable request/response metadata transport; no polling between requests."""
import json,os,select,subprocess,sys,tempfile,time,uuid
from contextlib import contextmanager
from pathlib import Path
from native_process_sampler_v13.nvml_query import form,render

class Session:
 def __init__(self,proof,command=None):
  self.proof=Path(proof);self.command=command;self.process=None;self.err=None
  self.nonce=uuid.uuid4().hex;self.sequence=0;self.failed=False;self.closed=False;self.started_ticks=None
 def request(self,arguments):
  form(arguments)
  if self.closed or self.failed:raise RuntimeError('Closed/failed metadata session cannot resume')
  began=time.monotonic();requested_ns=time.time_ns();self.sequence+=1
  try:
   if self.process is None:
    self.err=tempfile.TemporaryFile()
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]),PYTHONDONTWRITEBYTECODE='1')
    self.process=subprocess.Popen(self.command or [sys.executable,'-B','-m','native_metadata_stream_v1.worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,env=env,start_new_session=True)
    self.started_ticks=int(Path('/proc/%d/stat'%self.process.pid).read_text().split(') ',1)[1].split()[19])
   request=dict(nonce=self.nonce,sequence=self.sequence,arguments=arguments)
   self.process.stdin.write((json.dumps(request)+'\n').encode());self.process.stdin.flush()
   payload=b''
   while True:
    remaining=3-(time.monotonic()-began)
    if remaining<=0:raise TimeoutError('Metadata three-second request deadline exceeded')
    ready,_,_=select.select([self.process.stdout],[],[],remaining)
    if not ready:raise TimeoutError('Metadata three-second request deadline exceeded')
    chunk=os.read(self.process.stdout.fileno(),65536)
    if not chunk:raise RuntimeError('Metadata worker EOF: '+str(self.process.poll()))
    payload+=chunk
    if len(payload)>16*2**20:raise ValueError('Metadata response exceeds bounded payload')
    if b'\n' in chunk:
     if not payload.endswith(b'\n') or payload.count(b'\n')!=1:raise ValueError('Unsolicited metadata responses')
     response=json.loads(payload);break
   assert response['request']==request and response['worker_pid']==self.process.pid
   assert requested_ns<=response['request_started_unix_ns']<=response['data']['unix_ns']<=response['response_ready_unix_ns']<=time.time_ns()
   # Full original validation/render is included in the unchanged deadline.
   rendered=render(response['data'],arguments)
   if time.monotonic()-began>3:raise TimeoutError('Metadata parsing exceeded three-second request deadline')
   return rendered,response['data'],dict(worker_pid=self.process.pid,worker_started_ticks=self.started_ticks,nonce=self.nonce,request_sequence=self.sequence,request_started_unix_ns=response['request_started_unix_ns'],response_ready_unix_ns=response['response_ready_unix_ns'],parent_received_unix_ns=time.time_ns(),previous_call_frames_used=False)
  except BaseException:self.failed=True;raise
 def close(self):
  if self.closed:return
  self.closed=True
  if self.process is None:return
  forced=False
  try:
   self.process.stdin.close();code=self.process.wait(timeout=3)
  except subprocess.TimeoutExpired:
   forced=True;self.process.terminate()
   try:code=self.process.wait(timeout=3)
   except subprocess.TimeoutExpired:self.process.kill();code=self.process.wait(timeout=3)
  note=dict(exit_code=code,actual_wait_return=True,proc_absent_after_wait=not Path('/proc/%d'%self.process.pid).exists(),forced=forced,worker_pid=self.process.pid,worker_started_ticks=self.started_ticks,nonce=self.nonce,request_count=self.sequence,failed_request=self.failed,no_CUDA_or_GPU_work=True)
  self.proof.parent.mkdir(parents=True,exist_ok=True)
  with self.proof.open('x')as f:json.dump(note,f,indent=2);f.write('\n')
  self.process.stdout.close();self.err.close()
  if forced or code!=0 or not note['proc_absent_after_wait']:raise RuntimeError('Metadata worker failed actual normal termination: '+json.dumps(note))

class DoctorProxy:
 def __init__(self,original):self.original=original;self.report=None;self.session=None;self.scope=None;self.context_count=0
 def __getattr__(self,name):return getattr(self.original,name)
 @contextmanager
 def context(self,scope):
  if scope not in ('monitor','release'):raise ValueError('Explicit metadata lifecycle scope required')
  previous=self.session;previous_scope=self.scope;self.context_count+=1
  session=Session(self.report.with_name('NVML_metadata_%03d_%s_actual_exit.json'%(self.context_count,scope)))
  self.session=session;self.scope=scope
  try:yield session
  finally:
   self.session=previous;self.scope=previous_scope;session.close()
 def query(self,arguments):
  if self.session is None:raise RuntimeError('Metadata lifecycle context required')
  began=time.monotonic()
  try:rendered,raw,transport=self.session.request(arguments)
  except BaseException as exc:
   self.record(dict(arguments=arguments,error=type(exc).__name__,timeout_seconds=3,elapsed_seconds=time.monotonic()-began,automatic_retry=False,CLI_used=False));raise
  self.record(dict(arguments=arguments,raw_NVML=raw,transport=transport,lifecycle_scope=self.scope,elapsed_seconds=time.monotonic()-began,timeout_seconds=3,automatic_retry=False,CLI_used=False))
  return rendered
 def record(self,row):
  if self.report is not None:
   with self.report.open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
