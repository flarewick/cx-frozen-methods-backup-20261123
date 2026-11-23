"""Fresh request/response account scans; unchanged total three-second bound."""
import json,os,select,subprocess,sys,tempfile,threading,time,uuid
from pathlib import Path
from contextlib import contextmanager
from native_fast_inventory_v1.inventory import validate as validate_inventory
def validate(data,kind):
 if kind=='inventory':return validate_inventory(data)
 assert kind=='rule4'and data['read_only']is True and len(data['cards'])==7
 cards=data['cards'];assert [r['index']for r in cards]==list(range(7))and len({r['uuid']for r in cards})==7
 for r in cards:
  assert r['CUDA_context_created']is False and isinstance(r['uuid'],str)and r['uuid'].startswith('GPU-')
  assert set(r['own_pids'])<=set(r['pids'])and all(type(p)is int and p>0 for p in r['pids'])
  assert type(r['memory_known'])is bool and type(r['total_bytes'])is int and r['total_bytes']>4096*2**20
  if r['memory_known']:
   assert type(r['own_memory_bytes'])is int and 0<=r['own_memory_bytes']<=r['total_bytes']
   assert r['own_fraction']==r['own_memory_bytes']/r['total_bytes']
  else:assert r['own_memory_bytes']is None and r['own_fraction']is None
  assert r['attributable_utilization']is None or(type(r['attributable_utilization'])is int and 0<=r['attributable_utilization']<=100)
 assert data['own_gpu_count']==sum(bool(r['own_pids'])for r in cards)
 return data
class Session:
 def __init__(self,proof,command=None):
  self.proof=Path(proof);self.log=self.proof.with_suffix('.requests.jsonl');self.command=command
  self.process=None;self.err=None;self.sequence=0;self.nonce=uuid.uuid4().hex
  self.failed=False;self.closed=False;self.owner=threading.get_ident();self.started_ticks=None
 def request(self,kind):
  assert kind in('inventory','rule4')and threading.get_ident()==self.owner
  if self.failed or self.closed:raise RuntimeError('Failed/closed account session cannot resume')
  began=time.monotonic();requested_ns=time.time_ns();self.sequence+=1;request=dict(nonce=self.nonce,sequence=self.sequence,kind=kind)
  try:
   if self.process is None:
    self.err=tempfile.TemporaryFile()
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    self.process=subprocess.Popen(self.command or[sys.executable,'-B','-m','native_account_stream_v1.worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,env=env,start_new_session=True)
    self.started_ticks=int(Path('/proc',str(self.process.pid),'stat').read_text().split(') ',1)[1].split()[19])
   self.process.stdin.write((json.dumps(request)+'\n').encode());self.process.stdin.flush();payload=b''
   while True:
    remaining=3-(time.monotonic()-began)
    if remaining<=0 or not select.select([self.process.stdout],[],[],max(0,remaining))[0]:raise TimeoutError('Account scan three-second deadline exceeded')
    chunk=os.read(self.process.stdout.fileno(),65536)
    if not chunk:raise RuntimeError('Account scan worker EOF: '+str(self.process.poll()))
    payload+=chunk
    if len(payload)>16*2**20:raise ValueError('Account response exceeds bounded payload')
    if b'\n'in chunk:
     if not payload.endswith(b'\n')or payload.count(b'\n')!=1:raise ValueError('Unsolicited account response')
     response=json.loads(payload);break
   assert response['request']==request and response['worker_pid']==self.process.pid
   assert requested_ns<=response['request_started_unix_ns']<=response['response_ready_unix_ns']<=time.time_ns()
   data=validate(response['data'],kind)
   if time.monotonic()-began>3:raise TimeoutError('Account scan validation exceeded three seconds')
   self.record(dict(**response,parent_requested_unix_ns=requested_ns,parent_received_unix_ns=time.time_ns(),elapsed_seconds=time.monotonic()-began,timeout_seconds=3,automatic_retry=False,CLI_used=False,previous_call_frames_used=False))
   return data
  except BaseException as e:
   self.failed=True;self.record(dict(request=request,error=type(e).__name__,timeout_seconds=3,elapsed_seconds=time.monotonic()-began,automatic_retry=False,CLI_used=False));raise
 def record(self,row):
  self.log.parent.mkdir(parents=True,exist_ok=True)
  with self.log.open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 def close(self):
  if self.closed:return
  self.closed=True
  if self.process is None:return
  forced=False
  try:self.process.stdin.close();code=self.process.wait(timeout=3)
  except subprocess.TimeoutExpired:
   forced=True;self.process.terminate()
   try:code=self.process.wait(timeout=3)
   except subprocess.TimeoutExpired:self.process.kill();code=self.process.wait(timeout=3)
  note=dict(exit_code=code,actual_wait_return=True,proc_absent_after_wait=not Path('/proc',str(self.process.pid)).exists(),worker_pid=self.process.pid,worker_started_ticks=self.started_ticks,request_count=self.sequence,nonce=self.nonce,forced=forced,failed_request=self.failed,no_CUDA_or_GPU_work=True)
  self.proof.parent.mkdir(parents=True,exist_ok=True)
  with self.proof.open('x')as f:json.dump(note,f,indent=2);f.write('\n')
  self.process.stdout.close();self.err.close()
  if forced or code!=0 or not note['proc_absent_after_wait']:raise RuntimeError('Account worker failed normal exit: '+json.dumps(note))
_local=threading.local()
@contextmanager
def context(proof):
 previous=getattr(_local,'session',None);session=Session(proof);_local.session=session
 try:yield session
 finally:_local.session=previous;session.close()
def inventory():
 session=getattr(_local,'session',None)
 if session is None:raise RuntimeError('Explicit fresh account scan lifecycle required')
 return session.request('inventory')
def snapshot():
 session=getattr(_local,'session',None)
 if session is None:raise RuntimeError('Explicit fresh rule4 scan lifecycle required')
 return session.request('rule4')
