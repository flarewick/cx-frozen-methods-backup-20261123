"""CPU-parent validation after GPU fork; ordered, nonce-bound file IPC."""
from pathlib import Path
import json,os,signal,threading,time
def read(p):return json.loads(Path(p).read_text())
def atomic(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if read(p)!=value:raise ValueError('Conflicting immutable IPC record: '+str(p))
  return
 tmp=p.with_name(p.name+'.tmp')
 with tmp.open('x')as f:json.dump(value,f,allow_nan=False,indent=2);f.write('\n')
 tmp.replace(p)
def request(ipc,index,folder,*,nonce,protocol_identity,case_id):
 if type(index)is not int or index<0:raise ValueError('Nonnegative case index required')
 atomic(Path(ipc)/'requests'/('%03d.json'%index),dict(index=index,folder=str(folder),
  nonce=nonce,protocol_identity=protocol_identity,child_pid=os.getpid(),case_id=case_id))
def wait_for_parent(ipc,*,nonce,protocol_identity,timeout=30):
 p=Path(ipc)/'parent_validator_ready.json';began=time.monotonic()
 while not p.exists():
  if time.monotonic()-began>timeout:raise TimeoutError('CPU parent validator not ready')
  time.sleep(.01)
 value=read(p)
 if(value.get('nonce')!=nonce or value.get('protocol_identity')!=protocol_identity
     or value.get('child_pid')!=os.getpid()):raise ValueError('CPU parent ready identity mismatch')
 return value
class ParentValidator:
 def __init__(self,dataset,ipc,case_ids,*,nonce,protocol_identity,bridge_count,validate,commit,assert_cpu=lambda:None):
  self.dataset=Path(dataset).resolve();self.ipc=Path(ipc).resolve()
  if not self.ipc.is_relative_to(self.dataset):raise ValueError('IPC outside dataset')
  self.ids=tuple(case_ids);self.nonce=nonce;self.protocol=protocol_identity;self.bridges=bridge_count
  if not 0<=bridge_count<=len(self.ids):raise ValueError('Invalid bridge count')
  self.validate=validate;self.commit=commit;self.assert_cpu=assert_cpu
  self.parent_pid=os.getpid();self.child_pid=None;self.thread=None;self.pidfd=None
  self.terminal=threading.Event();self.finished=threading.Event();self.failed=threading.Event()
  self.child_code=None;self.completed=0;self.error=None
 def start_after_fork(self,child_pid):
  if os.getpid()!=self.parent_pid or self.thread is not None:raise ValueError('CPU validator starts once in original parent')
  if type(child_pid)is not int or child_pid<=0 or child_pid==self.parent_pid:raise ValueError('Actual fork child required')
  status=Path('/proc',str(child_pid),'status').read_text()
  ppid=int(next(s.split()[1]for s in status.splitlines()if s.startswith('PPid:')))
  if ppid!=self.parent_pid:raise ValueError('Validator child is not our fork child')
  self.child_pid=child_pid;self.pidfd=os.pidfd_open(child_pid);self.assert_cpu()
  self.thread=threading.Thread(target=self._run,name='CPU-case-validator',daemon=False);self.thread.start()
 def child_exited(self,code):
  if type(code)is not int:raise ValueError('Actual waitpid exit code required')
  self.child_code=code;self.terminal.set()
 def _decode(self,p,index):
  value=read(p)
  expected={'index','folder','nonce','protocol_identity','child_pid','case_id'}
  if set(value)!=expected:raise ValueError('Unexpected request fields')
  if(value['index']!=index or value['nonce']!=self.nonce or value['protocol_identity']!=self.protocol
     or value['child_pid']!=self.child_pid or value['case_id']!=self.ids[index]):
   raise ValueError('Request identity/index mismatch')
  if Path(value['folder']).is_absolute():raise ValueError('Relative case folder required')
  folder=(self.dataset/value['folder']).resolve()
  if not folder.is_relative_to(self.dataset)or folder==self.dataset or not folder.is_dir():
   raise ValueError('Case folder outside dataset or missing')
  if read(folder/'case.json').get('case_id')!=self.ids[index]:raise ValueError('Folder case identity mismatch')
  return folder
 def cancel_child(self):
  if self.pidfd is not None and not self.terminal.is_set():
   try:signal.pidfd_send_signal(self.pidfd,signal.SIGTERM)
   except ProcessLookupError:pass
 def _run(self):
  try:
   if self.bridges==0:
    atomic(self.ipc/'bridges_passed.json',dict(nonce=self.nonce,protocol_identity=self.protocol,bridges=0))
   while self.completed<len(self.ids):
    i=self.completed;p=self.ipc/'requests'/('%03d.json'%i)
    if not p.exists():
     if self.terminal.is_set():
      if self.child_code==0:raise ValueError('Normal child exited without complete request '+str(i))
      break
     self.terminal.wait(.025);continue
    self.assert_cpu();folder=self._decode(p,i)
    result=self.validate(i,folder)
    if not isinstance(result,dict)or result.get('passed')is not True:raise ValueError('Case validation failed')
    if i<self.bridges and result.get('compatibility_bridge_passed')is not True:
     raise ValueError('Compatibility bridge failed before commit')
    record=self.commit(i,folder,result)
    self.assert_cpu()
    atomic(self.ipc/'acks'/('%03d.json'%i),dict(index=i,case_id=self.ids[i],nonce=self.nonce,
      protocol_identity=self.protocol,record=record,validated=True))
    self.completed+=1
    if self.completed==self.bridges:
     atomic(self.ipc/'bridges_passed.json',dict(nonce=self.nonce,protocol_identity=self.protocol,bridges=self.bridges))
  except BaseException as exc:
   self.error=exc;self.failed.set();self.cancel_child()
   atomic(self.ipc/'validation_failure.json',dict(index=self.completed,nonce=self.nonce,
      protocol_identity=self.protocol,type=type(exc).__name__,reason=str(exc)))
  finally:
   try:
    atomic(self.ipc/'validation_summary.json',dict(completed=self.completed,total=len(self.ids),
      failed=self.failed.is_set(),all_cases_validated=self.completed==len(self.ids),
      nonce=self.nonce,protocol_identity=self.protocol,parent_pid=self.parent_pid,
      actual_child_pid=self.child_pid))
   except BaseException as exc:
    if self.error is None:self.error=exc
    self.failed.set();self.cancel_child()
   finally:self.finished.set()
 def join(self,timeout=600):
  if self.thread is None:raise ValueError('Validator never started')
  self.thread.join(timeout)
  if self.thread.is_alive():raise TimeoutError('CPU validation did not finish')
  try:
   if self.error is not None:raise self.error
   return read(self.ipc/'validation_summary.json')
  finally:
   if self.pidfd is not None:os.close(self.pidfd);self.pidfd=None
