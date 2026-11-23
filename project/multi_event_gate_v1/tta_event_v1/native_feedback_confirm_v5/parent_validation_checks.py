"""Actual CPU fork/file-IPC tests; no simulator, Torch or GPU."""
from pathlib import Path
import os,json,time,uuid,threading
from parent_validation import ParentValidator,request,atomic,read,wait_for_parent
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_feedback_confirm_v5/cpu_validator_checks'
O.mkdir(parents=True,exist_ok=True)
PROTO='c'*64;NONCE='d'*32
def run(label):
 root=O/(label+'_'+uuid.uuid4().hex[:12]);root.mkdir();ipc=root/'ipc'
 n=4 if label=='ordered_after_exit'else 2;ids=['case'+str(i)for i in range(n)]
 for i,cid in enumerate(ids):
  folder=root/('case%03d'%i);folder.mkdir();atomic(folder/'case.json',dict(case_id=cid))
 records=[];after_exit=[];holder={}
 def validate(i,f):
  if label=='validate_error':raise ValueError('Recorded CPU validation failure fixture')
  assert read(f/'payload.json')['complete']is True
  if label=='ordered_after_exit'and i>=2:
   assert holder['service'].terminal.wait(3),'CPU must finish this validation after child exit'
   after_exit.append(i)
  return dict(passed=True,compatibility_bridge_passed=not(label=='bridge_error'and i==0))
 def commit(i,f,result):
  assert os.getpid()==holder['parent']and threading.current_thread().name=='CPU-case-validator'
  records.append(i);return dict(case_id=ids[i],source='CPU software fixture')
 bridges=2 if label in('ordered_after_exit','bridge_error')else 0
 service=ParentValidator(root,ipc,ids,nonce=NONCE,protocol_identity=PROTO,bridge_count=bridges,validate=validate,commit=commit)
 holder.update(service=service,parent=os.getpid())
 assert len(list(Path('/proc/self/task').iterdir()))==1,'Fork must precede CPU validation thread'
 child=os.fork()
 if child==0:
  try:
   assert len(list(Path('/proc/self/task').iterdir()))==1
   wait_for_parent(ipc,nonce=NONCE,protocol_identity=PROTO,timeout=3)
   atomic(root/'child.json',dict(pid=os.getpid(),threads=1,parent=os.getppid()))
   def send(i,**change):
    atomic(root/('case%03d'%i)/'payload.json',dict(complete=True))
    kw=dict(nonce=NONCE,protocol_identity=PROTO,case_id=ids[i]);kw.update(change)
    request(ipc,i,'case%03d'%i,**kw)
   if label=='ordered_after_exit':
    send(1);assert not(ipc/'bridges_passed.json').exists();send(0)
    began=time.monotonic()
    while not(ipc/'bridges_passed.json').exists():
     if time.monotonic()-began>3:raise TimeoutError('Bridge gate did not open')
     time.sleep(.005)
    assert [read(ipc/'acks'/('%03d.json'%i))['index']for i in(0,1)]==[0,1]
    send(3);send(2)
   elif label=='bad_nonce':send(0,nonce='e'*32)
   elif label=='bad_folder':
    request(ipc,0,'../outside',nonce=NONCE,protocol_identity=PROTO,case_id=ids[0])
   elif label=='wrong_case_folder':
    atomic(root/'case001/payload.json',dict(complete=True))
    request(ipc,0,'case001',nonce=NONCE,protocol_identity=PROTO,case_id=ids[0])
   elif label=='partial_request':
    atomic(ipc/'requests/000.json.tmp',dict(index=0))
   elif label=='normal_missing':send(0)
   else:
    send(0);began=time.monotonic()
    while not(ipc/'validation_failure.json').exists():
     if time.monotonic()-began>3:raise TimeoutError('CPU failure did not cancel child')
     time.sleep(.005)
    os._exit(1)
   os._exit(0)
  except BaseException:os._exit(125)
 service.start_after_fork(child)
 atomic(ipc/'parent_validator_ready.json',dict(nonce=NONCE,protocol_identity=PROTO,child_pid=child))
 duplicate_refused=False
 try:service.start_after_fork(child)
 except ValueError:duplicate_refused=True
 assert duplicate_refused
 waited,status=os.waitpid(child,0);code=os.waitstatus_to_exitcode(status)
 assert waited==child
 service.child_exited(code)
 error=None
 try:summary=service.join(5)
 except BaseException as exc:error=str(exc);summary=read(ipc/'validation_summary.json')
 assert service.pidfd is None and not service.thread.is_alive()
 if label=='ordered_after_exit':
  assert code==0 and error is None and records==list(range(n))and after_exit==[2,3]
  assert summary['all_cases_validated']and not summary['failed']
 else:
  assert error is not None and summary['failed']and not summary['all_cases_validated']
  assert records==([0]if label=='normal_missing'else [])
  if bridges:assert not(ipc/'bridges_passed.json').exists()
  if label in('bridge_error','validate_error'):assert code!=0
 return dict(label=label,passed=True,child_exit=code,committed=records,CPU_commits_after_child_exit=after_exit,
     parent_thread_started_only_after_real_fork=True,pidfd_cancellation_available=True,
     malformed_or_missing_requests_rejected=error is not None,root=str(root))
def main():
 rows=[run(label)for label in('ordered_after_exit','bad_nonce','bad_folder','wrong_case_folder','partial_request','normal_missing','bridge_error','validate_error')]
 v=dict(passed=True,scenarios=len(rows),rows=rows,scope='Real CPU forks and file IPC only. No Torch/CUDA/model/native simulation/performance or TTA claim.')
 import hashlib
 v['source_sha256']={str(C/n):hashlib.sha256((C/n).read_bytes()).hexdigest()for n in('parent_validation.py','parent_validation_checks.py')}
 p=O/('report_'+uuid.uuid4().hex[:12]+'.json');atomic(p,v)
 atomic(O.parent/'parent_validator_checks.json',v)
 print('PARENT_VALIDATOR_CPU_FORK_CHECKS',len(rows),str(p),flush=True)
if __name__=='__main__':main()
