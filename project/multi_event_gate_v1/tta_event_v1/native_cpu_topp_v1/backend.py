"""Parallelize the unchanged pure-CPU MPLIB TOPP; never fork after CUDA."""
import ast,textwrap,ctypes,inspect,json,os,pickle,queue,random,signal,socket,sys,threading,time,traceback,types
from pathlib import Path
from multiprocessing.connection import Connection
D=Path(__file__).resolve().parents[3]/'deployment'
import hashlib
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT=Path(__file__).resolve().parents[1]/'outputs/native_cpu_topp_v1'
PROOF=OUT/'TOPP_pool_CPU_v52.json'
_INSTALLED=False
POOL=None
ORIGINAL=None
class SolverInfrastructureError(BaseException):pass

def original_function():
 import mplib
 fn=mplib.Planner.TOPP
 if getattr(fn,'_offloaded_TOPP_v52',False):return ORIGINAL
 node=ast.parse(textwrap.dedent(inspect.getsource(fn)));attributes={n.attr for n in ast.walk(node)if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='self'}
 if attributes!={'joint_vel_limits','joint_acc_limits'}:raise RuntimeError('Original TOPP gained mutable scene/state dependencies')
 return fn

def rss_bytes():return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
def _serve(connection,fn,parent):
 import numpy as np,torch
 # This child was forked in the one-thread/no-CUDA prelude, before the GPU child.
 ctypes.CDLL(None).prctl(1,signal.SIGTERM)
 if os.getppid()!=parent:os._exit(71)
 assert not torch.cuda.is_initialized()and len(list(Path('/proc/self/task').iterdir()))==1
 connection.send(dict(ready=True,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),cuda_initialized=False,os_threads=1,original_TOPP_code_sha256=sha(inspect.getsourcefile(fn))))
 while True:
  try:message=connection.recv()
  except EOFError:break
  if message is None:break
  seq,limits,args,kwargs,np_state,py_state=message
  try:
   assert not torch.cuda.is_initialized()and len(os.sched_getaffinity(0))<=16 and rss_bytes()<=128*1024**3
   np.random.set_state(np_state);random.setstate(py_state);before=torch.get_rng_state().clone()
   proxy=types.SimpleNamespace(joint_vel_limits=limits[0],joint_acc_limits=limits[1]);arrays=[*limits,*[v for v in args if isinstance(v,np.ndarray)]];snapshots=[v.copy()for v in arrays]
   try:
    result=fn(proxy,*args,**kwargs);reply=dict(ok=True,value=result)
   except Exception as error:reply=dict(ok=False,error=error)
   if not torch.equal(before,torch.get_rng_state()):raise SolverInfrastructureError('Original TOPP unexpectedly used Torch RNG')
   if any(not(v.shape==old.shape and v.dtype==old.dtype and v.tobytes()==old.tobytes())for v,old in zip(arrays,snapshots)):raise SolverInfrastructureError('Original TOPP mutated caller numeric input')
   connection.send((seq,reply,np.random.get_state(),random.getstate()))
  except SolverInfrastructureError as error:
   connection.send((seq,dict(infrastructure_error=str(error)),None,None));break
  except Exception as error:
   connection.send((seq,dict(infrastructure_error=repr(error),traceback=traceback.format_exc()),None,None));break
 connection.close();os._exit(0)

class Pool:
 def __init__(self,count=4):
  import torch
  if torch.cuda.is_initialized()or torch.cuda._is_in_bad_fork()or len(list(Path('/proc/self/task').iterdir()))!=1:raise RuntimeError('TOPP pool requires original one-thread/no-CUDA parent')
  self.exit_rows=[];self.parent=os.getpid();self.entries=[];self.free=queue.Queue();self.sequence=0;self.lock=threading.Lock();self.closed=False;fn=original_function()
  try:
   for index in range(count):
    a,b=socket.socketpair();client=Connection(a.detach());server=Connection(b.detach());pid=os.fork()
    if pid==0:
     client.close()
     for old in self.entries:old['connection'].close()
     _serve(server,fn,self.parent)
    server.close();entry=dict(pid=pid,connection=client,index=index);self.entries.append(entry)
    if not client.poll(30):raise RuntimeError('CPU TOPP worker did not become ready')
    receipt=client.recv();assert receipt['ready']and receipt['pid']==pid and receipt['affinity']==sorted(os.sched_getaffinity(0))and not receipt['cuda_initialized'];entry['receipt']=receipt;self.free.put(entry)
  except BaseException:self.close();raise
 def call(self,planner,*args,**kwargs):
  import numpy as np
  if self.closed:raise SolverInfrastructureError('Closed TOPP worker pool')
  with self.lock:self.sequence+=1;seq=self.sequence
  try:entry=self.free.get(timeout=60)
  except queue.Empty as error:raise SolverInfrastructureError('CPU solver pool stalled')from error
  try:
   # Original lane routing supplies exact caller RNG states; no scene/model crosses IPC.
   request=(seq,(np.asarray(planner.joint_vel_limits).copy(),np.asarray(planner.joint_acc_limits).copy()),args,kwargs,np.random.get_state(),random.getstate())
   try:
    entry['connection'].send(request)
    if not entry['connection'].poll(60):raise SolverInfrastructureError('Original TOPP worker timed out')
    returned,reply,np_state,py_state=entry['connection'].recv()
   except (OSError,EOFError,pickle.PickleError)as error:raise SolverInfrastructureError('CPU solver communication failed')from error
   if returned!=seq or 'infrastructure_error'in reply:raise SolverInfrastructureError(str(reply))
   np.random.set_state(np_state);random.setstate(py_state)
   if not reply['ok']:raise reply['error']
   return reply['value']
  finally:self.free.put(entry)
 def close(self):
  if self.closed:return
  self.closed=True
  if os.getpid()!=self.parent:
   for entry in self.entries:entry['connection'].close()
   return
  for entry in self.entries:
   try:entry['connection'].send(None)
   except (OSError,EOFError):pass
   entry['connection'].close()
  for entry in self.entries:
   until=time.monotonic()+5
   while True:
    try:pid,status=os.waitpid(entry['pid'],os.WNOHANG)
    except ChildProcessError:break
    if pid:
     self.exit_rows.append(dict(pid=pid,exit_code=os.waitstatus_to_exitcode(status),actual_waitpid=True,forced_termination=False));break
    if time.monotonic()>until:
     # These are un-reaped direct children created above; PID cannot be reused.
     os.kill(entry['pid'],signal.SIGTERM);pid,status=os.waitpid(entry['pid'],0);self.exit_rows.append(dict(pid=pid,exit_code=os.waitstatus_to_exitcode(status),actual_waitpid=True,forced_termination=True));break
    time.sleep(.01)

