"""Parent-only signal-delivery stops. Never PTRACE_SYSCALL or seccomp tracing."""
import ctypes,errno,json,os,signal,time
from pathlib import Path
SEIZE=0x4206;GETSIGINFO=0x4202;CONT=7;LISTEN=0x4208;EXITKILL=0x00100000
libc=ctypes.CDLL(None,use_errno=True)
libc.ptrace.argtypes=[ctypes.c_uint,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
libc.ptrace.restype=ctypes.c_long

def ptrace(request,pid,addr=0,data=0):
 value=libc.ptrace(request,pid,ctypes.c_void_p(addr),ctypes.c_void_p(data))
 if value==-1:
  error=ctypes.get_errno();raise OSError(error,os.strerror(error))
 return value

def attach(pid):
 # Only our direct child is permitted. No clone/syscall/exec tracing options.
 assert Path('/proc',str(pid),'status').exists()
 status=dict(x.split(':',1)for x in Path('/proc',str(pid),'status').read_text().splitlines()if ':'in x)
 assert int(status['PPid'])==os.getpid()
 ptrace(SEIZE,pid,data=EXITKILL)

def wait_and_record(pid,path,*,original_wait,events,**identity):
 events=Path(events);assert not events.exists();events.parent.mkdir(parents=True,exist_ok=True)
 count=0
 with events.open('x')as out:
  while True:
   peek=os.waitid(os.P_PID,pid,os.WEXITED|os.WSTOPPED|os.WNOWAIT)
   assert peek.si_pid==pid
   if peek.si_code in(os.CLD_EXITED,os.CLD_KILLED,os.CLD_DUMPED):
    # Original implementation performs and records the one actual exit wait.
    code=original_wait(pid,path,**identity)
    break
   observed,status=os.waitpid(pid,os.WUNTRACED)
   assert observed==pid and os.WIFSTOPPED(status)
   sig=os.WSTOPSIG(status);event=status>>16;info=ctypes.create_string_buffer(128)
   if event==128:
    if sig in(signal.SIGSTOP,signal.SIGTSTP,signal.SIGTTIN,signal.SIGTTOU):ptrace(LISTEN,pid)
    else:ptrace(CONT,pid,data=0)
    continue
   try:ptrace(GETSIGINFO,pid,data=ctypes.addressof(info))
   except OSError as e:
    if e.errno!=errno.EINVAL:raise
    # SEIZE group-stop: retain stop rather than injecting another stop.
    ptrace(LISTEN,pid);continue
   raw=bytes(info);import struct
   signo,error,kind=struct.unpack_from('iii',raw,0)
   if event==128:
    ptrace(CONT,pid,data=0);continue
   assert signo==sig
   row=dict(signal=sig,si_code=kind,observed_monotonic_ns=time.monotonic_ns(),
            tracee_pid=pid,original_signal_forwarded_once=True,no_syscall_tracing=True)
   if kind in(0,-1,-6):
    sender,uid=struct.unpack_from('iI',raw,16)
    row.update(sender_pid_in_tracer_namespace=sender,sender_uid=uid,sender_verified=True,
               external_PID_may_be0_and_must_not_be_guessed=True)
   else:row.update(sender_verified=False,sender_uid=None,sender_pid_in_tracer_namespace=None)
   out.write(json.dumps(row)+'\n');out.flush();count+=1
   ptrace(CONT,pid,data=sig)
 receipt=Path(events).with_name('signal_event_wait_actual_exit.json')
 receipt.write_text(json.dumps(dict(actual_exit_code=code,actual_original_wait_used=True,
                    observed_signal_stops=count,no_syscall_or_clone_tracing=True,
                    no_signal_ignored_or_synthesized=True,CUDA_initialized_by_tracer=False),indent=2)+'\n')
 return code
