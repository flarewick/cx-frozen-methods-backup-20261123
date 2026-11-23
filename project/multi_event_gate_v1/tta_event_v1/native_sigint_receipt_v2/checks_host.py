"""Actual CPU-container ABBA verifies signal attribution and unchanged exits."""
from pathlib import Path
import json,os,signal,subprocess,sys,time
from native_policy_execution_v1.common import T,write,read,sha
P=T.parents[1]
from .receipt import read_receipts,LIBRARY
sys.path.insert(0,str(P/'deployment'))
import user_container
O=T/'outputs/native_sigint_receipt_v2/checks_v1'
CANON=Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA')

def actual(case,traced,mode,sapien=False):
 root=O/case;assert not root.exists();root.mkdir(parents=True)
 canonical=str(CANON/root.relative_to(P))
 script=str(CANON/(T/'native_onpolicy_arm_collection_v6/scripts/cpu.sh').relative_to(P))
 cmd=user_container.command(['bash',script,'-m','native_sigint_receipt_v2.fixture','--root',canonical,'--mode',mode],mount_project=True)
 if traced:cmd.append('--receipt')
 if sapien:cmd.append('--sapien')
 with(root/'host.log').open('x')as log:
  process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  code=None
  try:
   began=time.monotonic()
   while not(root/'ready.json').exists():
    assert process.poll()is None,(case,'early exit',process.returncode)
    assert time.monotonic()-began<60;time.sleep(.05)
   if mode=='interrupt':
    matches=[]
    for p in Path('/proc').iterdir():
     if not p.name.isdigit():continue
     try:
      if p.stat().st_uid!=os.getuid():continue
      args=(p/'cmdline').read_bytes().split(b'\0')
      if args and Path(args[0].decode()).name.startswith('python') and b'native_sigint_receipt_v2.fixture'in args and canonical.encode()in args:matches.append(int(p.name))
     except OSError:pass
    assert len(matches)==1,matches;os.kill(matches[0],signal.SIGINT)
   code=process.wait(timeout=60)
  finally:
   if process.poll()is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   exit=dict(exit_code=process.returncode,actual_Popen_wait_return=True,pid=process.pid,proc_absent_after_wait=not Path('/proc',str(process.pid)).exists())
   write(root/'actual_exit.json',exit)
 assert code==0 if mode=='normal'else code in(-2,130),(case,code)
 if mode!='normal':assert read(root/'handled.json')['signum']==signal.SIGINT
 return root

def main():
 O.mkdir(parents=True,exist_ok=True);assert not(O/'CPU_actual_signal_checks_v1.json').exists()
 roots=[actual('baseline_normal',False,'normal'),actual('traced_normal',True,'normal'),actual('traced_interrupt',True,'interrupt'),actual('baseline_interrupt',False,'interrupt')]
 assert read(roots[0]/'normal.json')==read(roots[1]/'normal.json')
 normal=read(roots[0]/'ready.json');traced=read(roots[1]/'ready.json');assert traced['seccomp_filters']==normal['seccomp_filters']and traced['OS_threads']==normal['OS_threads']
 assert read_receipts(roots[1]/'SIGINT_kernel_receipts.bin')==[]
 receipts=read_receipts(roots[2]/'SIGINT_kernel_receipts.bin');assert len(receipts)==1,receipts
 rec=receipts[0];assert rec['siginfo_verified']and rec['sender_verified'];assert rec['signal']==signal.SIGINT and rec['si_code']==0 and rec['sender_uid']==os.getuid()and rec['sender_pid_in_receiver_namespace']==0 and rec['receiver_pid']==read(roots[2]/'ready.json')['pid']and rec['monotonic_ns']>0,rec
 assert read(roots[2]/'actual_exit.json')['exit_code']==read(roots[3]/'actual_exit.json')['exit_code']
 one=actual('saved_one_argument_interrupt',True,'one_argument')
 recs=read_receipts(one/'SIGINT_kernel_receipts.bin');assert len(recs)==1 and not recs[0]['siginfo_verified']and not recs[0]['sender_verified']and recs[0]['si_code']is None and recs[0]['sender_uid']is None and recs[0]['sender_pid_in_receiver_namespace']is None,recs
 assert read(one/'actual_exit.json')['exit_code']==read(roots[3]/'actual_exit.json')['exit_code']
 physical=[actual('CPU_physx_baseline_normal',False,'normal',True),actual('CPU_physx_receipt_normal',True,'normal',True),actual('CPU_physx_receipt_interrupt',True,'interrupt',True),actual('CPU_physx_baseline_interrupt',False,'interrupt',True)]
 assert read(physical[0]/'normal.json')==read(physical[1]/'normal.json')
 assert read(physical[2]/'actual_exit.json')['exit_code']==read(physical[3]/'actual_exit.json')['exit_code']
 for root in physical:assert read(root/'CPU_physx.json')['actual_CPU_system_step']and not read(root/'CPU_physx.json')['CUDA_initialized']
 phys_receipts=read_receipts(physical[2]/'SIGINT_kernel_receipts.bin');assert len(phys_receipts)==1,phys_receipts
 if phys_receipts[0]['sender_verified']:assert phys_receipts[0]['sender_uid']==os.getuid()
 else:assert phys_receipts[0]['sender_uid']is None
 roots.extend([one,*physical])
 assert all(not read(r/'ready.json')['CUDA_initialized']for r in roots)
 for r in roots:
  text=(r/'host.log').read_text();assert not any(word in text for word in ['Operation not permitted','failed to install seccomp','disabling seccomp'])
 sources=[p for p in Path(__file__).parent.glob('*')if p.is_file()]
 write(O/'CPU_actual_signal_checks_v1.json',dict(passed=True,actual_container_normal_and_SIGINT_ABBA=True,sender_UID_and_SI_USER_verified=True,external_sender_PID_hidden_as0_by_PID_namespace=True,source_PID_must_not_be_guessed=True,original_interrupt_wait_exit_code_unchanged=True,normal_tensor_numpy_and_RNG_hashes_exact=True,no_continuous_trace_or_new_threads_or_seccomp_filters=True,original_handler_called_once=True,saved_one_argument_call_safely_keeps_sender_unknown=True,actual_CPU_PhysX_ABBA_exits_and_tensor_RNG_exact=True,CPU_PhysX_interrupt_receipt=phys_receipts[0],normal_receipts_empty=True,no_signal_injection_or_ignoring=True,no_CUDA_GPU_native_or_optimizer_work=True,source_sha256={str(p):sha(p)for p in sources},input_sha256={str(p):sha(p)for root in roots for p in root.glob('*')if p.is_file()}|{str(LIBRARY):sha(LIBRARY)}))
 print('CPU_ACTUAL_ASYNC_SIGINT_RECEIPT_AND_ORIGINAL_HANDLER_PASS',flush=True)

if __name__=='__main__':main()
