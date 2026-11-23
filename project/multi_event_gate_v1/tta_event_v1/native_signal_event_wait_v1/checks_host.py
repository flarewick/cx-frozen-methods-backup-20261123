"""Actual container CPU ABBA; checks target-only waits and original exits."""
from pathlib import Path
import json,os,signal,subprocess,sys,time
from native_policy_execution_v1.common import T,write,read,sha
P=T.parents[1];sys.path.insert(0,str(P/'deployment'));import user_container
CANON=Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA')
O=T/'outputs/native_signal_event_wait_v1/checks_v2'

def actual(name,traced,mode):
 root=O/name;root.mkdir(parents=True);canonical=str(CANON/root.relative_to(P))
 script=str(CANON/(T/'native_onpolicy_arm_collection_v17/scripts/cpu.sh').relative_to(P))
 cmd=user_container.command(['bash',script,'-m','native_signal_event_wait_v1.fixture','--root',canonical,'--mode',mode],mount_project=True)
 if traced:cmd.append('--traced')
 with(root/'host.log').open('x')as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   began=time.monotonic()
   while not(root/'ready.json').exists():
    assert proc.poll()is None,(name,'early exit',proc.returncode)
    assert time.monotonic()-began<60;time.sleep(.02)
   ready=read(root/'ready.json');assert ready['original_siginfo_flag_lost']and not ready['CUDA_initialized']
   if mode!='normal':
    matches=[]
    for p in Path('/proc').iterdir():
     if not p.name.isdigit():continue
     try:
      if p.stat().st_uid!=os.getuid():continue
      args=(p/'cmdline').read_bytes().split(b'\0')
      if b'native_signal_event_wait_v1.fixture'not in args or canonical.encode()not in args:continue
      fields=dict(x.split(':',1)for x in(p/'status').read_text().splitlines()if ':'in x)
      if int(fields['NSpid'].split()[-1])==ready['child_pid']:matches.append(int(p.name))
     except OSError:pass
    assert len(matches)==1,matches;target=matches[0]
    if mode=='groupstop':
     os.kill(target,signal.SIGSTOP);time.sleep(.2)
     status=Path('/proc',str(target),'status').read_text();assert any(x in status for x in ['(stopped)','(tracing stop)']),status[:200]
     os.kill(target,signal.SIGCONT);time.sleep(.2)
    os.kill(target,signal.SIGINT)
   code=proc.wait(timeout=30)
  finally:
   if proc.poll()is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   write(root/'actual_host_exit.json',dict(exit_code=proc.returncode,actual_Popen_wait_return=True,pid=proc.pid,proc_absent_after_wait=not Path('/proc',str(proc.pid)).exists()))
 assert code==0,(name,code)
 parent=read(root/'parent_exit.json');assert parent['child_code']==(0 if mode=='normal'else 130)and parent['no_extra_parent_threads']
 if mode!='normal':assert read(root/'handled.json')['once']
 return root

def main():
 assert not O.exists();O.mkdir(parents=True)
 normal=[actual('normal_A',False,'normal'),actual('normal_B',True,'normal'),actual('normal_B2',True,'normal'),actual('normal_A2',False,'normal')]
 stats=[read(p/'normal.json')for p in normal]
 for key in ['tensor','array','RNG','threads','CUDA_initialized']:assert len({x[key]for x in stats})==1,(key,stats)
 assert all(x['threads']==1 and not x['CUDA_initialized']for x in stats)
 # Excludes previous strace failure mode (millions of involuntary stops).
 assert max(x['voluntary_context_switches']+x['nonvoluntary_context_switches']for x in stats)<1000,stats
 for p in [normal[1],normal[2]]:assert(p/'events.jsonl').read_text()==''and read(p/'signal_event_wait_actual_exit.json')['observed_signal_stops']==0
 pairs=[actual('interrupt_A',False,'interrupt'),actual('interrupt_B',True,'interrupt'),actual('groupstop_A',False,'groupstop'),actual('groupstop_B',True,'groupstop')]
 from native_sigint_receipt_v2.receipt import read_receipts
 for p in pairs:
  receipt=read_receipts(p/'SIGINT_kernel_receipts.bin');assert len(receipt)==1 and not receipt[0]['sender_verified'],receipt
 for p in [pairs[1],pairs[3]]:
  rows=[json.loads(x)for x in(p/'events.jsonl').read_text().splitlines()];sigs=[x for x in rows if x['signal']==2]
  assert len(sigs)==1 and sigs[0]['sender_verified']and sigs[0]['sender_uid']==os.getuid()and sigs[0]['sender_pid_in_tracer_namespace']==0,sigs
  assert all(x['original_signal_forwarded_once']and x['no_syscall_tracing']for x in rows)
  exit=read(p/'original_exit.json');assert exit['waited_child_inner_pid']==exit['child_inner_pid']and exit['proc_absent_after_wait']
 sources=[p for p in Path(__file__).parent.iterdir()if p.is_file()]
 write(O/'CPU_actual_signal_wait_checks_v1.json',dict(passed=True,actual_container_ABBA=True,actual_lost_SA_SIGINFO_reproduced=True,original_v2_keeps_sender_unknown=True,
   parent_signal_event_verifies_sender_UID_and_kernel_code=True,external_sender_PID0_not_guessed=True,
   original_exit_wait_and_interrupt130_preserved=True,original_SIGSTOP_SIGCONT_preserved=True,
   actual_tensor_numpy_RNG_threads_exact=True,normal_signal_stops0=True,no_syscall_or_clone_tracing=True,
   normal_CPU_context_switches=stats,no_GPU_performance_claim=True,no_CUDA_or_optimizer=True,
   source_sha256={str(p):sha(p)for p in sources},input_sha256={str(p):sha(p)for r in [*normal,*pairs]for p in r.iterdir()if p.is_file()}))
 print('CPU_SIGNAL_ONLY_PARENT_WAIT_ABBA_PASS',flush=True)
if __name__=='__main__':main()
