"""Original real-query gate and cancellation; CPU only, no simulation."""
from pathlib import Path
import threading,time
from .startup import StagedLanes
from .common import O,write

def main():
 for total in(8,10):
  cancel=threading.Event();events=[];gate=StagedLanes(cancel,events.append,lanes=total);returned=[]
  assert gate.wait(0)and gate.wait(1)
  threads=[threading.Thread(target=lambda n=n:returned.append((n,gate.wait(n))))for n in range(2,total)]
  for t in threads:t.start()
  assert not gate.release('actual_once_policy_query_complete',lane=0,query=1,tick=0)
  time.sleep(.03);assert not returned
  assert gate.release('actual_once_policy_query_complete',lane=1,query=0,tick=0)
  for t in threads:t.join(timeout=2)
  assert sorted(returned)==[(n,True)for n in range(2,total)]and events[0]['released_lanes']==list(range(2,total))and len(events)==1
  assert not gate.release('actual_once_policy_query_complete',lane=0,query=0,tick=0)
  cancel=threading.Event();gate=StagedLanes(cancel,events.append,lanes=total);cancel.set();assert gate.wait(2)is False
 write(O/'CPU_staged_FIFO_checks.json',dict(passed=True,both8_and10_real_lanes=True,first_two_admitted_without_filler=True,remaining_block_until_actual_query0tick0=True,wrong_query_rejected=True,exact_remaining_admission_once=True,cancellation_honored=True,no_GPU_context_or_policy_called=True))
 print('REAL_QUERY_STAGED_FIFO_CPU_PASS',flush=True)
if __name__=='__main__':main()
