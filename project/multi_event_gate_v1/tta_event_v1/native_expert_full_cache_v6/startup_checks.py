"""Check blocked admission, single release and cancellation on the real startup gate."""
import threading,time
from .startup import StagedLanes
from .paths import O,C,write,sha

def main():
 rows=[];cancel=threading.Event();gate=StagedLanes(cancel,rows.append)
 assert gate.wait(0)and gate.wait(1)
 finished=[];threads=[threading.Thread(target=lambda i=i:finished.append((i,gate.wait(i))))for i in range(2,6)]
 for t in threads:t.start()
 time.sleep(.15);assert finished==[]and rows==[]
 assert gate.release('actual_once_policy_query_complete',lane=0,original_index=55,query=0,tick=0)
 assert not gate.release('actual_once_policy_query_complete',lane=1,original_index=15,query=0,tick=0)
 for t in threads:t.join(1)
 assert not any(t.is_alive()for t in threads)and sorted(finished)==[(i,True)for i in range(2,6)]and len(rows)==1
 fallback=[];g2=StagedLanes(threading.Event(),fallback.append)
 assert g2.release('initial_case_terminal_without_query',lane=0,original_index=55,query_count=0)and g2.wait(5)and len(fallback)==1
 stop=threading.Event();g3=StagedLanes(stop,lambda _:None);out=[];worker=threading.Thread(target=lambda:out.append(g3.wait(4)));worker.start();stop.set();worker.join(1)
 assert not worker.is_alive()and out==[False]and not g3.wait(0)
 try:gate.release('timer_or_filler')
 except ValueError:pass
 else:raise AssertionError('Fabricated release reason accepted')
 write(O/'startup_checks.json',dict(passed=True,waiting_lanes_blocked_until_real_release=True,cancelled_waiters_terminate=True,single_release=True,missing_initial_case_fallback=True,CPU_only=True,GPU_calls=0,
 source_sha256={str(C/n):sha(C/n)for n in('startup.py','startup_checks.py','runtime.py')}))
 print('STAGED_REAL_LANE_ADMISSION_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
