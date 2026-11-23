"""Finite queue permutation/real concurrency/cancel tests without GPU or policy labels."""
import threading,time
from .work import FixedWorkQueue

def check():
 tasks=('open_laptop','grab_roller','beat_block_hammer','turn_switch','adjust_bottle')
 for count in(10,15):
  for per_task in(0,1,2,3):
   cached={t:{'items':[{}]*per_task}for t in tasks};cancel=threading.Event();q=FixedWorkQueue(count,tasks,cached,cancel)
   seen=[]
   for i in range(count):
    n=q.take();q.mark_selected(n);q.register_case(str(n),n);q.mark_case_ready(str(n));seen.append(n)
   assert q.take()is None and sorted(seen)==list(range(count))
   assert set(seen[:q.prefix])==set(range(q.prefix))
   for j in range(5):assert [n for n in seen if n%5==j]==list(range(j,count,5))
   if q.prefix in(0,count):
    assert seen==[j+5*r for j in(3,0,2,1,4)for r in range(count//5)]
    assert set(seen[:2*(count//5)])==set(range(3,count,5))|set(range(0,count,5))
   assert q.selected==q.ready==set(range(count))
 # Exact original cached-prefix readiness still blocks a cold dispatch.
 cancel=threading.Event();q=FixedWorkQueue(15,tasks,{t:{'items':[{}]}for t in tasks},cancel)
 prefix=[q.take()for _ in range(5)]
 for n in prefix:q.mark_selected(n);q.register_case(str(n),n)
 received=[];done=threading.Event()
 def take():
  try:received.append(q.take())
  finally:done.set()
 thread=threading.Thread(target=take);thread.start();assert not done.wait(.1)
 for n in prefix:q.mark_ready(n,reason='actual_once_policy_query_complete')
 assert done.wait(2);thread.join();assert received==[8]
 # Cancellation cannot escape the original cache readiness condition.
 cancel=threading.Event();q=FixedWorkQueue(10,tasks,{t:{'items':[{}]}for t in tasks},cancel)
 for _ in range(5):q.take()
 cancel.set()
 try:q.take()
 except InterruptedError:pass
 else:raise AssertionError('Cancelled prefix dispatch accepted')
 try:FixedWorkQueue(10,('unknown',),{},threading.Event())
 except AssertionError:pass
 else:raise AssertionError('Unexpected task set accepted')
 return True
