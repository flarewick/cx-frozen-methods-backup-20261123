"""Bounded real scene workers: original full first, then independent whole branches."""
from collections import deque
import threading
class BranchQueue:
 def __init__(self,work,count,cancel):
  assert work.prefix in(0,count),'Only cold or complete verified prepared prefixes; no mixed prefix admission'
  self.work,self.count,self.cancel=work,count,cancel;self.cv=threading.Condition();self.pending=deque();self.leaders=0;self.submitted=set();self.completed=set()
 def take(self):
  with self.cv:
   while True:
    if self.cancel.is_set():raise InterruptedError('Peer cancelled real branch queue')
    if self.pending:return self.pending.popleft()
    if self.leaders<self.count:
     index=self.work.take();assert index is not None;self.leaders+=1;return('full',index,None,None)
    if len(self.completed)==self.count:return None
    self.cv.wait(.05)
 def submit(self,item,shared):
  with self.cv:
   key=item['case_id'];assert key not in self.submitted and shared['done']=={'M_event_full'}
   assert shared['branches'][0]=='M_event_full'and len(set(shared['branches']))==len(shared['branches'])
   self.submitted.add(key)
   self.pending.extend(('branch',item,shared,branch)for branch in shared['branches'][1:]);self.cv.notify_all()
 def complete(self,key):
  with self.cv:
   assert key in self.submitted and key not in self.completed
   self.completed.add(key);self.cv.notify_all()
