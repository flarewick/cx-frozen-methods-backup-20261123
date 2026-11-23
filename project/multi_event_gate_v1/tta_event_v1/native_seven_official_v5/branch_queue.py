"""Bounded real scene workers: original full first, then independent whole branches."""
from collections import deque
import threading
class BranchQueue:
 def __init__(self,work,count,cancel,*,early=False):
  assert 0<=work.prefix<=count,'Verified prefix stays within the required case set'
  self.early=bool(early);self.active_references=0;self.max_active_references=0
  self.work,self.count,self.cancel=work,count,cancel;self.cv=threading.Condition();self.pending=deque();self.leaders=0;self.submitted=set();self.completed=set()
 def take(self):
  with self.cv:
   while True:
    if self.cancel.is_set():raise InterruptedError('Peer cancelled real branch queue')
    if self.pending:
     if not self.early:return self.pending.popleft()
     for i,job in enumerate(self.pending):
      if job[3]!='handoff_reference'or self.active_references==0:
       self.pending.rotate(-i);value=self.pending.popleft();self.pending.rotate(i)
       if value[3]=='handoff_reference':self.active_references+=1;self.max_active_references=max(self.max_active_references,self.active_references)
       return value
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
 def publish_initial(self,item,shared,witness):
  with self.cv:
   assert set(witness)=={'query','actor_observation_tick','base_infer_calls','actual_internal_noise_calls','native_environment_unchanged','source_policy_file','source_policy_sha256','only_q0_provenance_no_full_result_or_labels','full_result_absent_at_dispatch'}
   assert witness['query']==witness['actor_observation_tick']==0
   assert witness['base_infer_calls']==witness['actual_internal_noise_calls']==1
   assert witness['native_environment_unchanged']and witness['only_q0_provenance_no_full_result_or_labels']and witness['full_result_absent_at_dispatch']
   assert self.early and shared.get('early_pairing')and not shared['done']
   key=item['case_id'];assert key not in self.submitted and key in self.work.case_index and self.work.case_index[key]in self.work.ready
   assert witness['source_policy_file'].endswith('/M_event_full/policy000.npz')and len(witness['source_policy_sha256'])==64
   assert shared['branches'][0]=='M_event_full'and len(set(shared['branches']))==len(shared['branches'])
   self.submitted.add(key)
   self.pending.extend(('branch',item,shared,branch)for branch in shared['branches'][1:]);self.cv.notify_all()
 def retire(self,job):
  if not self.early or job[0]!='branch'or job[3]!='handoff_reference':return
  with self.cv:
   assert self.active_references==1;self.active_references=0;self.cv.notify_all()
 def complete(self,key):
  with self.cv:
   assert key in self.submitted and key not in self.completed
   self.completed.add(key);self.cv.notify_all()
