"""Earlier cached cases enter native execution before cold expert screening."""
import threading
class FixedWorkQueue:
 def __init__(self,count,tasks,cached,cancel):
  self.count=count;self.cancel=cancel;self.condition=threading.Condition();self.next=0;self.selected=set();self.ready=set();self.case_index={};self.prefix=0
  assert tuple(tasks)==('beat_block_hammer','turn_switch','adjust_bottle','open_laptop')
  for number in range(count):
   if number//len(tasks)>=len(cached.get(tasks[number%len(tasks)],{}).get('items',[])):break
   self.prefix+=1
  # Fixed engineering priority from the diagnosed serial tail, never success labels.
  priority={t:i for i,t in enumerate(('turn_switch','open_laptop','beat_block_hammer','adjust_bottle'))}
  key=lambda n:(priority[tasks[n%len(tasks)]],n//len(tasks))
  self.order=sorted(range(self.prefix),key=key)+sorted(range(self.prefix,count),key=key)
  assert sorted(self.order)==list(range(count))
 def take(self):
  with self.condition:
   while self.next>=self.prefix and len(self.ready.intersection(range(self.prefix)))<self.prefix:
    if self.cancel.is_set():raise InterruptedError('Peer cancelled before cached prefix entered native work')
    self.condition.wait(.05)
   if self.next>=self.count:return None
   result=self.order[self.next];self.next+=1;return result
 def mark_selected(self,number):
  with self.condition:
   assert number in self.order[:self.next] and number not in self.selected
   self.selected.add(number);self.condition.notify_all()

 def register_case(self,case_id,number):
  with self.condition:
   assert number in self.selected and case_id not in self.case_index
   self.case_index[case_id]=number
 def mark_case_ready(self,case_id):
  return self.mark_ready(self.case_index[case_id],reason='actual_once_policy_query_complete')
 def mark_ready(self,number,*,reason):
  assert reason in('actual_once_policy_query_complete','actual_case_terminal')
  with self.condition:
   assert number in self.selected
   self.ready.add(number);self.condition.notify_all()
