import threading
class OrderedScreen:
 def __init__(self,select,tasks,cancel):
  self.select,self.tasks,self.cancel=select,list(tasks),cancel;self.condition=threading.Condition();self.next_index={t:i for i,t in enumerate(tasks)}
 def next(self,number):
  task=self.tasks[number%len(self.tasks)]
  with self.condition:
   while self.next_index[task]!=number:
    if self.cancel.is_set():raise InterruptedError('Peer cancelled while waiting for fixed task order')
    self.condition.wait(.05)
  # One real owner-serialized expert screen. Other tasks can queue their own
  # original policy work; never access the shared planner from a worker.
  item=self.select(task)
  with self.condition:
   assert self.next_index[task]==number;self.next_index[task]+=len(self.tasks);self.condition.notify_all()
  return item
