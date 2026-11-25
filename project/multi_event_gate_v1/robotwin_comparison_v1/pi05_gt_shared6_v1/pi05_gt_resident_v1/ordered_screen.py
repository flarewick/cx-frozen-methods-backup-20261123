"""Preserve registered per-task order with a variable one-to-eight-lane tail."""
import threading
class OrderedScreen:
 def __init__(self,select,tasks,cancel):
  self.select,self.tasks,self.cancel=select,list(tasks),cancel;self.condition=threading.Condition();self.pending={t:[i for i,x in enumerate(tasks)if x==t]for t in tasks}
 def next(self,number):
  task=self.tasks[number]
  with self.condition:
   while self.pending[task][0]!=number:
    if self.cancel.is_set():return None
    self.condition.wait(.05)
  try:return self.select(task)
  finally:
   with self.condition:assert self.pending[task].pop(0)==number;self.condition.notify_all()
