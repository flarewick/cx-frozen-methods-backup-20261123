"""Ordered per-task admission of real screens; no outcome-dependent scheduling."""
import threading
class OrderedScreen:
 def __init__(self,select,tasks,cancel):
  assert len(tasks)==2
  self.select,self.tasks,self.cancel=select,tasks,cancel
  self.turn=[0,1];self.conditions=[threading.Condition(),threading.Condition()]
 def next(self,number):
  if not 0<=number<8:raise ValueError('Invalid lane')
  slot=number%2
  with self.conditions[slot]:
   while self.turn[slot]!=number:
    if self.cancel.is_set():return None
    self.conditions[slot].wait(.05)
   if self.cancel.is_set():return None
   # Keep the order lock through the real original screen, even for errors.
   try:item=self.select(self.tasks[slot])
   except BaseException:
    self.cancel.set();self.conditions[slot].notify_all();raise
   self.turn[slot]+=2;self.conditions[slot].notify_all();return item
