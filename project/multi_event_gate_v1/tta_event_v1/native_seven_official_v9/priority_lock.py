"""FIFO within priority; current recovery precedes prospective preparation."""
import contextlib,heapq,itertools,threading
class PriorityLock:
 def __init__(self,cancel):
  self.cancel=cancel;self.condition=threading.Condition();self.waiting=[]
  self.sequence=itertools.count();self.local=threading.local();self.owner=None
 @contextlib.contextmanager
 def background(self):
  previous=getattr(self.local,'priority',0);self.local.priority=1
  try:yield
  finally:self.local.priority=previous
 def __enter__(self):
  ident=threading.get_ident()
  with self.condition:
   assert self.owner!=ident,'The original expert lock is non-reentrant'
   ticket=(getattr(self.local,'priority',0),next(self.sequence),ident)
   heapq.heappush(self.waiting,ticket)
   try:
    while self.owner is not None or self.waiting[0]!=ticket:
     if self.cancel.is_set():raise InterruptedError('Peer cancelled before original expert work')
     self.condition.wait(.05)
    if self.cancel.is_set():raise InterruptedError('Peer cancelled before original expert work')
    assert heapq.heappop(self.waiting)==ticket;self.owner=ident
   except BaseException:
    self.waiting.remove(ticket);heapq.heapify(self.waiting);self.condition.notify_all();raise
  return self
 def __exit__(self,*exc):
  with self.condition:
   assert self.owner==threading.get_ident();self.owner=None;self.condition.notify_all()
