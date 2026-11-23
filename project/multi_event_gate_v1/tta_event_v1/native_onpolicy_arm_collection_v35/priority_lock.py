"""One original planner; bounded fairness between required kinds of real work."""
import contextlib,itertools,threading

class FutureTail(RuntimeError):
 pass

class PriorityLock:
 def __init__(self,cancel):
  self.cancel=cancel;self.condition=threading.Condition();self.waiting=[]
  self.sequence=itertools.count();self.local=threading.local();self.owner=None
  self.last_foreground=None;self.foreground_burst=0;self.screen_burst=0
 @contextlib.contextmanager
 def background(self):
  previous=getattr(self.local,'priority',0);self.local.priority=1
  try:yield
  finally:self.local.priority=previous
 @contextlib.contextmanager
 def screen(self,allow_future=None):
  previous=(getattr(self.local,'screen',False),getattr(self.local,'allow_future',None))
  self.local.screen=True;self.local.allow_future=allow_future
  try:yield
  finally:self.local.screen,self.local.allow_future=previous
 def chosen(self):
  oldest={}
  for row in self.waiting:
   oldest.setdefault(row[1],row)
  if 'future'in oldest and(self.foreground_burst>=1 or not({'screen','recovery'}&oldest.keys())):
   return oldest['future']
  if 'screen'in oldest and'recovery'in oldest:
   return oldest['screen'if self.screen_burst<2 else'recovery']
  return min((row for kind,row in oldest.items()if kind!='future'),default=oldest.get('future'))
 def __enter__(self):
  ident=threading.get_ident()
  with self.condition:
   assert self.owner!=ident,'The original expert lock is non-reentrant'
   kind='future'if getattr(self.local,'priority',0)else('screen'if getattr(self.local,'screen',False)else'recovery')
   ticket=(next(self.sequence),kind,ident,getattr(self.local,'allow_future',None))
   self.waiting.append(ticket)
   try:
    while self.owner is not None or self.chosen()!=ticket:
     if self.cancel.is_set():raise InterruptedError('Peer cancelled before original expert work')
     self.condition.wait(.05)
    if self.cancel.is_set():raise InterruptedError('Peer cancelled before original expert work')
    if kind=='future'and ticket[3]is not None and not ticket[3]():
     raise FutureTail('No additional future expert is dispatched after the natural current-case tail')
    self.waiting.remove(ticket);self.owner=ident
    if kind=='future':self.foreground_burst=0
    else:
     self.foreground_burst+=1;self.last_foreground=kind
     self.screen_burst=self.screen_burst+1 if kind=='screen'else 0
   except BaseException:
    if ticket in self.waiting:self.waiting.remove(ticket)
    self.condition.notify_all();raise
  return self
 def __exit__(self,*exc):
  with self.condition:
   assert self.owner==threading.get_ident();self.owner=None;self.condition.notify_all()
