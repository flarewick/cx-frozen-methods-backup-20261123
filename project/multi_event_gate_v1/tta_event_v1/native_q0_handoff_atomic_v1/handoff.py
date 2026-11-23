"""Return the actual q0 task before queued future expert preparation can overtake it."""
import ast,inspect,textwrap
from native_q0_task_handoff_v3.handoff import capture,take,detach
OLD="with self.serial_screen_lock,self.locks[name],c.rng.bind(lane):return operation()"
NEW="""with self.serial_screen_lock,self.locks[name],c.rng.bind(lane):
  item=operation()
  _atomic_owner_detach(self,name,item,lane)
  return item"""

def transform(source):
 assert source.count(OLD)==1
 changed=source.replace(OLD,NEW)
 assert restore(changed)==source
 ast.parse(changed)
 return changed

def restore(source):
 assert source.count(NEW)==1
 return source.replace(NEW,OLD)

def schedule_class(original,torch,models):
 method=original.next;source=textwrap.dedent(inspect.getsource(method))
 def owner_detach(self,name,item,lane):
  assert self.serial_screen_lock.owner==__import__('threading').get_ident()
  return self.c.call(lane,detach,self,name,item,torch,models)
 namespace=dict(method.__globals__,_atomic_owner_detach=owner_detach)
 exec(compile(transform(source),__file__,'exec'),namespace)
 class AtomicHandoffSchedule(original):
  next=namespace[method.__name__]
 return AtomicHandoffSchedule
