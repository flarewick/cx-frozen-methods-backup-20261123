"""Real lock/factory concurrency; no actual native/GPU detach is claimed."""
import ast,contextlib,inspect,textwrap,threading,time
from pathlib import Path
from types import SimpleNamespace
from native_policy_execution_v1.common import T,write,sha
from native_onpolicy_collection_v28.priority_lock import PriorityLock
from native_q0_task_handoff_v3 import handoff as old
from . import handoff as new

class CPUBase:
 def next(self,name):
  c=self.c;lane=self.lanes[name]
  def operation():return self.operation(name)
  with self.serial_screen_lock,self.locks[name],c.rng.bind(lane):return operation()

def contention(factory):
 cancel=threading.Event();lock=PriorityLock(cancel);events=[];gate=threading.Event();errors=[];calls=[];models=object();torch=object()
 def dispatch(lane,fn,*args):
  assert fn is old.detach and lock.owner==threading.get_ident()
  assert args[0]is obj and args[1]==lane and args[2]['case_id']==lane and args[3]is torch and args[4]is models
  calls.append(lane);events.append(('detach',lane))
 c=SimpleNamespace(call=dispatch,rng=SimpleNamespace(bind=lambda lane:contextlib.nullcontext()))
 cls=factory(CPUBase,torch,models);obj=cls();obj.c=c;obj.serial_screen_lock=lock;obj.locks={str(i):threading.Lock()for i in range(9)};obj.lanes={str(i):str(i)for i in range(9)}
 def operation(name):
  events.append(('screen',name))
  if name=='0':
   gate.set();deadline=time.monotonic()+5
   while True:
    with lock.condition:waiting=len(lock.waiting)
    if waiting==8:break
    assert time.monotonic()<deadline,'Eight real waiting workers required';cancel.wait(.001)
  return dict(case_id=name)
 obj.operation=operation
 def worker(name):
  try:
   if name!='0':assert gate.wait(5)
   result=obj.next(name);assert result==dict(case_id=name);events.append(('ready',name))
  except BaseException as e:errors.append(repr(e));cancel.set()
 threads=[threading.Thread(target=worker,args=(str(i),))for i in range(9)]
 for thread in threads:thread.start()
 for thread in threads:thread.join(8);assert not thread.is_alive()
 assert not errors and len(calls)==9 and set(calls)==set(obj.lanes)and lock.owner is None
 return events

def main():
 from native_onpolicy_collection_v28.screen import Schedule
 source=textwrap.dedent(inspect.getsource(Schedule.next));assert new.restore(new.transform(source))==source
 assert new.capture is old.capture and new.take is old.take and new.detach is old.detach
 before=contention(old.schedule_class);after=contention(new.schedule_class)
 assert sum(kind=='screen'for kind,name in before[:before.index(('detach','0'))])==9
 assert after[0:2]==[('screen','0'),('detach','0')]
 for events in(before,after):assert sorted(name for kind,name in events if kind=='screen')==[str(i)for i in range(9)]
 out=T/'outputs/native_q0_handoff_atomic_v1';out.mkdir(exist_ok=True)
 write(out/'CPU_actual_lock_handoff_checks_v1.json',dict(passed=True,
  original_actual_Schedule_source_reversible_exact=True,original_capture_take_detach_function_objects_exact=True,
  actual_nine_thread_original_lock_and_factories=True,old_first_detach_after_screens=9,new_first_detach_after_screens=1,
  original_no_extra_screen_or_detach_calls=True,original_nonreentrant_lock_preserved=True,
  original_events=before,new_events=after,CPU_dispatch_stub_only_not_actual_native_or_GPU=True,
  native_controls_resources_and_actual_exit_still_required=True,
  source_sha256={str(f):sha(f)for f in Path(__file__).parent.glob('*')if f.is_file()},
  input_sha256={str(f):sha(f)for f in[T/'native_onpolicy_collection_v28/screen.py',T/'native_onpolicy_collection_v28/priority_lock.py',Path(old.__file__)]}))
 print('CPU_REAL9_LOCK_HANDOFF_FUTURE_QUEUE_REGRESSION_PASS',flush=True)
if __name__=='__main__':main()
