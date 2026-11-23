"""Scoped worker entry only; original accounting/selection/deadlines untouched."""
from contextlib import contextmanager
from pathlib import Path
import inspect,sys
from native_process_sampler_v13.stream import Session as OriginalSession
class OrderedSession(OriginalSession):
 def __init__(self,index,proof=None,command=None,deadline=3):
  assert command is None
  super().__init__(index,proof,command=[sys.executable,'-B','-m','native_nvml_query_order_v3.worker',str(index),'--serve'],deadline=deadline)
@contextmanager
def context():
 from native_process_sampler_v13 import stream,metrics
 from native_onpolicy_sampler_health_v1 import health
 old_session=stream.Session;old_health=health.Session;old_one=metrics.one_shot_process_utilization
 assert old_session is OriginalSession and old_health is OriginalSession
 source=inspect.getsource(old_one)
 old="[sys.executable,'-B',str(Path(__file__).with_name('worker.py')),str(index)]"
 new="[sys.executable,'-B','-m','native_nvml_query_order_v3.worker',str(index)]"
 assert source.count(old)==1
 stream.Session=OrderedSession;health.Session=OrderedSession
 exec(compile(source.replace(old,new),metrics.__file__,'exec'),metrics.__dict__)
 try:yield
 finally:stream.Session=old_session;health.Session=old_health;metrics.one_shot_process_utilization=old_one
