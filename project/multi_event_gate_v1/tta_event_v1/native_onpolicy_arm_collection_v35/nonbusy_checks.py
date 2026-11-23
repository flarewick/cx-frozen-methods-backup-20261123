"""Run the actual production wait loop in a real CPU thread before CUDA."""
import ast,threading,time,queue,traceback
from types import SimpleNamespace
from .common import C
from native_early_next_preparation_v1.schedule import ready

def run():
 tree=ast.parse((C/'runtime.py').read_text())
 execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
 wait=next(n for n in ast.walk(execute)if isinstance(n,ast.FunctionDef)and n.name=='prepare_future')
 assert sum(isinstance(n,ast.Call)and ast.unparse(n)=="future_ready.wait(0.05)"for n in ast.walk(wait))==1
 cancel=threading.Event();event=threading.Event();calls=[];failures=[];stats={}
 work=SimpleNamespace(condition=threading.Condition(),ready=set())
 def counted(*args):calls.append(time.monotonic());return ready(*args)
 namespace=dict(spec=dict(future_preparation_each_task=2,future_preparation_population=dict(split='fit',seed_start=7124000,first_index=0),dataset_split='adapter_validation'),
  early_next_ready=counted,work=work,future_ready=event,future_done=threading.Event(),c=SimpleNamespace(cancel=cancel),
  traceback=traceback,errors=queue.Queue(),write=lambda *args:failures.append(args),prepared={'session':C})
 exec(compile(ast.fix_missing_locations(ast.Module(body=[wait],type_ignores=[])),str(C/'runtime.py'),'exec'),namespace)
 def actual_wait():
  began=time.thread_time();wall=time.monotonic();namespace['prepare_future']()
  stats.update(thread_CPU_seconds=time.thread_time()-began,wall_seconds=time.monotonic()-wall)
 thread=threading.Thread(target=actual_wait);thread.start();time.sleep(.19);cancel.set();thread.join(2)
 assert not thread.is_alive()and not failures and namespace['errors'].empty()
 assert 1<=len(calls)<=8 and stats['thread_CPU_seconds']<.05
 return dict(passed=True,actual_production_wait_loop=True,iterations=len(calls),**stats,
  cancellation_preserved=True,no_GPU_or_work_or_readiness_fabricated=True)
