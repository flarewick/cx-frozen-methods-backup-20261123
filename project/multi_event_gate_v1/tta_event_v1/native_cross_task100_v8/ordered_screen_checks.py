"""Reverse-start all lanes, cancellation/error release and exact task seed order."""
import threading,time,ast
from .ordered_screen import OrderedScreen
from .common import C,T,O,write,sha
def main():
 events=[];counts={'cup':0,'shoe':0};cancel=threading.Event();outputs={}
 def select(task):
  i=counts[task];counts[task]+=1;events.append((task,i));return (task,i)
 ordered=OrderedScreen(select,('cup','shoe'),cancel)
 ts=[threading.Thread(target=lambda n=n:outputs.__setitem__(n,ordered.next(n)))for n in reversed(range(8))]
 for t in ts:t.start()
 for t in ts:t.join(2)
 assert not any(t.is_alive()for t in ts)
 assert outputs=={n:(('cup','shoe')[n%2],n//2)for n in range(8)}
 c=threading.Event();g=OrderedScreen(select,('cup','shoe'),c);out=[];t=threading.Thread(target=lambda:out.append(g.next(6)));t.start();c.set();t.join(1);assert out==[None]
 def fail(task):raise RuntimeError('screen failure')
 c=threading.Event();g=OrderedScreen(fail,('cup','shoe'),c);out=[];t=threading.Thread(target=lambda:out.append(g.next(2)));t.start()
 try:g.next(0)
 except RuntimeError:pass
 else:raise AssertionError('Failure swallowed')
 t.join(1);assert out==[None]and c.is_set()
 def body(path,name):return next(n for n in ast.walk(ast.parse(path.read_text()))if isinstance(n,ast.FunctionDef)and n.name==name)
 assert ast.dump(body(C/'runtime.py','run_case'))==ast.dump(body(T/'native_cross_task100_v7/runtime.py','run_case'))
 assert (C/'screen.py').read_text()==(T/'native_cross_task100_v7/screen.py').read_text()
 write(O/'CPU_ordered_screen_checks.json',dict(passed=True,reverse_thread_start_preserves_per_task_exact_seed_and_lane_order=True,cancellation_and_screen_errors_fail_closed=True,entire_run_case_AST_exact=True,entire_original_expert_screen_source_exact=True,no_cached_expert_outcome_or_masks_imported=True,no_GPU_calls=True,source_sha256={str(p):sha(p)for p in C.glob('*.py')}))
 print('ORDERED_INTERLEAVED_SCREENS_CPU_PASS',flush=True)
if __name__=='__main__':main()
