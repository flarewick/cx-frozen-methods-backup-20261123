"""CPU-only stage ordering, cancellation, completed-query gate and exact return."""
import ast,threading,time
from pathlib import Path
from .startup import StagedLanes,actor_class
from .common import C,T,O,write,sha

def main():
 rows=[];cancel=threading.Event();g=StagedLanes(cancel,rows.append);assert g.wait(0)and g.wait(1)
 ready=[];threads=[threading.Thread(target=lambda i=i:ready.append((i,g.wait(i))))for i in range(2,8)]
 for t in threads:t.start()
 time.sleep(.1);assert not ready
 assert not g.release('actual_once_policy_query_complete',lane=0,query=1,tick=32)
 try:g.release('timer_or_filler',lane=0)
 except ValueError:pass
 else:raise AssertionError('Timer/filler gate accepted')
 try:g.release('actual_case_terminal',lane=2)
 except RuntimeError:pass
 else:raise AssertionError('Out-of-order stage accepted')
 returned=threading.Event();entered=threading.Event();original_result=object();calls=[]
 class Original:
  def query(self,observed,record,folder):
   calls.append((observed,record,folder));entered.set();assert returned.wait(2);return original_result
 Wrapped=actor_class(Original,g,{'saved_case':0});result=[]
 q=threading.Thread(target=lambda:result.append(Wrapped().query(object(),dict(query=0,global_tick=0),Path('/tmp/saved_case/M_event_full'))));q.start();assert entered.wait(1);assert not g.four.is_set()and not ready
 returned.set();q.join(1);assert result==[original_result]and result[0]is original_result and len(calls)==1
 time.sleep(.1);assert sorted(ready)==[(2,True),(3,True)]and len(rows)==1 and not g.eight.is_set()
 assert not g.release('actual_once_policy_query_complete',lane=1,query=0,tick=0)
 assert g.release('actual_once_policy_query_complete',lane=2,query=0,tick=0)
 for t in threads:t.join(1)
 assert not any(t.is_alive()for t in threads)and sorted(ready)==[(i,True)for i in range(2,8)]and len(rows)==2
 cancel2=threading.Event();g2=StagedLanes(cancel2,lambda _:None);out=[];t=threading.Thread(target=lambda:out.append(g2.wait(7)));t.start();cancel2.set();t.join(1);assert out==[False]and not g2.wait(0)
 fallback=[];g3=StagedLanes(threading.Event(),fallback.append);assert g3.release('actual_case_terminal',lane=0)and g3.wait(2);assert g3.release('actual_case_terminal',lane=3)and g3.wait(7)
 def run_case(f):return next(n for n in ast.walk(ast.parse(f.read_text()))if isinstance(n,ast.FunctionDef)and n.name=='run_case')
 new_case=run_case(C/'runtime.py');old_case=run_case(T/'native_cross_task100_v7/runtime.py')
 new_branch=next(n for n in ast.walk(new_case)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='branches'for t in n.targets));old_branch=next(n for n in ast.walk(old_case)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='branches'for t in n.targets));assert isinstance(new_branch.value,ast.IfExp);new_branch.value.test=old_branch.value.test
 assert ast.dump(new_case)==ast.dump(old_case)
 for name in ['semantic.py','semantic_worker.py','preflight.py','preload.py','analysis.py']:
  old=(T/'native_cross_task100_v7'/name).read_text();new=(C/name).read_text().replace('native_cross_task100_v17','native_cross_task100_v7');assert ast.dump(ast.parse(old))==ast.dump(ast.parse(new)),name
 write(O/'CPU_startup_checks.json',dict(passed=True,stages=[2,4,8],actual_original_query_must_return_before_release=True,no_timer_filler_release=True,single_release_per_stage=True,cancel_stops_waiters=True,actual_case_terminal_fallback=True,original_query_called_once_and_output_object_identical=True,original_entire_run_case_AST_exact_except_additional_whole_zero_shadow_controls=True,original_four_CPU_semantic_and_preflight_analysis_AST_exact=True,GPU_calls=0,source_sha256={str(p):sha(p)for p in C.glob('*.py')}))
 print('STAGED_2_4_8_REAL_QUERY_CPU_PASS',flush=True)
if __name__=='__main__':main()
