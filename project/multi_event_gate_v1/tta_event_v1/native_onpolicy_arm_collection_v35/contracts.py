from pathlib import Path
import ast,threading
C=Path(__file__).resolve().parent;OLD=C.parent/'native_onpolicy_full_v1'
def fn(s,name):return next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name==name)
def check():
 expected=ast.parse("from native_arm_phase_feedback_v1.episode import execute as original_execute\nfrom native_q0_policy_overlap_v1.episode import make_execute\nexecute=make_execute(original_execute)\nCONDITIONS=('M_event_full',)")
 actual=ast.parse((C/'episode.py').read_text());actual.body=[n for n in actual.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str))];assert ast.dump(actual)==ast.dump(expected)
 old=(OLD/'host.py').read_text();new=(C/'host.py').read_text();from native_process_missing_guard_v2.terminal import terminal_ast_exact
 assert terminal_ast_exact(fn(new,'terminal'),fn(old,'terminal'))
 for p in C.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
 from .ordered_screen import OrderedScreen
 cancel=threading.Event();seen=[]
 def select(task):seen.append(task);return task
 tasks=['a','b','c','d','e'];queue=OrderedScreen(select,tasks,cancel)
 for number in range(25):assert queue.next(number)==tasks[number%5]
 assert seen==tasks*5 and all(v==25+i for i,v in enumerate(queue.next_index.values()))
 runtime=(C/'runtime.py').read_text();assert 'prescheduled=None'in runtime and 'actual_native_full_episodes'not in runtime and 'lanes in(4,8,9,10)'in runtime
 print('ORIGINAL_FULL_AND_STRICT_AVERAGE_LIMITS_WITH_EXPLICIT_IDLE_ZERO_BOUND_PASS',flush=True)
