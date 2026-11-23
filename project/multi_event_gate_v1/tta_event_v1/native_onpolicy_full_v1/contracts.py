from pathlib import Path
import ast
C=Path(__file__).resolve().parent;OLD=C.parent/'native_object_expert_train_v5'
def fn(s,name):return next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name==name)
def check():
 for name in('cache.py','cached_screen.py','completed_planners.py','forward_cache.py','initial_integrity.py','ordered_screen.py','owner_precision.py','prefetch.py','preflight.py','python_scheduling.py','scene_cache.py','semantic.py','semantic_worker.py','setup_fairness.py','startup.py','expert_reference.py'):
  assert (C/name).read_text()==(OLD/name).read_text().replace('native_object_expert_train_v5','native_onpolicy_full_v1'),name
 old=(OLD/'host.py').read_text().replace('native_object_expert_train_v5','native_onpolicy_full_v1').replace('CPU_expert_teacher_audit.json','CPU_full_capture_audit.json');new=(C/'host.py').read_text()
 for n in('canonical','wait_file','choose','execute_batch','terminal'):assert ast.dump(fn(old,n),include_attributes=False)==ast.dump(fn(new,n),include_attributes=False),n
 expected=ast.parse("from native_feedback_application_v1.episode import execute\nCONDITIONS=('M_event_full',)")
 actual=ast.parse((C/'episode.py').read_text());actual.body=[n for n in actual.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str))]
 assert ast.dump(actual)==ast.dump(expected),'Frozen full dispatch changed'
 for f in C.rglob('*.py'):ast.parse(f.read_text())
 print('FROZEN_FULL_ONPOLICY_ORIGINAL_ACTION_AND_RESOURCE_CONTRACTS_PASS',flush=True)
