import ast
from .common import C,T
RULES={
'runtime.py':[(" prescheduled=[schedule.next(spec['tasks'][n%2])for n in range(lanes)]"," prescheduled=[schedule.next(name)for name in spec['task_slots']]"),(" from .prefetch import PrefetchClient"," from .prefix_reference import REFERENCE as REFERENCE_ROOT\n from .prefetch import PrefetchClient"),("     if diagnostic is None and spec['batch_index']==8:","     if diagnostic is None and (REFERENCE_ROOT/'cases'/item['case_id']/'baseline/query000_environment_audit.json').exists():")],
'analysis.py':[("spec['cases_per_batch']//2 for t in spec['tasks']","spec['task_slots'].count(t) for t in spec['tasks']")],
'prefix_reference.py':[(" if spec['batch_index']!=8:return"," if not spec.get('task_slots'):return"),("  newcase=root/'cases'/oldcase.name","  newcase=root/'cases'/oldcase.name\n  if not newcase.exists():continue"),(" assert cases==10 and checked>=10"," assert checked>=cases")],
}
RULES['preload.py']=[(" prepared=prepare(spec,root)\n try:"," prepared=prepare(spec,root)\n from native_cpu_topp_v1.runtime import prepare as prepare_TOPP,close as close_TOPP\n pool=prepare_TOPP(root,write)\n try:"),("  if code:raise SystemExit(code)","  if code:raise SystemExit(code)\n  close_TOPP(pool,root,write)"),(" finally:prepared['storage'].cleanup()"," finally:\n  pool.close();prepared['storage'].cleanup()")]
def audit():
 for name in ['runtime.py','baseline.py','episode.py','analysis.py','preflight.py','preload.py','screen.py','semantic.py','semantic_worker.py','startup.py','cache.py','owner_precision.py','prefetch.py','cached_screen.py','initial_integrity.py','scene_cache.py','forward_cache.py','completed_planners.py','prefix_reference.py']:
  new=(C/name).read_text().replace('native_frozen100_v22','native_frozen100_v13')
  for old,value in reversed(RULES.get(name,[])):
   assert new.count(value)==1,(name,value)
   new=new.replace(value,old)
  assert ast.dump(ast.parse(new))==ast.dump(ast.parse((T/'native_frozen100_v13'/name).read_text())),name
 return True
