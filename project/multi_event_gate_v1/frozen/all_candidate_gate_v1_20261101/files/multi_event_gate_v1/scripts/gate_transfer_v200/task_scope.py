"""Task-specific runtime certification; gate/candidate inference is untouched."""
import ast,functools,inspect,json
from pathlib import Path
_INSTALLED=False

def install():
 global _INSTALLED
 if _INSTALLED:return
 import benefit_followup_protocol_v2 as p
 from evaluate import host
 plan=json.loads((p.TRAINING/'training_plan.json').read_text());dataset=host(plan['dataset']);protocol=json.loads((dataset/'protocol.json').read_text());runtime=Path(protocol['runtime_source'])
 if not runtime.exists():runtime=p.D/'.work/rootfs'/runtime.relative_to('/')
 base=ast.parse((runtime/'envs/_base_task.py').read_text());init=next(n for n in ast.walk(base)if isinstance(n,ast.FunctionDef)and n.name=='_init_task_env_');params=init.args.args;defaults=dict(zip([a.arg for a in params[-len(init.args.defaults):]],init.args.defaults));assert ast.literal_eval(defaults['table_xy_bias'])==[0,0]
 task_hash={}
 for name in p.TASKS:
  file=runtime/'envs'/(name+'.py');tree=ast.parse(file.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==name);assert [ast.unparse(b)for b in cls.bases]==['Base_Task']
  assert not any(isinstance(n,ast.FunctionDef)and n.name in ('get_obs','take_action','_update_render','get_cluttered_table','_init_task_env_')for n in cls.body)
  setup=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='setup_demo');body=[n for n in setup.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str))]
  assert len(body)==1 and isinstance(body[0],ast.Expr);call=body[0].value;assert ast.unparse(call.func)=='super()._init_task_env_'and not call.args and len(call.keywords)==1 and call.keywords[0].arg is None
  success=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='check_success');assert not any(isinstance(n,ast.Attribute)and any(v in n.attr for v in ('camera','render','now_obs','get_obs'))for n in ast.walk(success))
  task_hash[name]=p.sha(file)
 oldproof=p.D/'reports/target_shared6_render_sync_cpu_20261024.json';proof=json.loads(oldproof.read_text());control=p.D/'reports/target_collection_control_cpu_20261024.json';cv=json.loads(control.read_text());assert proof['passed']and cv['passed']and proof['helper_sha256']==p.sha(p.D/'shared6_render_sync.py')and proof['prefix_helper_sha256']==p.sha(p.D/'render_sync_materialization.py')
 import benefit_pilot_scene_bounds as bounds
 for task in p.TASKS:
  if task in bounds.BIAS:assert bounds.BIAS[task]==(0.,0.)
  else:bounds.BIAS[task]=(0.,0.)
 # Original renderer prefix equivalence proof + actual inherited-method and task-source checks.
 import shared6_render_sync as visual
 original=visual.install
 @functools.wraps(original)
 def certified(task,note=lambda *a,**kw:None,verify_proof=True):
  name=type(task).__name__
  if name not in task_hash:return original(task,note,verify_proof)
  assert p.sha(Path(inspect.getsourcefile(type(task))))==task_hash[name]
  file=Path(inspect.getsourcefile(inspect.unwrap(task._update_render)));assert p.sha(file)==cv['instrumented_base_sha256']
  assert p.sha(oldproof)==proof_sha and p.sha(control)==control_sha
  return original(task,note,verify_proof=False)
 proof_sha=p.sha(oldproof);control_sha=p.sha(control);visual.install=certified
 # New tasks have no expert visual replay certificate. Preserve complete original expert rendering.
 import benefit_expert_visual_sync_v13 as expert
 before=expert.install
 @functools.wraps(before)
 def full_expert(task,*args,**kwargs):
  if type(task).__name__ in task_hash:return None
  return before(task,*args,**kwargs)
 expert.install=full_expert
 value=dict(passed=True,CPU_only=True,task_source_sha256=task_hash,raw_base_sha256=p.sha(runtime/'envs/_base_task.py'),instrumented_base_sha256=cv['instrumented_base_sha256'],original_renderer_CPU_proof_sha256=proof_sha,original_control_CPU_proof_sha256=control_sha,actual_new_task_setup_inherits_default_zero_bias=True,actual_new_success_has_no_visual_reads=True,actual_new_task_has_no_visual_method_overrides=True,policy_visual_deferral_certified_by_same_original_AST_proof=True,expert_visual_deferral_disabled_for_uncertified_new_tasks=True,original_full_expert_visual_updates_preserved=True,helper_sha256=p.sha(__file__))
 target=p.ROOT/'transfer_task_runtime_CPU.json'
 if target.exists():assert json.loads(target.read_text())==value
 else:p.atomic(target,value)
 _INSTALLED=True
