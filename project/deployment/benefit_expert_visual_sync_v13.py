"""Defer unobserved visual tails during original expert play_once only.

Every original light/RNG prefix runs, every get_obs executes the complete
original update, and dense control/physics/planning functions stay unchanged.
"""
import functools,inspect,json,time,contextlib
from pathlib import Path
from render_sync_materialization import prefix_method

def install(task,note=lambda *a,**k:None,verify=True,gate=None):
 if getattr(task,'_cx_expert_visual_sync_v13',False):return
 if task.save_data or task.render_freq or task.eval_video_path is not None:raise RuntimeError('Expert deferral needs original no-save/headless/no-video mode')
 if verify:
  if gate is None:raise RuntimeError('Original scene reader gate required')
  import benefit_followup_protocol_v2 as p
  proof=json.loads((p.D/'reports/target_shared6_render_sync_cpu_20261024.json').read_text());control=json.loads((p.D/'reports/target_collection_control_cpu_20261024.json').read_text())
  if not proof['passed']or not control['passed']:raise RuntimeError('Original render/control proofs required')
  if proof['prefix_helper_sha256']!=p.sha(p.D/'render_sync_materialization.py'):raise RuntimeError('Original render-prefix helper changed')
  name=type(task).__name__
  if name not in ('lift_pot','dump_bin_bigbin')or p.sha(Path(inspect.getsourcefile(type(task))))!=proof['task_source_sha256'][name]:raise RuntimeError('Unverified expert task')
  source=Path(inspect.getsourcefile(inspect.unwrap(task._update_render)))
  if p.sha(source)!=control['instrumented_base_sha256']:raise RuntimeError('Only original instrumented private expert base is allowed')
 original_plan=task.play_once;original_update=task._update_render;original_obs=task.get_obs
 prefix=prefix_method(inspect.unwrap(original_update));inside=materializing=deferred=observations=0
 @functools.wraps(original_plan)
 def plan(*a,**kw):
  nonlocal inside
  if inside:raise RuntimeError('Nested expert plan')
  if task.save_data or task.render_freq or task.eval_video_path is not None:raise RuntimeError('Expert mode changed')
  inside+=1;began=time.monotonic();before=deferred
  try:return original_plan(*a,**kw)
  finally:
   inside-=1
   note('expert_visual_sync_deferred_v13',deferred_in_plan=deferred-before,total_deferred=deferred,observations=observations,plan_seconds=time.monotonic()-began)
 @functools.wraps(original_obs)
 def observation(*a,**kw):
  nonlocal materializing,observations
  materializing+=1;observations+=1
  try:return original_obs(*a,**kw)
  finally:materializing-=1
 @functools.wraps(original_update)
 def update(*a,**kw):
  nonlocal deferred
  if inside and not materializing:
   deferred+=1
   with (gate.read()if gate is not None else contextlib.nullcontext()):return prefix(*a,**kw)
  return original_update(*a,**kw)
 task.play_once=plan;task.get_obs=observation;task._update_render=update;task._cx_expert_visual_sync_v13=True

_INSTALLED=False

def install_mixed_hook():
 global _INSTALLED
 if _INSTALLED:return
 import collection_pool
 original=collection_pool.prepare_task
 @functools.wraps(original)
 def prepare(task,lane,c,native):
  result=original(task,lane,c,native)
  if not lane.name.startswith('mixed_original_expert/'):return result
  setup=task.setup_demo
  @functools.wraps(setup)
  def configured(*a,**kw):
   result=setup(*a,**kw);install(task,c.note,gate=c.scene_gate);return result
  task.setup_demo=configured;return result
 collection_pool.prepare_task=prepare;_INSTALLED=True
