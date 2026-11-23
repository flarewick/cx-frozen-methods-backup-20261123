"""Scientific source equivalence and independent-output contracts."""
import ast,os
from .common import C,T,O,TASKS,COHORT

def audit():
 from .source_dimensions import original_dimensions
 parent=T/'native_cross_task100_v19'
 unchanged=['preload.py','screen.py','startup.py','cache.py','ordered_screen.py','owner_precision.py','prefetch.py','cached_screen.py']
 for name in unchanged:
  old=(parent/name).read_text().replace('native_cross_task100_v19','native_frozen100_v12').replace('native_transfer_semantics_v1','native_transfer_semantics_v2')
  assert ast.dump(ast.parse(old))==ast.dump(ast.parse(original_dimensions(name,(C/name).read_text()))),name
 old=(parent/'runtime.py').read_text();a=old.index("  if spec['batch_index']==2:");b=old.index('  scheduled[number]',a)
 old=(old[:a]+old[b:]).replace("prior_batch002_identity_exact=spec['batch_index']==2,",'').replace("spec['batch_index']in(0,2)","spec['batch_index']==0")
 old=old.replace('from native_best100_endpoint_v1.episode import execute as episode,CONDITIONS','from .episode import execute as episode,CONDITIONS').replace("spec['conditions']==['M_event_full','TTA']","spec['conditions']==['baseline','M_event_full','TTA']").replace("branches=['M_event_full','zero_update','shadow_update','TTA']if spec['batch_index']==0and","branches=['baseline','M_event_full','zero_update','shadow_update','TTA']if spec['batch_index']==0 and")
 runtime=original_dimensions('runtime.py',(C/'runtime.py').read_text()).replace("\n startup.four.set();startup.eight.set()\n write(root/'initial_real_lanes_admitted.json',dict(lanes=list(range(8)),only_original_real_episode_workers=True,no_extra_queries_or_GPU_compute=True,guard_clock_not_reset=True,unused_allocator_scene_hook_active=True))",'').replace("\n from .scene_cache import install as install_scene_cache\n install_scene_cache(c,torch,models,root/'scene_allocator_reclamation.jsonl')",'')
 assert ast.dump(ast.parse(old))==ast.dump(ast.parse(runtime))
 def infer(path):
  tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='FrozenInstances')
  return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='infer'))
 assert infer(T/'native_transfer_semantics_v1/semantic.py')==infer(T/'native_transfer_semantics_v2/semantic.py')
 assert O==T/'outputs/native_frozen100_v12'/COHORT
 assert TASKS in [('move_stapler_pad','place_mouse_pad'),('place_fan','place_bread_skillet')]
 assert "outputs/native_frozen100_v1"in(C/'preparation.py').read_text()
 assert "['env','CX_TTA100_COHORT='+COHORT"in(C/'host.py').read_text()
 # All deployment parameter values come from the immutable093 parent.
 assert "branches=['baseline','M_event_full','zero_update','shadow_update','TTA']if spec['batch_index']==0"in(C/'runtime.py').read_text()
 assert "native_process_sampler_v7 import guard"in(C/'host.py').read_text()
 assert not any(str(p).startswith(str(T/'outputs/native_cross_task100_v19/batches'))for p in O.glob('**/accepted/*.json'))
 # The independent baseline differs only in condition, capture import, and pure-base action selection.
 baseline=(T/'native_best100_endpoint_v1/episode.py').read_text().replace('from .capture import capture','from native_best100_endpoint_v1.capture import capture').replace("CONDITIONS=('M_event_full','zero_update','shadow_update','TTA','wrong_feedback')","CONDITIONS=('baseline',)")
 baseline=baseline.replace(' composer=EventComposer(event_config);queue=FeedbackQueue();bridge=None',' queue=FeedbackQueue();bridge=None')
 baseline=baseline.replace("   actions,delta,event=composer.step(proposal['base'],proposal['raw_delta'],proposal['legal'],adapted_ARM=direction)","   actions=proposal['base'].copy();delta=np.zeros_like(actions)\n   event=dict(query=q,event=None,evidence={},decision=dict(active=False,selected_event=None),adapted_direction_supplied=False,adaptation_applied=False,baseline_once_base_verbatim=True)")
 assert ast.dump(ast.parse(baseline))==ast.dump(ast.parse((C/'baseline.py').read_text()))
 # Full and TTA have no new episode implementation; dispatch directly to the frozen function.
 from .episode import frozen_execute
 from native_best100_endpoint_v1.episode import execute
 assert frozen_execute is execute
 for name in ('runtime.py','baseline.py','episode.py','analysis.py','preflight.py','preload.py','screen.py','semantic.py','semantic_worker.py','startup.py','cache.py','ordered_screen.py','owner_precision.py','prefetch.py','cached_screen.py'):
  new=original_dimensions(name,(C/name).read_text()).replace('native_frozen100_v12','native_frozen100_v1')
  if name=='runtime.py':new=new.replace("\n startup.four.set();startup.eight.set()\n write(root/'initial_real_lanes_admitted.json',dict(lanes=list(range(8)),only_original_real_episode_workers=True,no_extra_queries_or_GPU_compute=True,guard_clock_not_reset=True,unused_allocator_scene_hook_active=True))",'').replace("\n from .scene_cache import install as install_scene_cache\n install_scene_cache(c,torch,models,root/'scene_allocator_reclamation.jsonl')",'')
  assert ast.dump(ast.parse(new))==ast.dump(ast.parse((T/'native_frozen100_v1'/name).read_text())),name
 return True
