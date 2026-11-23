"""Freeze each qualified scene's original clutter bounds; no shared-list drift.

Original Base_Task.get_cluttered_table mutates its default xlim/ylim lists.
The dump qualification ran alone and called it once per chronological attempt.
Reconstruct those exact BEFORE-call lists, using sequential float additions.
Never alter original source/defaults, physics, random calls or scene membership.
"""
import functools,inspect
DEFAULT=dict(xlim=[-.59,.59],ylim=[-.34,.34],zlim=[.741])
BIAS={'lift_pot':(0.,0.),'dump_bin_bigbin':(.3,0.)}
def before_bounds(task,seed,attempts):
 index=next((i for i,r in enumerate(attempts)if r['seed']==seed),0)
 bounds={k:list(v)for k,v in DEFAULT.items()};bias=BIAS[task]
 for _ in range(index):
  for key,offset in zip(('xlim','ylim'),bias):
   for i in(0,1):bounds[key][i]+=offset
 return bounds

def contract(task,attempts):
 return dict(schema=1,original_mutable_default_side_effect='xlim/ylim += table_xy_bias before random draw',reconstruct='per-scene BEFORE-call lists from chronological qualification attempt ordinal; sequential original float additions',original_table_bias=list(BIAS[task]),scenes={str(r['seed']):before_bounds(task,r['seed'],attempts)for r in attempts if r['eligible']},original_algorithm_unchanged=True,global_defaults_never_mutated=True)

def install(task,name,attempts):
 original=task.get_cluttered_table;signature=inspect.signature(original)
 if any(signature.parameters[k].default!=v for k,v in DEFAULT.items()):raise RuntimeError('Original clutter defaults already changed before isolation')
 @functools.wraps(original)
 def clutter(*args,**kwargs):
  if args or any(k in kwargs for k in DEFAULT):raise RuntimeError('Unexpected explicit clutter bounds in original setup')
  if not hasattr(task,'_pilot_scene_seed'):raise RuntimeError('Missing scoped original initialization seed')
  expected=BIAS[name]
  if tuple(task.table_xy_bias)!=expected:raise RuntimeError('Task table bias changed')
  return original(**kwargs,**before_bounds(name,task._pilot_scene_seed,attempts))
 task.get_cluttered_table=clutter
