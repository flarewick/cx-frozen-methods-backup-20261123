"""New-study scene-local defaults; physical index is always zero by protocol."""
import functools,inspect
from benefit_pilot_scene_bounds import DEFAULT,BIAS

def install(task,name):
 original=task.get_cluttered_table;sig=inspect.signature(original)
 if any(sig.parameters[k].default!=v for k,v in DEFAULT.items()):raise RuntimeError('Original global scene bounds already contaminated')
 @functools.wraps(original)
 def clutter(*args,**kwargs):
  if args or any(k in kwargs for k in DEFAULT):raise RuntimeError('Unexpected explicit clutter arguments')
  if tuple(task.table_xy_bias)!=BIAS[name]:raise RuntimeError('Task bias changed')
  return original(**kwargs,**{k:list(v)for k,v in DEFAULT.items()})
 task.get_cluttered_table=clutter
