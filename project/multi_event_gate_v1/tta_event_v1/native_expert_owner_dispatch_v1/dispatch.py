"""Route individual original expert planner methods, never expose GPU state."""
import threading
METHODS=frozenset(('plan_path','plan_batch','update_point_cloud'))
def route(pooled,c):
 original=pooled.__getattribute__;init=pooled.__init__;setter=pooled.__setattr__
 def get(self,name):
  if threading.get_ident()==c.service.owner or name.startswith('_cx_')or name in('__class__','__dict__','plan_grippers'):
   return original(self,name)
  lane=c.rng.current()
  if lane is None or getattr(lane,'phase',None)!='expert_screen':raise RuntimeError('Planner dispatch outside actual expert worker lane')
  if name not in METHODS:raise RuntimeError('Mutable planner attribute cannot escape owner: '+name)
  def call(*args,**kwargs):
   def execute():return original(self,name)(*args,**kwargs)
   execute._cx_stage='original_expert_'+name
   return c.call(lane,execute)
  return call
 pooled.__getattribute__=get
 assert pooled.__init__ is init and pooled.__setattr__ is setter
 return pooled
