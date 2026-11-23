"""Time only original native methods; no alternate init/render/physics/RNG."""
import time,weakref,json,threading
METHODS=('setup_scene','create_table_and_wall','load_robot','load_camera','together_open_gripper','load_actors','get_cluttered_table','check_stable')
def install(task,path):
 ref=weakref.ref(task);name=type(task).__name__
 for method in METHODS:
  fn=getattr(type(task),method)
  def timed(*a,_fn=fn,_method=method,**k):
   before=time.monotonic_ns()
   try:return _fn(ref(),*a,**k)
   finally:
    row=dict(task_type=name,method=_method,duration_s=(time.monotonic_ns()-before)/1e9,thread=threading.get_ident(),original_function_once=True,profile_only=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a')as f:f.write(json.dumps(row)+'\n')
  setattr(task,method,timed)
 return task
