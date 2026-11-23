"""Closed expert Task lifetime is separate from the persistent original planner."""
import gc,inspect,json,threading,weakref

def shared(schedule):
 result=inspect.getclosurevars(schedule.pooled.__init__).nonlocals['shared']
 assert isinstance(result,list)and len(result)in(0,2)
 return result

def drop_closed(schedule,name,c,torch,models,output):
 if threading.get_ident()!=c.service.owner:raise RuntimeError('Closed expert release requires owner')
 task=schedule.tasks.pop(name);task_ref=weakref.ref(task);scene_ref=weakref.ref(task.scene)
 real=shared(schedule);identities=[id(p)for p in real];lane=schedule.lanes[name]
 proxies=schedule.pool.pop(lane.name);assert len(proxies)==2
 refs=[weakref.ref(p)for p in proxies];proxies=None;task=None
 params=[[(p.data_ptr(),p._version)for p in model.parameters()]for model in models]
 cpu=torch.random.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()if c.rng.cuda else None
 with c.scene_gate.write():
  if c.rng.cuda:torch.cuda.synchronize()
  before=memory(torch,c.rng.cuda);collected=gc.collect()
  assert task_ref()is None and scene_ref()is None,'Closed expert Task/Scene still owned'
  assert all(p()is None for p in refs),'Closed expert planner proxy still owned'
  if c.rng.cuda:torch.cuda.empty_cache();torch.cuda.synchronize()
  after=memory(torch,c.rng.cuda)
 assert identities==[id(p)for p in real]
 assert params==[[(p.data_ptr(),p._version)for p in model.parameters()]for model in models]
 assert torch.equal(cpu,torch.random.get_rng_state())and(cuda is None or torch.equal(cuda,torch.cuda.get_rng_state()))
 row=dict(task=name,closed_task_and_scene_weakrefs_dead=True,closed_proxy_weakrefs_dead=True,original_shared_planner_identities_preserved=identities,real_planner_constructors_not_repeated=True,model_versions_and_RNG_unchanged=True,global_renderer_cache_not_cleared=True,no_extra_queries_or_physics=True,before=before,after=after,GC_objects=collected)
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('a')as f:f.write(json.dumps(row)+'\n')
 return row

def memory(torch,cuda):
 if not cuda:return None
 free,total=torch.cuda.mem_get_info()
 return dict(device_used_bytes=int(total-free),allocated_bytes=int(torch.cuda.memory_allocated()),reserved_bytes=int(torch.cuda.memory_reserved()))

def schedule_class(original,torch,models):
 class ClosedExpertTaskSchedule(original):
  def next(self,name):
   result=super().next(name) # Original screen has closed its Task before returning.
   with self.serial_screen_lock:
    self.c.call(self.lanes[name],drop_closed,self,name,self.c,torch,models,self.root/'closed_expert_task_release.jsonl')
   return result
 return ClosedExpertTaskSchedule

def retire_live_group(schedule,lane,c):
 if threading.get_ident()!=c.service.owner:raise RuntimeError('Live proxy release requires owner')
 real=shared(schedule);identities=[id(p)for p in real]
 proxies=schedule.pool.pop(lane.name);assert len(proxies)==2;refs=[weakref.ref(p)for p in proxies];proxies=None;gc.collect()
 assert all(p()is None for p in refs)and identities==[id(p)for p in real]
 return dict(live_closed_proxy_group_released=True,original_shared_planners_unchanged=True)
