"""Destroy only the completed batch's original expert-owned objects."""
import gc,inspect,threading,weakref
from .common import write

def release_completed(schedule,c,torch,models,output,items,lanes):
 if threading.get_ident()!=c.service.owner:raise RuntimeError('Completed expert release escaped original owner')
 assert len(items)==lanes and all(items)and len({row['case_id']for row in items})==lanes
 assert sum(len(entry['rows'])for entry in schedule.state.values())>=lanes
 shared=inspect.getclosurevars(schedule.pooled.__init__).nonlocals['shared']
 assert isinstance(shared,list)and len(shared)in(0,2)
 references=[weakref.ref(real)for real in shared];task_refs=[weakref.ref(task)for task in schedule.tasks.values()]
 def version():return [[(p.data_ptr(),p._version,str(p.dtype),tuple(p.shape))for p in m.parameters()]for m in models]
 before_params=version();cpu=torch.random.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
 torch.cuda.synchronize();free,total=torch.cuda.mem_get_info();before=int(total-free)
 with c.scene_gate.write():
  schedule.tasks.clear();schedule.pool.clear();shared.clear();collected=gc.collect()
  assert all(ref()is None for ref in references),'Completed original planner still owned'
  assert all(ref()is None for ref in task_refs),'Completed original expert task still owned'
  torch.cuda.empty_cache();torch.cuda.synchronize()
 free,total=torch.cuda.mem_get_info();after=int(total-free)
 assert version()==before_params and torch.equal(cpu,torch.random.get_rng_state())and torch.equal(cuda,torch.cuda.get_rng_state())
 write(output,dict(passed=True,completed_selected_cases=lanes,real_planners_released=len(references),expert_tasks_released=len(task_refs),all_original_expert_planner_and_task_weakrefs_dead=True,device_used_before_bytes=before,device_used_after_bytes=after,GC_objects=collected,original_expert_screen_and_per_task_seed_order_unchanged=True,policy_uses_only_original_stateless_gripper_interpolation=True,models_versions_dtypes_and_Torch_RNG_unchanged=True,no_global_renderer_or_shader_cache_cleared=True,guard_clock_not_reset=True))
