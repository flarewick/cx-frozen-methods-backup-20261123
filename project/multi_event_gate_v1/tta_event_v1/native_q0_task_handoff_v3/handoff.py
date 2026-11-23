"""Owner-only transfer of the actual reset q0 Scene; no tensor/result substitution."""
import gc,json,threading,weakref
from native_expert_case_release_v1.release import shared

def append(path,row):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')

def capture(schedule,task,initial,case_id):
 c=schedule.c;lane=c.rng.current()
 assert threading.get_ident()==c.service.owner and lane is not None
 assert initial['native_action_count']==int(task.take_action_cnt)==0
 if not hasattr(schedule,'q0_pending'):schedule.q0_pending={}
 assert case_id not in schedule.q0_pending
 schedule.q0_pending[case_id]=dict(task=task,initial=initial,rng=c.rng.capture(lane))

def detach(schedule,name,item,torch,models):
 c=schedule.c;assert threading.get_ident()==c.service.owner
 case_id=item['case_id'];pending=schedule.q0_pending[case_id]
 task=schedule.tasks.pop(name);assert task is pending['task']and int(task.take_action_cnt)==0
 real=shared(schedule);identities=[id(p)for p in real];assert len(real)==2
 before=[[(p.data_ptr(),p._version)for p in m.parameters()]for m in models]
 cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()if c.rng.cuda else None
 proxies=schedule.pool.pop(schedule.lanes[name].name);assert len(proxies)==2
 assert task.robot.left_planner is proxies[0]and task.robot.right_planner is proxies[1]
 refs=[weakref.ref(p)for p in proxies]
 task.robot.left_planner=schedule.replay();task.robot.right_planner=schedule.replay();proxies=None;gc.collect()
 assert all(p()is None for p in refs),'Transferred Task retained mutable expert proxy'
 from native_scene_free_topp_v2.solver import detach as detach_world
 with c.scene_gate.write():world_release=detach_world(task)
 assert identities==[id(p)for p in real]
 assert before==[[(p.data_ptr(),p._version)for p in m.parameters()]for m in models]
 assert torch.equal(cpu,torch.get_rng_state())and(cuda is None or torch.equal(cuda,torch.cuda.get_rng_state()))
 append(schedule.root/'q0_task_handoff.jsonl',dict(case_id=case_id,exact_task_and_scene_retained=True,expert_proxy_weakrefs_dead=True,original_shared_planner_identities_preserved=identities,model_versions_and_RNG_unchanged=True,native_action_count=0,old_expert_lane=schedule.lanes[name].name,no_extra_primary_queries_or_physics=True,scene_free_TOPP=world_release))

def schedule_class(original,torch,models):
 class HandoffSchedule(original):
  def next(self,name):
   item=super().next(name)
   with self.serial_screen_lock:self.c.call(self.lanes[name],detach,self,name,item,torch,models)
   return item
 return HandoffSchedule

def take(schedule,case_id,lane):
 c=schedule.c;assert threading.get_ident()==c.service.owner
 pending=getattr(schedule,'q0_pending',{}).pop(case_id,None)
 if pending is None:return None
 task=pending.pop('task');assert task.take_action_cnt==0
 c.rng.restore(lane,pending.pop('rng'))
 return task,pending.pop('initial')
