"""Original expert-valid sequential seed stream; never a deployable feature."""
import argparse,importlib,threading
from .common import read,write,T
class Schedule:
 def __init__(self,c,prepared,spec,root,original_planner):
  from concurrent_rng import Lane
  from cx_tta_feedback_confirm5.lanes import gripper_only
  import envs.robot.robot as robot
  from collection_capture_gate import install as capture_install
  from shared6_planner_pool import install as pool_install
  robot.CuroboPlanner=original_planner;self.capture_restore=capture_install(original_planner,c);self.pool=pool_install(c);self.pooled=robot.CuroboPlanner;from native_expert_owner_dispatch_v1.dispatch import route;self.pooled=route(self.pooled,c);self.serial_screen_lock=threading.Lock();self.replay=gripper_only(original_planner);robot.CuroboPlanner=self.replay
  self.c,self.prepared,self.spec,self.root=c,prepared,spec,root;self.tasks={};self.lanes={t:Lane('expert_'+t)for t in spec['tasks']};self.locks={t:threading.Lock()for t in spec['tasks']}
  self.record_bridge=None;self.on_initial=None;self.state={t:dict(spec['initial_screen_state'][t],rows=[])for t in spec['tasks']};self.warmed=set()
 def next(self,name):
  c=self.c;lane=self.lanes[name]
  def operation():
   import numpy as np
   from cx_tta.collect_smoke import make_config
   from cx_tta.evaluate_b2 import seed_all
   from cx_tta.expert_screening import run_expert_plan,ExpertPlanningFailure,planning_failure_record
   from cx_tta.attribution_physics import physical_snapshot
   from cx_tta.eval_diagnostics import json_hash
   from envs.utils.create_actor import UnStableError
   import envs.robot.robot as robot
   phase=getattr(lane,'phase',None);lane.phase='expert_screen'
   cfg=make_config(self.prepared['runtime'],argparse.Namespace(task=name,task_config='demo_randomized'),self.prepared['session']);cfg.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
   def setup(task,**kwargs):
    def owner_setup():
     saved=robot.CuroboPlanner;assert saved is self.replay
     robot.CuroboPlanner=self.pooled
     try:return task.setup_demo(**kwargs)
     finally:robot.CuroboPlanner=saved
    owner_setup._cx_stage='setup';owner_setup._cx_priority=5
    return c.call(lane,owner_setup)
   try:
    if name not in self.tasks:self.tasks[name]=c.call(lane,lambda:c.prepare_task(getattr(importlib.import_module('envs.'+name),name)(),lane))
    task=self.tasks[name];state=self.state[name]
    if name not in self.warmed:
     try:
      c.call(lane,seed_all,state['next_seed']);setup(task,now_ep_num=state['accepted'],seed=state['next_seed'],**dict(cfg,need_plan=True,save_data=False,is_test=True))
     except UnStableError as exc:write(self.root/'screening'/(name+'_warmup.json'),dict(reason=str(exc),phase='discarded_setup_only_no_expert_or_policy_outcome'))
     finally:task.close_env(clear_cache=False)
     self.warmed.add(name)
    while True:
     seed=state['next_seed'];index=state['accepted'];assert self.spec['actual_seed_start']<=seed<self.spec['actual_seed_start']+self.spec['max_screen_attempts_per_task']and index<self.spec['episodes_per_task']
     row=dict(task=name,seed=seed,episode_index=index,eligible=False)
     try:
      c.call(lane,seed_all,seed);setup(task,now_ep_num=index,seed=seed,**dict(cfg,need_plan=True,save_data=False,is_test=True))
      native=physical_snapshot(task);assert native['usable'];row['initial_physical_sha256']=json_hash(native)
      from .expert_reference import record,finalize
      self.record_bridge.begin_case('expert/'+name+':'+str(seed),lane.name)
      try:info,reference_receipt=record(task,lambda:run_expert_plan(task),root=self.root,case_id=name+':'+str(seed))
      finally:
       audited=self.record_bridge.finish_case();write(self.root/'expert_readout_bridges'/(name+':'+str(seed)+'.json'),audited)
      assert audited['native_calls']==reference_receipt['actual_expert_physics_steps']
      row.update(expert_plan_success=bool(task.plan_success),expert_success=bool(task.check_success()));row['eligible']=row['expert_plan_success']and row['expert_success']
      finalize(self.root,name+':'+str(seed),reference_receipt,expert_plan_success=row['expert_plan_success'],expert_task_success=row['expert_success'])
      if row['eligible']:row['scene_info']=info['info']
      else:row['reason']='expert_plan_or_success_failed'
     except UnStableError as exc:row.update(reason='unstable_scene',error=str(exc))
     except ExpertPlanningFailure as exc:row.update(planning_failure_record(exc))
     finally:task.close_env(clear_cache=False)
     state['rows'].append(row);state['next_seed']+=1
     write(self.root/'screening'/name/('%d.json'%seed),row)
     if not row['eligible']:continue
     # Exact official unseen-language generator on the re-created accepted q0.
     try:
      c.call(lane,seed_all,seed);setup(task,now_ep_num=index,seed=seed,**dict(cfg,need_plan=True,save_data=False,is_test=True))
      assert json_hash(physical_snapshot(task))==row['initial_physical_sha256']
      from importlib.util import spec_from_file_location,module_from_spec
      module_spec=spec_from_file_location('best100_official_language',self.prepared['runtime']/'description/utils/generate_episode_instructions.py');language=module_from_spec(module_spec);module_spec.loader.exec_module(language)
      from cx_tta import recovery_contract as recovery
      pre_language_rng=c.call(lane,recovery.rng_capture) if self.on_initial is not None else None
      values=language.generate_episode_descriptions(name,[row['scene_info']],100);instruction=str(np.random.choice(values[0]['seen']))
      if self.on_initial is not None:
       post_language_rng=c.call(lane,recovery.rng_capture)
       try:
        # The real policy re-seeds/setup then discards before language draws.
        # Restore only within this diagnostic read, then restore language state.
        c.call(lane,recovery.rng_restore,pre_language_rng)
        c.call(lane,self.on_initial,task,instruction,name,seed,index)
       finally:c.call(lane,recovery.rng_restore,post_language_rng)
     finally:task.close_env(clear_cache=False)
     item=dict(task=name,seed=seed,episode_index=index,instruction=instruction,case_id=name+':'+str(seed),initial_physical_sha256=row['initial_physical_sha256']);state['accepted']+=1
     write(self.root/'accepted'/(item['case_id']+'.json'),item);return item
   finally:lane.phase=phase
  operation._cx_priority=8;operation._cx_stage='official_expert_valid_screen'
  with self.serial_screen_lock,self.locks[name],c.rng.bind(lane):return operation()
 def finish(self):
  write(self.root/'screening_complete.json',dict(complete=True,state={t:{k:v for k,v in s.items()if k!='rows'}for t,s in self.state.items()},attempts={t:len(s['rows'])for t,s in self.state.items()}))
