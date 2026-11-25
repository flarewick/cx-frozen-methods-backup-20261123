"""Fixed independent DEV list or previously registered official EVAL list; no outcome filtering."""
import argparse,importlib,threading
from .common import read,write
class Schedule:
 def __init__(self,c,prepared,spec,root,original_planner):
  from concurrent_rng import Lane
  from cx_tta_feedback_confirm5.lanes import gripper_only
  import envs.robot.robot as robot
  robot.CuroboPlanner=gripper_only(original_planner)
  self.c,self.prepared,self.spec,self.root=c,prepared,spec,root;self.lanes={t:Lane('registered_'+t)for t in spec['tasks']};self.locks={t:threading.Lock()for t in spec['tasks']};self.items={t:iter([x for x in spec['registered_cases']if x['task']==t])for t in spec['tasks']};self.on_initial=None
 def next(self,name):
  c=self.c;lane=self.lanes[name]
  with self.locks[name]:item=next(self.items[name],None)
  if item is None:return None
  def setup():
   from cx_tta.collect_smoke import make_config
   from cx_tta.evaluate_b2 import seed_all
   from native_expert_smoke_v2.observe import observe
   from cx_tta.attribution_physics import physical_snapshot
   from cx_tta.eval_diagnostics import json_hash
   cfg=make_config(self.prepared['runtime'],argparse.Namespace(task=name,task_config='demo_randomized'),self.prepared['session']);cfg.update(eval_mode=self.spec['phase']=='EVAL',save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
   task=c.prepare_task(getattr(importlib.import_module('envs.'+name),name)(),lane);task._rwc_registered_instruction=item['instruction']
   try:
    seed_all(item['seed']);task.setup_demo(now_ep_num=item['episode_index'],seed=item['seed'],**dict(cfg,need_plan=True,save_data=False,is_test=True))
    snapshot=physical_snapshot(task)
    write(self.root/'initial_diagnostics'/(item['case_id']+'.json'),snapshot)
    assert snapshot['usable'],snapshot['coverage']['errors']
    physical=json_hash(snapshot)
    if item.get('initial_physical_sha256'):assert physical==item['initial_physical_sha256']
    selected=dict(item,initial_physical_sha256=physical)
    if self.on_initial:self.on_initial(task,item['instruction'],name,item['seed'],item['episode_index'])
    write(self.root/'accepted'/(item['case_id']+'.json'),selected);return selected
   finally:task.close_env(clear_cache=False)
  return c.call(lane,setup)
 def finish(self):write(self.root/'screening_complete.json',dict(complete=True,no_expert_rescreen=True,phase=self.spec['phase'],cases=self.spec['registered_cases']))
