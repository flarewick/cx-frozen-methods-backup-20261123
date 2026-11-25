"""Keep seen-domain DEV initialization while restoring the official action limit."""
from pathlib import Path
import yaml
from cx_tta_feedback_confirm5.lanes import Coordinator as Original,Lane,gripper_only
class Coordinator(Original):
 def prepare_task(self,task,lane):
  task=super().prepare_task(task,lane);original=task.setup_demo
  def setup(*a,**k):
   value=original(*a,**k)
   if task.step_lim is None:
    limits=yaml.safe_load(Path('task_config/_eval_step_limit.yml').read_text());task.step_lim=int(limits.get(task.task_name,1000));assert task.step_lim>0
   if k.get('eval_mode',False):
    from .official_prefix import consume
    def language():return consume(dict(task=task.task_name,seed=k['seed'],prompt=task._rwc_registered_instruction))
    self.call(lane,language)
   return value
  task.setup_demo=setup;return task
