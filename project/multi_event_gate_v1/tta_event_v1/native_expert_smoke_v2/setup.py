"""Original expert scene mode with the unchanged native action budget after setup."""
import copy
from pathlib import Path
import yaml
from cx_tta_feedback_confirm5.throughput import discard_observation

def replay_prefix(task,item,prefix,cfg,observe,seed_all):
 if prefix:raise ValueError('Only original expert q0 reference in this fixed pilot')
 assert cfg['eval_mode']is False
 seed_all(item['seed'])
 task.setup_demo(now_ep_num=item['episode_index'],seed=item['seed'],**dict(cfg,need_plan=False,save_data=False))
 assert task.step_lim is None and task.take_action_cnt==0
 limits=yaml.safe_load((Path(cfg['cx_runtime'])/'task_config/_eval_step_limit.yml').read_text())
 task.step_lim=int(limits[item['task']]);assert task.step_lim>0
 task.set_instruction(instruction=item['instruction']);discard_observation(task)
 return observe()
