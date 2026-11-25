"""Unchanged original G foundation predict body, current-device numerical reference only."""
import numpy as np,torch,json
from pathlib import Path
from .common import C,read

def predict(base,head,robots,projection,observed,prompt,seed):
 from cx_tta.official_onset_domain import OfficialOutputSupportPolicy
 from cx_tta.cache_joint_prefix import encode
 from cx_tta.train_output_support import robot_limits
 policy=object.__new__(OfficialOutputSupportPolicy)
 policy.base_policy=base;policy.model=head;policy.robots=robots;policy.projection=projection;policy.condition='M_event_full';policy.encode_eager=torch.compiler.disable(encode);policy.sampler=base._sample_actions;policy.query_recorder=None;policy.force_zero=True;policy.output_mode='bounded';policy.derived_checkpoint_sha256='6e0de463a03d8bac0e160aa46cf955f8575b78ecd38d1e799392e253a7d289e7'
 payload=torch.load(C/'pi05_gt_shared6_v1/frozen_shared6_head_v1/adapter_seed0.pt',map_location='cpu',weights_only=True);policy.joint_limits=robot_limits(json.loads(Path(payload['metadata']['robot_config_path']).read_text()));policy.reset_history();policy.episode_seed=seed
 observation=dict(observation={name:dict(rgb=im)for name,im in zip(('head_camera','right_camera','left_camera'),observed['rgb'])},joint_action=dict(vector=observed['command_state']))
 _,_,base=policy.predict(observation,prompt,None,enabled=True)
 return dict(base=base.copy(),raw_delta=policy.last_numeric['raw_delta'].copy(),legal={k:v.copy()for k,v in policy.last_legal_inputs.items()})
