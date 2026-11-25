"""Finite original G warmup/steady protocol, no commands or task state transitions."""
import numpy as np,torch,hashlib
from .common import write
from .official_encoder import encode_actual
from native_nextchunk_v1.sampler_fork import full_hash

def warmup(c,lane,base,initial,prompt,root,head,robots,projection,case_id):
 if getattr(base,'_GT_finite_steady_checked',False):return
 state=c.rng.capture(lane);views=[(initial['rgb'],initial['command_state'])]*3
 obs=dict(state=initial['command_state'].copy(),prompt=prompt,images={k:im.transpose(2,0,1)for k,im in zip(('cam_high','cam_right_wrist','cam_left_wrist'),initial['rgb'])})
 try:
  for _ in range(2):encode_actual(base,views,prompt);base.infer(obs,noise=None)
  common=c.rng.capture(lane);values=[]
  for _ in range(4):
   c.rng.restore(lane,common);encode_actual(base,views,prompt);values.append(np.asarray(base.infer(obs,noise=None)['actions'],np.float32).copy())
  exact=all(np.array_equal(v,values[0])for v in values);assert exact,'Official sampler no-action steady repeat mismatch'
  artifact=torch.compiler.save_cache_artifacts()
  if artifact is not None:
   data,info=artifact;(root/'compiled_diagnostic_artifacts.bin').write_bytes(data);write(root/'compiled_diagnostic_export.json',dict(artifact_sha256=hashlib.sha256(data).hexdigest(),original_sampler_source_and_weights_unchanged=True,actual_no_action_steady_exact=True,scope='Compiler cache export only; historical-G q0 and full-trace controls still required',base_step0_parameters_not_updated=True))
  # Execute the unchanged original G foundation predict body on this device with identical current inputs/RNG.
  from cx_tta.official_onset_domain import OfficialOutputSupportPolicy
  from cx_tta.cache_joint_prefix import encode
  from cx_tta.train_output_support import robot_limits
  from .common import C,read
  import json
  policy=object.__new__(OfficialOutputSupportPolicy)
  policy.base_policy=base;policy.model=head;policy.robots=robots;policy.projection=projection;policy.condition='M_event_full';policy.encode_eager=torch.compiler.disable(encode);policy.sampler=base._sample_actions;policy.query_recorder=None;policy.force_zero=True;policy.output_mode='bounded';policy.derived_checkpoint_sha256='6e0de463a03d8bac0e160aa46cf955f8575b78ecd38d1e799392e253a7d289e7'
  payload=torch.load(C/'pi05_gt_shared6_v1/frozen_shared6_head_v1/adapter_seed0.pt',map_location='cpu',weights_only=True);policy.joint_limits=robot_limits(json.loads(__import__('pathlib').Path(payload['metadata']['robot_config_path']).read_text()));policy.reset_history();policy.episode_seed=int(case_id.rsplit(':',1)[1])
  observation=dict(observation={name:dict(rgb=im)for name,im in zip(('head_camera','right_camera','left_camera'),initial['rgb'])},joint_action=dict(vector=initial['command_state']))
  torch.cuda.manual_seed(int(case_id.rsplit(':',1)[1]));c.rng.restore(lane,state);_,_,actual_base=policy.predict(observation,prompt,None,enabled=True)
  ref=read(C/'pi05_gt_shared6_v1/original_G_query0_manifest_v1.json')['records'][case_id]
  with np.load(ref['query0_file'],allow_pickle=False)as z:old_base=z['base'].copy();old_raw=z['raw_delta'].copy()
  np.savez_compressed(root/'fresh_original_G_foundation_query0.npz',base=actual_base,raw_delta=policy.last_numeric['raw_delta'],**{'legal__'+k:v for k,v in policy.last_legal_inputs.items()})
  write(root/'fresh_original_G_foundation_query0.json',dict(original_frozen_predict_body=True,head_weights_current_shared6=True,sampler_unchanged=True,no_policy_actions=True,historical_base_byte_exact=np.array_equal(actual_base,old_base),historical_base_max_abs=float(np.max(np.abs(actual_base-old_base))),historical_head_byte_exact=np.array_equal(policy.last_numeric['raw_delta'],old_raw),original_diagnostics=policy.last_diagnostics))
  base._GT_finite_steady_checked=True
 finally:c.rng.restore(lane,state)
 assert full_hash(c.rng.capture(lane))==full_hash(state)
 write(root/'official_finite_steady_probe.json',dict(passed=True,warmup_queries=2,restored_RNG_steady_queries=4,action_attempts=0,base_exact=exact,native_scene_RNG_restored=True,no_TTA_fit_or_state_update=True,actual_raw_current_observation_only=True))
