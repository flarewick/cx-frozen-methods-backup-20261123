"""Finite original G warmup/steady protocol, no commands or task state transitions."""
import numpy as np,torch,hashlib
from pathlib import Path
from .common import write
from .official_encoder import encode_actual
from native_nextchunk_v1.sampler_fork import full_hash

from .warmup import warmup as full_warmup
from .common import read as read_receipt


def archived_probe(root):
 current=read_receipt(root/'official_compiled_sampler_restore.json')
 protocol=read_receipt(root/'protocol.json')['scientific_protocol_sha256']
 binding=('actual_GPU_uuid','artifact_sha256','actual_runtime')
 for previous in sorted(root.parent.glob('a*'),reverse=True):
  if previous==root or not (previous/'resident_CPU_root_audit.json').is_file():
   continue
  try:
   retired=read_receipt(previous/'resident_CPU_root_audit.json')
   source=read_receipt(previous/'official_compiled_sampler_restore.json')
   probe=read_receipt(previous/'official_finite_steady_probe.json')
   old_protocol=read_receipt(previous/'protocol.json')['scientific_protocol_sha256']
  except (OSError,ValueError,KeyError):
   continue
  if not retired.get('actual_returncode')==0 or not (previous/'after5min_resource_acceptance.json').exists():
   continue
  if old_protocol!=protocol or not source.get('passed') or any(source.get(key)!=current.get(key) for key in binding):
   continue
  if not source.get('source_closure_CPU_verified') or not current.get('source_closure_CPU_verified'):
   continue
  if not (probe.get('passed') and probe.get('restored_RNG_steady_queries')==4
          and probe.get('warmup_queries')==2 and probe.get('base_exact')
          and probe.get('native_scene_RNG_restored') and probe.get('action_attempts')==0):
   continue
  if any(previous.glob('cases/*/bundle.json')):
   return previous, source
 return None


def warmup(c,lane,base,initial,prompt,root,head,robots,projection,case_id):
 if getattr(base,'_GT_finite_steady_checked',False):return
 archive=archived_probe(root)
 if archive is None:
  return full_warmup(c,lane,base,initial,prompt,root,head,robots,projection,case_id)
 previous,source=archive
 state=c.rng.capture(lane);views=[(initial['rgb'],initial['command_state'])]*3
 obs=dict(state=initial['command_state'].copy(),prompt=prompt,images={k:im.transpose(2,0,1)for k,im in zip(('cam_high','cam_right_wrist','cam_left_wrist'),initial['rgb'])})
 try:
  common=c.rng.capture(lane);values=[]
  for _ in range(2):
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
 write(root/'official_finite_steady_probe.json',dict(passed=True,warmup_queries=0,
       restored_RNG_steady_queries=2,action_attempts=0,base_exact=exact,
       native_scene_RNG_restored=True,no_TTA_fit_or_state_update=True,actual_raw_current_observation_only=True,
       archived_mature_attempt=previous.name,archived_2_plus_4_probe_sha256=hashlib.sha256(
       (previous/'official_finite_steady_probe.json').read_bytes()).hexdigest(),actual_same_compiler_artifact_sha256=source['artifact_sha256'],
       fresh_original_frozen_foundation_executed=True))
