"""S2o live heads through frozen legal input/sampler path; no gripper preclip."""
import time
from pathlib import Path
import numpy as np
import torch
from .structured_action_deploy import StructuredActionPolicy,CAPS
from .structured_action_adapter import FORWARD_KEYS
from .output_support_adapter import OutputSupportAdapter,CONDITIONS,validate_native_arms
from .train_output_support import BASE,read_checkpoint,robot_limits
from .audit import digest
from .deploy import POLICY_CAMERA_KEYS,unpack_observation
from .structured_features import pack_observable
from .recovery_contract import rng_capture,rng_fingerprint
from .eval_diagnostics import array_hash,json_hash


class OutputSupportPolicy(StructuredActionPolicy):
    def __init__(self,base,training_run,condition,force_zero=False):
        if condition not in ('L0',*CONDITIONS):raise ValueError('Unknown live cell')
        super().__init__(base,BASE/'S0_seed0.pt','S0' if condition!='L0' else 'L0')
        cell='A_bounded_clipped' if condition=='L0' else condition
        checkpoint=Path(training_run)/(cell+'_seed0.pt');payload=read_checkpoint(checkpoint,cell)
        meta=payload['metadata']
        for key in ('channels','geometry_mean','projection_path','projection_sha256','robot_config_path','robot_config_sha256','parameters'):
            if meta[key]!=self.metadata[key]:raise ValueError('Deployment identity mismatch: '+key)
        self.model=OutputSupportAdapter(32,meta['geometry_mean'],cell)
        self.model.load_state_dict(payload['model'],strict=True);self.model.eval().requires_grad_(False)
        self.joint_limits=robot_limits(__import__('json').loads(Path(meta['robot_config_path']).read_text()))
        self.condition=condition;self.output_mode=meta['output_mode'];self.force_zero=force_zero
        self.derived_checkpoint_sha256=digest(checkpoint)
        self.reset_history()

    def reset_history(self):
        super().reset_history();self.last_numeric=None;self.last_legal_inputs=None

    def predict(self, observation, instruction, noise, *, enabled):
        if noise is not None or enabled != (self.condition != 'L0'):
            raise ValueError('Official internal-noise condition required')
        if not isinstance(instruction, str) or not instruction:
            raise ValueError('Grounded instruction required')
        if torch._dynamo.config.disable or self.base_policy._sample_actions is not self.sampler:
            raise RuntimeError('Compiled sampling interface changed')
        if not hasattr(self.sampler, '_torchdynamo_orig_callable'):
            raise RuntimeError('Official compiled sampler required')
        images, state = unpack_observation(observation)
        state = state.copy()
        # Current-only deterministic auxiliary encoding. No action inference here.
        before = rng_fingerprint(rng_capture())
        cfg = self.base_policy._model.paligemma_with_expert.paligemma.language_model.config
        implementation = cfg._attn_implementation
        started = time.perf_counter()
        try:
            joint, feature_meta = self.encode_eager(self.base_policy, images, state, instruction)
        finally:
            cfg._attn_implementation = implementation
        after_feature = rng_fingerprint(rng_capture())
        if before != after_feature:
            raise RuntimeError('Auxiliary prefix encoding changed sampler RNG')
        if self.base_policy._sample_actions is not self.sampler or torch._dynamo.config.disable:
            raise RuntimeError('Auxiliary extraction changed compiled sampler')
        has_history = self.previous is not None
        old_joint, old_state = self.previous if has_history else (np.zeros_like(joint), np.zeros_like(state))
        structured = pack_observable(joint[None], state[None], old_joint[None], old_state[None],
            np.asarray([has_history]), self.projection, self.robots)
        feature_seconds = time.perf_counter() - started
        policy_obs = {'state': state.copy(), 'prompt': instruction,
            'images': {name: image.transpose(2, 0, 1) for name, image in zip(POLICY_CAMERA_KEYS, images)}}
        # The sole stochastic query, with exactly the official internal-noise call.
        base = np.asarray(self.base_policy.infer(policy_obs, noise=None)['actions'], np.float32).copy()
        after_sampler = rng_fingerprint(rng_capture())
        if base.shape != (32, 14) or not np.isfinite(base).all():
            raise ValueError('Invalid native frozen-policy chunk')
        if before == after_sampler:
            raise RuntimeError('Internal action sampler did not advance RNG')
        tensor_inputs = {key: torch.from_numpy(value.copy()) for key, value in structured.items()}
        with torch.inference_mode():
            result = self.model(tensor_inputs, torch.from_numpy(base[None]), enabled=enabled)
        actions = result['actions'][0].detach().cpu().numpy().copy()
        delta = result['delta'][0].detach().cpu().numpy().copy()
        if actions.shape != base.shape or delta.shape != base.shape:
            raise ValueError('Invalid structured native action shape')
        if not np.isfinite(actions).all() or not np.isfinite(delta).all():
            raise ValueError('Nonfinite structured action')
        if self.output_mode=='bounded' and np.any(np.abs(delta) > CAPS[None] + 1e-6):
            raise ValueError('Structured correction exceeds original native caps')
        raw_delta=delta.copy()
        if self.force_zero:
            actions=base.copy();delta=np.zeros_like(base)
        domain=validate_native_arms(actions,self.joint_limits)
        if self.query_count<2:
            self.last_legal_inputs={k:v.copy() for k,v in structured.items()}
            self.last_legal_inputs['base']=base[None].copy()
        else:self.last_legal_inputs=None
        if not enabled and (not np.array_equal(actions, base) or np.any(delta)):
            raise RuntimeError('L0 bypass is not exact identity')
        if after_sampler != rng_fingerprint(rng_capture()):
            raise RuntimeError('Deterministic action adapter changed RNG')
        if self.base_policy._sample_actions is not self.sampler or torch._dynamo.config.disable:
            raise RuntimeError('Adapter changed compiled sampling interface')
        components = {name: array_hash(image) for name, image in zip(POLICY_CAMERA_KEYS, images)}
        components['command_state'] = array_hash(state)
        diagnostic = dict(query=self.query_count, has_history=has_history,
            history_query=self.query_count - 1 if has_history else None,
            observation_sha256=json_hash(components), observation_components=components,
            structured_input_sha256={key: array_hash(value) for key, value in structured.items()},
            rng_before_query=before, rng_after_feature=after_feature, rng_after_sampler=after_sampler,
            base_chunk_sha256=array_hash(base), action_chunk_sha256=array_hash(actions),
            max_abs_correction=float(np.abs(delta).max()), feature_seconds=feature_seconds,
            base_infer_calls=1, auxiliary_prefix_calls=1, **feature_meta)
        diagnostic.update(evaluated_condition=self.condition,output_mode=self.output_mode,
            derived_checkpoint_sha256=self.derived_checkpoint_sha256,history_policy='real',
            arm_strength=0. if self.condition=='L0' else 1.,gripper_strength=0. if self.condition=='L0' else 1.,
            raw_delta_sha256=array_hash(raw_delta),applied_delta_sha256=array_hash(delta),
            raw_max_abs_correction=float(np.abs(raw_delta).max()),applied_max_abs_correction=float(np.abs(delta).max()),
            robot_domain=domain,history_role='Actual previous32-action query, metadata never model input')
        self.last_numeric=dict(command_state=state.copy(),base=base.copy(),raw_delta=raw_delta.copy(),
            applied_delta=delta.copy(),actions=actions.copy())
        if self.query_recorder is not None:
            self.query_recorder(dict(seed=self.episode_seed, **diagnostic))
        self.last_diagnostics = diagnostic
        self.previous = (joint.copy(), state.copy())
        self.query_count += 1
        return actions, delta, base
