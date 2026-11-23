"""One sampled VLA query and causal observable features for S2j deployment."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audit import digest
from .cache_joint_prefix import encode
from .deploy import POLICY_CAMERA_KEYS, unpack_observation
from .eval_diagnostics import array_hash, json_hash
from .recovery_contract import rng_capture, rng_fingerprint
from .robot_kinematics import RobotFK
from .structured_features import pack_observable

SCOPE = 'S2j structured bounded direction residual'
CAPS = np.asarray([.1] * 6 + [.2] + [.1] * 6 + [.2], np.float32)


def read_checkpoint(path, expected_condition=None):
    """Validate deployable metadata without creating a policy or using CUDA."""
    path = Path(path).resolve()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if set(payload) != {'model', 'metadata'}:
        raise ValueError('Expected S2j model and metadata checkpoint')
    meta = payload['metadata']
    if meta['scope'] != SCOPE or meta['condition'] not in ('S0', 'SP'):
        raise ValueError('Unknown structured action checkpoint scope/condition')
    if expected_condition is not None and meta['condition'] != expected_condition:
        raise ValueError('Checkpoint condition does not match requested evaluation')
    # Evaluation cannot silently accept an initial/best/intermediate/test head.
    run = Path(meta['training_run']).resolve()
    if path != run / f'{meta["condition"]}_seed0.pt' or meta.get('updates') != 1400 or meta.get('seed') != 0:
        raise ValueError('Only preregistered final1400 seed0 S0/SP checkpoints are deployable')
    protocol_file, result_file = run / 'training_protocol.json', run / 'training_result.json'
    protocol = json.loads(protocol_file.read_text())
    result = json.loads(result_file.read_text())
    if (protocol['scope'] != SCOPE or protocol['updates'] != 1400 or protocol['seed'] != 0
            or protocol['conditions'] != {'S0': 0., 'SP': .1}
            or digest(protocol_file) != meta['training_protocol_sha256']):
        raise ValueError('Checkpoint does not match its frozen training protocol')
    if result.get('status') != 'complete' or len(result.get('reports', [])) != 2:
        raise ValueError('Completed paired training result is required')
    reports = {row['condition']: row for row in result['reports']}
    if set(reports) != {'S0', 'SP'}:
        raise ValueError('Both final control/candidate checkpoint reports are required')
    for condition, report in reports.items():
        final = run / f'{condition}_seed0.pt'
        if (report['updates'] != 1400 or Path(report['checkpoint']).resolve() != final
                or digest(final) != report['checkpoint_sha256']
                or report['initial_state_sha256'] != protocol['initial_state_sha256']):
            raise ValueError('Changed/nonfinal paired checkpoint: ' + condition)
    prepared = run / 'protocol.json'
    if protocol['source_sha256'].get(str(prepared)) != digest(prepared):
        raise ValueError('Prepared training cache protocol changed after training freeze')
    if (meta.get('initial_state_sha256') != protocol['initial_state_sha256']
            or meta.get('auxiliary_coefficient') != protocol['conditions'][meta['condition']]
            or meta['channels'] != protocol['channel_dim']
            or meta['geometry_mean'] != protocol['geometry_mean']
            or meta['caps_native_units'] != protocol['caps_native_units']):
        raise ValueError('Checkpoint metadata differs from its paired training protocol')
    if meta['channels'] != 32 or len(meta['geometry_mean']) != 24:
        raise ValueError('Unexpected structured channel/geometry contract')
    if not np.isfinite(meta['geometry_mean']).all():
        raise ValueError('Nonfinite geometry prior')
    np.testing.assert_array_equal(np.asarray(meta['caps_native_units'], np.float32), CAPS)
    for kind in ('projection', 'robot_config'):
        source = Path(meta[kind + '_path']).resolve()
        if digest(source) != meta[kind + '_sha256']:
            raise ValueError('Changed deployment dependency: ' + kind)
    with np.load(meta['projection_path'], allow_pickle=False) as data:
        if set(data.files) != {'mean', 'components', 'scale'}:
            raise ValueError('Unexpected shared channel PCA')
        projection = {key: data[key].copy() for key in data.files}
    if projection['components'].shape != (32, 2048) or projection['mean'].shape != (2048,):
        raise ValueError('Invalid channel PCA shapes')
    if projection['scale'].shape != (32,) or np.any(projection['scale'] <= 0):
        raise ValueError('Invalid channel scale')
    if not all(np.isfinite(value).all() for value in projection.values()):
        raise ValueError('Nonfinite shared PCA')
    return payload, projection


class StructuredActionPolicy:
    """Same sampler and auxiliary encoding in L0/S0/SP; history is query-causal."""
    def __init__(self, base_policy, checkpoint, condition, query_recorder=None):
        from .structured_action_adapter import StructuredActionAdapter
        if condition not in ('L0', 'S0', 'SP'):
            raise ValueError('Unknown structured action condition')
        payload, self.projection = read_checkpoint(checkpoint, None if condition == 'L0' else condition)
        self.metadata = payload['metadata']
        self.condition = condition
        self.base_policy = base_policy
        self.model = StructuredActionAdapter(self.metadata['channels'], self.metadata['geometry_mean'])
        self.model.load_state_dict(payload['model'], strict=True)
        self.model.eval().requires_grad_(False)
        np.testing.assert_array_equal(self.model.caps.detach().cpu().numpy(), CAPS)
        config = json.loads(Path(self.metadata['robot_config_path']).read_text())
        self.robots = [RobotFK(config, side) for side in ('left', 'right')]
        self.encode_eager = torch.compiler.disable(encode)
        self.sampler = base_policy._sample_actions
        self.query_recorder = query_recorder
        self.reset_history()

    def check_robot_config(self, config):
        """Static calibration/embodiment must agree with the training FK convention."""
        for side, trained in zip(('left', 'right'), self.robots):
            live = RobotFK(config, side)
            if (live.names != trained.names or live.tip != trained.tip
                    or digest(live.urdf) != digest(trained.urdf)
                    or not np.array_equal(live.root, trained.root)):
                raise ValueError('Live robot configuration differs from training: ' + side)

    def reset_history(self):
        self.previous = None
        self.query_count = 0
        self.episode_seed = None
        self.last_diagnostics = None

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
        if np.any(np.abs(delta) > CAPS[None] + 1e-6):
            raise ValueError('Structured correction exceeds original native caps')
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
        if self.query_recorder is not None:
            self.query_recorder(dict(seed=self.episode_seed, **diagnostic))
        self.last_diagnostics = diagnostic
        self.previous = (joint.copy(), state.copy())
        self.query_count += 1
        return actions, delta, base
