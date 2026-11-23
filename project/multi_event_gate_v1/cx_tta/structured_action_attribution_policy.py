"""Frozen S0 interventions for attribution, without changing its sampler path.

This is a diagnostic wrapper, not a newly trained policy or a task router. All
presets use the frozen deployment encoder, legal-input packing, compiled sampler
and RNG guards by delegating to ``StructuredActionPolicy.predict``. Only the
deterministic head's applied residual and, in one preset, past inputs change.
"""
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import structured_action_deploy as frozen_deploy
from .audit import digest
from .deploy import unpack_observation
from .eval_diagnostics import array_hash
from .recovery_contract import rng_capture, rng_fingerprint
from .structured_action_adapter import ARM_INDICES


@dataclass(frozen=True)
class AttributionCondition:
    arm_strength: float
    gripper_strength: float
    history: str
    candidate_enabled: bool = True


CONDITIONS = {
    'L0': AttributionCondition(0., 0., 'real', False),
    'S0_zero': AttributionCondition(0., 0., 'real'),
    'S0': AttributionCondition(1., 1., 'real'),
    'S0_quarter': AttributionCondition(.25, .25, 'real'),
    'S0_no_gripper': AttributionCondition(1., 0., 'real'),
    'S0_no_history': AttributionCondition(1., 1., 'zero'),
}
GRIPPER_INDICES = (6, 13)


def _strength_vector(arm_strength, gripper_strength, base):
    values = (arm_strength, gripper_strength)
    if any(not isinstance(value, Real) or isinstance(value, bool)
           or not np.isfinite(value) or not 0 <= value <= 1
           for value in values):
        raise ValueError('Diagnostic strengths must be finite numbers in [0,1]')
    strength = torch.full((14,), float(arm_strength), dtype=base.dtype, device=base.device)
    strength[list(GRIPPER_INDICES)] = float(gripper_strength)
    return strength


def apply_component_strength(base, raw_delta, arm_strength, gripper_strength, *, raw_actions=None):
    """Scale arm/gripper deltas while preserving disabled coordinates bitwise.

    Exact-zero application returns a clone of the base rather than ``base + 0``:
    adding positive zero would change the sign bit of a negative-zero base. The
    unit-strength path may return the frozen head's existing actions unchanged.
    """
    if (not isinstance(base, torch.Tensor) or not base.is_floating_point()
            or not isinstance(raw_delta, torch.Tensor) or raw_delta.shape != base.shape
            or base.shape[-1:] != (14,) or raw_delta.dtype != base.dtype
            or raw_delta.device != base.device):
        raise ValueError('Base and raw delta must have matching floating [...,14] tensors')
    if not torch.isfinite(base).all() or not torch.isfinite(raw_delta).all():
        raise ValueError('Nonfinite diagnostic base or raw delta')
    strength = _strength_vector(arm_strength, gripper_strength, base)
    if not torch.any(strength):
        return base.clone(), torch.zeros_like(base)
    if torch.all(strength == 1):
        actions = base + raw_delta if raw_actions is None else raw_actions
        if (actions.shape != base.shape or actions.dtype != base.dtype
                or actions.device != base.device or not torch.isfinite(actions).all()):
            raise ValueError('Invalid frozen head actions')
        return actions, raw_delta
    delta = raw_delta * strength
    actions = torch.where(strength == 0, base, base + delta)
    return actions, delta


class _FrozenHeadIntervention(nn.Module):
    """Proxy preserves the original head call; stores pre-application deltas."""
    def __init__(self, model, spec):
        super().__init__()
        self.frozen_model = model
        self.spec = spec
        self.last_raw_delta = None

    @property
    def caps(self):
        return self.frozen_model.caps

    def forward(self, inputs, base, enabled=True):
        if enabled != self.spec.candidate_enabled:
            raise ValueError('Head enable flag differs from diagnostic preset')
        output = self.frozen_model(inputs, base, enabled=enabled)
        raw_delta = output['delta']
        if not torch.isfinite(raw_delta).all() or torch.any(raw_delta.abs() > self.caps + 1e-6):
            raise ValueError('Raw frozen correction exceeds original caps or is nonfinite')
        actions, delta = apply_component_strength(base, raw_delta,
            self.spec.arm_strength, self.spec.gripper_strength, raw_actions=output['actions'])
        self.last_raw_delta = raw_delta[0].detach().cpu().numpy().copy()
        return dict(output, actions=actions, delta=delta)


def source_contract():
    """Hash source paths actually used; not an assertion of numerical parity."""
    from . import structured_action_adapter, structured_features
    paths = (Path(__file__), Path(frozen_deploy.__file__),
             Path(structured_action_adapter.__file__), Path(structured_features.__file__))
    return {
        'scope': 'frozen S0 single-factor attribution; no learning',
        'source_sha256': {str(path.resolve()): digest(path) for path in paths},
        'conditions': {key: asdict(value) for key, value in CONDITIONS.items()},
        'sampler': 'one shared compiled object; noise=None; one infer call per query',
        'L0': 'frozen head enabled=False; raw and applied delta zero',
        'S0_zero': 'frozen head enabled=True; raw retained; applied exactly zero',
        'no_history': 'past fields and has_history zero; real observation history still tracked',
        'privileged_inputs': False,
    }


class StructuredAttributionPolicy(frozen_deploy.StructuredActionPolicy):
    """Diagnostic presets around an unmodified final S0 checkpoint.

    ``query_recorder`` receives the extended JSON diagnostic; ``numeric_recorder``
    receives one dictionary of copied float32 arrays. The latter is optional and
    both ``last_numeric`` and ``last_diagnostics`` are available to the runner.
    Construct all wrappers before resetting episode RNG or warming the sampler.
    """
    def __init__(self, base_policy, checkpoint, condition, query_recorder=None, numeric_recorder=None):
        if condition not in CONDITIONS:
            raise ValueError('Unknown frozen attribution condition: ' + str(condition))
        # Even L0 must validate the same final S0 checkpoint and calibration.
        super().__init__(base_policy, checkpoint, 'S0', query_recorder=None)
        self.attribution_query_recorder = query_recorder
        self.numeric_recorder = numeric_recorder
        self.model = _FrozenHeadIntervention(self.model, CONDITIONS[condition]).eval().requires_grad_(False)
        self.configure_condition(condition)

    @property
    def frozen_model(self):
        return self.model.frozen_model

    def configure_condition(self, condition):
        if condition not in CONDITIONS:
            raise ValueError('Unknown frozen attribution condition: ' + str(condition))
        if self.previous is not None or self.query_count:
            raise RuntimeError('Reset history before changing an attribution condition')
        self.condition = condition
        self.model.spec = CONDITIONS[condition]
        self.last_numeric = None

    def reset_history(self):
        super().reset_history()
        self.last_numeric = None
        if isinstance(getattr(self, 'model', None), _FrozenHeadIntervention):
            self.model.last_raw_delta = None

    def predict(self, observation, instruction, noise, *, enabled):
        spec = CONDITIONS[self.condition]
        if enabled != spec.candidate_enabled:
            raise ValueError('Enabled flag differs from diagnostic condition')
        real_previous = self.previous
        if spec.history == 'zero':
            # The frozen packer zeros every past field and has_history on this
            # path. After predict it stores this real current query for tracking.
            self.previous = None
        try:
            actions, delta, base = super().predict(observation, instruction, noise, enabled=enabled)
        except BaseException:
            self.previous = real_previous
            raise
        raw_delta = self.model.last_raw_delta
        if raw_delta is None or raw_delta.shape != base.shape:
            raise RuntimeError('Frozen raw delta was not retained')
        # Array bytes, rather than equal-valued comparison, enforce signed-zero
        # identity of the applied-zero conditions.
        if spec.arm_strength == spec.gripper_strength == 0 and actions.tobytes() != base.tobytes():
            raise RuntimeError('Applied-zero actions are not bitwise base identity')
        _, state = unpack_observation(observation)
        diagnostic = dict(self.last_diagnostics,
            attribution_condition=self.condition, arm_strength=spec.arm_strength,
            gripper_strength=spec.gripper_strength, history_policy=spec.history,
            real_history_available=real_previous is not None,
            candidate_enabled=spec.candidate_enabled,
            raw_delta_sha256=array_hash(raw_delta), applied_delta_sha256=array_hash(delta),
            raw_max_abs_arm=float(np.abs(raw_delta[:, ARM_INDICES]).max()),
            raw_max_abs_gripper=float(np.abs(raw_delta[:, GRIPPER_INDICES]).max()),
            applied_max_abs_arm=float(np.abs(delta[:, ARM_INDICES]).max()),
            applied_max_abs_gripper=float(np.abs(delta[:, GRIPPER_INDICES]).max()),
            exact_zero_applied=bool(not np.any(delta) and actions.tobytes() == base.tobytes()))
        self.last_diagnostics = diagnostic
        self.last_numeric = {
            'seed': self.episode_seed, 'query': diagnostic['query'],
            'command_state': np.asarray(state, np.float32).copy(),
            'base': base.copy(), 'raw_delta': raw_delta.copy(),
            'applied_delta': delta.copy(), 'actions': actions.copy(),
        }
        if self.attribution_query_recorder is not None:
            self.attribution_query_recorder(dict(seed=self.episode_seed, **diagnostic))
        if self.numeric_recorder is not None:
            self.numeric_recorder({key: value.copy() if isinstance(value, np.ndarray) else value
                                   for key, value in self.last_numeric.items()})
        if rng_fingerprint(rng_capture()) != diagnostic['rng_after_sampler']:
            raise RuntimeError('Attribution recording changed sampler RNG')
        if self.base_policy._sample_actions is not self.sampler or torch._dynamo.config.disable:
            raise RuntimeError('Attribution recording changed compiled sampling interface')
        return actions, delta, base
