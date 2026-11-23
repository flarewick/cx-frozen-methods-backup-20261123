"""Observable shared action residual with an auxiliary geometry readout.

Geometry targets are training labels only. Neither task/asset identifiers nor
predicted geometry enter the action head. Teacher clipping below defines an
offline direction target; it does not imply the capped residual can reproduce
the full teacher trajectory.
"""
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .geometry_readout import GeometryReadout


FORWARD_KEYS = frozenset({
    'visual', 'text', 'state', 'past_visual', 'past_text', 'past_state',
    'has_history', 'ee_pose', 'past_ee_pose',
})
ACTION_HORIZON = 32
ARM_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
RESIDUAL_CAPS = (.1,) * 6 + (.2,) + (.1,) * 6 + (.2,)


def _finite_tensor(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f'{name} must be a floating point tensor')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite')


class StructuredActionAdapter(GeometryReadout):
    """One legal-observation trunk, auxiliary geometry and capped action heads."""

    def __init__(self, channels=32, geometry_mean=None):
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError('channels must be a positive integer')
        if geometry_mean is None:
            geometry_mean = np.zeros(24, dtype=np.float32)
        mean = torch.as_tensor(geometry_mean, dtype=torch.float32)
        if mean.shape != (24,) or not torch.isfinite(mean).all():
            raise ValueError('geometry_mean must contain 24 finite values')
        super().__init__(channels, mean)
        self.geometry_head = self.head
        del self.head
        self.action_head = nn.Sequential(nn.Linear(83, 64), nn.SiLU(), nn.Linear(64, 14))
        nn.init.zeros_(self.action_head[-1].weight)
        nn.init.zeros_(self.action_head[-1].bias)
        self.register_buffer('caps', torch.tensor(RESIDUAL_CAPS, dtype=torch.float32))
        t = torch.linspace(0, 1, ACTION_HORIZON)
        self.register_buffer('time_encoding', torch.stack(
            [t, torch.sin(torch.pi * t), torch.cos(torch.pi * t),
             torch.sin(2 * torch.pi * t), torch.cos(2 * torch.pi * t)], dim=-1))

    def _validate(self, x, base):
        if not isinstance(x, Mapping) or set(x) != FORWARD_KEYS:
            raise ValueError('Forward inputs must contain exactly the nine observable fields; GT/metadata fields are forbidden')
        _finite_tensor(base, 'base')
        if base.ndim != 3 or base.shape[1:] != (ACTION_HORIZON, 14) or base.shape[0] == 0:
            raise ValueError('base must have shape [B,32,14] with B>0')
        n = base.shape[0]
        shapes = {
            'text': (n, 2, self.channels), 'past_text': (n, 2, self.channels),
            'state': (n, 14), 'past_state': (n, 14), 'has_history': (n, 1),
            'ee_pose': (n, 2, 9), 'past_ee_pose': (n, 2, 9),
        }
        for key, value in x.items():
            _finite_tensor(value, key)
            expected = ((n, 3, 2, 2, self.channels), (n, 3, 4, self.channels)) if key in ('visual', 'past_visual') else (shapes[key],)
            if tuple(value.shape) not in expected:
                raise ValueError(f'Invalid {key} shape: {tuple(value.shape)}')
            if value.dtype != base.dtype or value.device != base.device:
                raise ValueError(f'{key} must share base dtype and device')
        if not ((x['has_history'] == 0) | (x['has_history'] == 1)).all():
            raise ValueError('has_history must contain only zero or one')
        if self.caps.dtype != base.dtype or self.caps.device != base.device:
            raise ValueError('Model must share base dtype and device; use model.to(...)')

    def forward(self, x, base, enabled=True):
        if not isinstance(enabled, bool):
            raise ValueError('enabled must be a bool')
        self._validate(x, base)
        cur, cw = self.attention(x, '', 'conditional')
        past, pw = self.attention(x, 'past_', 'conditional')
        merged = torch.cat([
            cur, x['text'].flatten(1), past, x['past_text'].flatten(1),
            x['state'], x['past_state'], x['ee_pose'].flatten(1),
            x['past_ee_pose'].flatten(1), x['has_history'],
        ], dim=1)
        latent = self.trunk(merged)
        geometry = self.geometry_head(latent) + self.reg_mean
        if enabled:
            action_context = base.clone()
            action_context[..., list(ARM_INDICES)] /= torch.pi
            raw = self.action_head(torch.cat([
                latent[:, None, :].expand(-1, ACTION_HORIZON, -1), action_context,
                self.time_encoding[None].expand(len(base), -1, -1),
            ], dim=-1))
            delta = torch.tanh(raw) * self.caps
            actions = base + delta
        else:
            delta = torch.zeros_like(base)
            actions = base
        return dict(actions=actions, delta=delta, geometry=geometry,
                    current_attention=cw, past_attention=pw)


def action_direction_loss(pred_delta, base, target, valid, weights, caps):
    """Masked directional teacher loss, with no invalid-tail or zero-row labels.

    Average SmoothL1 over 14 coordinates, then over the true valid steps in each
    row. Row weights are applied as supplied, without redistribution. Targets in
    invalid steps or zero-weight rows are never used, including nonfinite ones.
    """
    for name, value in (('pred_delta', pred_delta), ('base', base), ('weights', weights), ('caps', caps)):
        _finite_tensor(value, name)
    if base.ndim != 3 or base.shape[1:] != (ACTION_HORIZON, 14):
        raise ValueError('base must have shape [B,32,14]')
    n = base.shape[0]
    if pred_delta.shape != base.shape or not isinstance(target, torch.Tensor) or target.shape != base.shape:
        raise ValueError('pred_delta and target must match base shape')
    if not isinstance(valid, torch.Tensor) or valid.shape != (n, ACTION_HORIZON):
        raise ValueError('valid must have shape [B,32]')
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError('valid must contain only zero or one')
    if weights.shape != (n,) or (weights < 0).any():
        raise ValueError('weights must be nonnegative [B]')
    if caps.shape != (14,) or (caps <= 0).any():
        raise ValueError('caps must be positive [14]')
    for value in (pred_delta, target, valid, weights, caps):
        if value.device != base.device:
            raise ValueError('All loss arguments must share a device')
    for value in (pred_delta, target, weights, caps):
        if not value.is_floating_point() or value.dtype != base.dtype:
            raise ValueError('Floating loss arguments must share base dtype')
    active = valid.bool() & (weights[:, None] > 0)
    if not torch.isfinite(target[active]).all():
        raise ValueError('Active teacher targets must be finite')
    safe_target = torch.where(active[..., None], target, base)
    desired = ((safe_target - base) / caps).clamp(-1, 1)
    per_step = F.smooth_l1_loss(pred_delta / caps, desired, reduction='none').mean(-1)
    per_row = (per_step * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    return (per_row * weights).sum()
