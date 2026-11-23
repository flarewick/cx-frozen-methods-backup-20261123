"""Relative chunk-time features; no scene, phase, or target input."""
import math
import torch
from torch import nn
from .pose_adapter import PoseAdapter
from .recovery_temporal import gripper_envelope


def time_features(horizon, mode, *, device, dtype):
    if horizon < 1 or mode not in ('repeat', 'fourier'):
        raise ValueError('Positive horizon and predefined time encoding required')
    t = torch.linspace(0, 1, horizon, device=device, dtype=dtype)
    if mode == 'repeat':
        extra = t[:, None].expand(-1, 8)
    else:
        frequency = torch.tensor([1, 2, 4, 8], device=device, dtype=dtype)
        phase = 2 * math.pi * t[:, None] * frequency[None]
        extra = torch.cat((phase.sin(), phase.cos()), dim=-1)
    # Same channel RMS in both modes, defined by positions alone, never fitted
    # to training or held data. Valid-tail lengths do not change the clock.
    extra = extra - extra.mean(dim=0, keepdim=True)
    extra = extra / extra.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    return torch.cat((t[:, None], extra / math.sqrt(8)), dim=-1)


class TimeEncodedAdapter(PoseAdapter):
    def __init__(self, context_dim, *, encoding, envelope=False):
        super().__init__(context_dim, proposal=True)
        if encoding not in ('repeat', 'fourier'):
            raise ValueError('Unknown fixed time encoding')
        self.encoding, self.envelope = encoding, envelope
        original = self.trunk[0]
        wider = nn.Linear(original.in_features + 8, original.out_features)
        with torch.no_grad():
            wider.weight[:, :original.in_features].copy_(original.weight)
            wider.weight[:, original.in_features:].zero_()
            wider.bias.copy_(original.bias)
        self.trunk[0] = wider

    def forward(self, context, base_action, *, enabled=True):
        if not enabled:
            return base_action
        if base_action.ndim != 3 or base_action.shape[-1] != 14:
            raise ValueError('Expected native [batch,horizon,14] action chunk')
        batch, horizon, _ = base_action.shape
        if context.shape != (batch, self.context_dim):
            raise ValueError('Invalid observable context shape')
        time = time_features(horizon, self.encoding, device=base_action.device, dtype=base_action.dtype)
        x = torch.cat((context[:, None].expand(-1, horizon, -1), base_action,
                       time[None].expand(batch, -1, -1)), dim=-1)
        delta = self.head(self.trunk(x)) * self.caps
        raw = base_action + delta
        if not self.envelope:
            return raw
        left = gripper_envelope(base_action[..., 6:7], delta[..., 6:7])
        right = gripper_envelope(base_action[..., 13:14], delta[..., 13:14])
        return torch.cat((raw[..., :6], left, raw[..., 7:13], right), dim=-1)
