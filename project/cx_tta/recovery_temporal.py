"""Task-independent command envelope and supervised command-difference loss.

Pairs are offline labels from audited continuous traces, never forward inputs.
The gripper envelope contains the original raw base command and [0, 1]. It
prevents additional overshoot while preserving the zero/disabled adapter.
It is NOT a claim of strict [0, 1] outputs or equivalent physical execution.
"""
import torch
from .pose_adapter import PoseAdapter


def gripper_envelope(base, delta):
    lower = torch.minimum(base, torch.zeros_like(base))
    upper = torch.maximum(base, torch.ones_like(base))
    up, down = upper - base, base - lower
    # At an endpoint and zero delta, choose the inward derivative. An outward
    # correction beyond the envelope remains exactly zero for nonzero delta.
    positive = (delta > 0) | ((delta == 0) & (up >= down))
    room = torch.where(positive, up, down)
    value = base + room * torch.tanh(delta / room.clamp_min(1e-6))
    return torch.maximum(lower, torch.minimum(upper, value))


class TemporalPoseAdapter(PoseAdapter):
    def __init__(self, context_dim, *, envelope=False):
        super().__init__(context_dim, proposal=True)
        self.envelope = envelope

    def forward(self, context, base_action, *, enabled=True):
        if not enabled:
            return base_action
        if not self.envelope:
            return super().forward(context, base_action)
        batch, horizon, _ = base_action.shape
        time = torch.linspace(0, 1, horizon, device=base_action.device, dtype=base_action.dtype)
        inputs = torch.cat((context[:, None].expand(-1, horizon, -1), base_action,
                            time[None, :, None].expand(batch, -1, -1)), dim=-1)
        delta = self.head(self.trunk(inputs)) * self.caps
        raw = base_action + delta
        left = gripper_envelope(base_action[..., 6:7], delta[..., 6:7])
        right = gripper_envelope(base_action[..., 13:14], delta[..., 13:14])
        return torch.cat((raw[..., :6], left, raw[..., 7:13], right), dim=-1)


def temporal_loss(pred, target, valid, caps, row_weights, pair_indices, pair_weights):
    """Match real command increments, including valid within-block gripper steps.

Within-block loss uses the original per-component balanced row weights. The
cross-block loss is separately balanced over recovery task/scene/boundary.
Both use the original BC scales (.1 rad and .2 normalized gripper), weight 1.
There is no penalty on a correctly predicted nonzero teacher increment.
"""
    mask = valid[:, 1:] & valid[:, :-1]
    error = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])) / caps
    per_row = (error.square().mean(-1) * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    within = (per_row * row_weights).sum()
    if not len(pair_indices):
        return within, pred.sum() * 0
    before, last, after = pair_indices.unbind(-1)
    jump = (pred[after, 0] - pred[before, last]) - (target[after, 0] - target[before, last])
    seam = (((jump / caps).square().mean(-1)) * pair_weights).sum()
    return within, seam
