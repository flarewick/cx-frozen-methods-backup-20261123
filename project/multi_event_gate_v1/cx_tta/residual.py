"""Minimal B2 baseline, not yet wired to RoboTwin.

Requires causal, nonprivileged context and an audited action convention.
Caps are supplied in the SAME units as base actions, never guessed here.
"""
import torch
from torch import nn


class SharedResidualAdapter(nn.Module):
    def __init__(self, context_dim, action_caps, hidden_dim=128):
        super().__init__()
        caps = torch.as_tensor(action_caps, dtype=torch.float32).clone()
        if caps.ndim != 1 or not len(caps) or not torch.isfinite(caps).all() or (caps <= 0).any():
            raise ValueError("Positive finite per-channel action caps required")
        if context_dim < 1 or hidden_dim < 1:
            raise ValueError("Invalid model dimension")
        self.context_dim = context_dim
        self.register_buffer("caps", caps)
        self.trunk = nn.Sequential(nn.Linear(context_dim + len(caps) + 1, hidden_dim),
                                   nn.LayerNorm(hidden_dim), nn.SiLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.head = nn.Linear(hidden_dim, len(caps))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, context, base_action, *, enabled=True):
        if base_action.ndim != 3 or base_action.shape[-1] != len(self.caps) or base_action.shape[1] < 1:
            raise ValueError("Expected [batch, horizon, action_dim]")
        if context.shape != (base_action.shape[0], self.context_dim):
            raise ValueError("Invalid context shape")
        if not torch.isfinite(context).all() or not torch.isfinite(base_action).all():
            raise ValueError("Nonfinite model input")
        if not enabled:
            return base_action, torch.zeros_like(base_action)
        batch, horizon, _ = base_action.shape
        time = torch.linspace(0, 1, horizon, device=base_action.device, dtype=base_action.dtype)
        x = torch.cat((context[:, None, :].expand(-1, horizon, -1),
                       base_action, time[None, :, None].expand(batch, -1, -1)), dim=-1)
        delta = torch.tanh(self.head(self.trunk(x))) * self.caps
        return base_action + delta, delta
