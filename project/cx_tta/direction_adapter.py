"""Expert-plan directional auxiliary prototype; no GT or future targets at inference."""
import torch
from torch import nn
from .residual import SharedResidualAdapter


class DirectionAdapter(SharedResidualAdapter):
    """Matched-capacity controls: BC, auxiliary-only, predicted-direction conditioning.

    The 14D direction is predicted from observable context/base action, never supplied
    from an expert at inference. It describes desired planned FK correction, not
    an object-progress goal or an action-effect prediction.
    """
    def __init__(self, context_dim=590, action_caps=None, hidden_dim=128, mode='conditioned'):
        if mode not in ('bc', 'auxiliary', 'conditioned'):
            raise ValueError('Unknown direction control')
        if action_caps is None:
            action_caps = [.1]*6+[.2]+[.1]*6+[.2]
        super().__init__(context_dim, action_caps, hidden_dim)
        self.mode = mode
        self.direction_head = nn.Linear(hidden_dim, 14)
        self.head = nn.Linear(hidden_dim+14, 14)
        for layer in (self.direction_head, self.head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, context, base_action, *, enabled=True):
        if base_action.ndim != 3 or base_action.shape[-1] != 14 or base_action.shape[1] < 1:
            raise ValueError('Expected [batch,horizon,14]')
        batch, horizon, _ = base_action.shape
        if context.shape != (batch, self.context_dim) or not torch.isfinite(context).all() or not torch.isfinite(base_action).all():
            raise ValueError('Invalid observable input')
        if not enabled:
            return base_action, torch.zeros_like(base_action), torch.zeros_like(base_action)
        t = torch.linspace(0, 1, horizon, device=base_action.device, dtype=base_action.dtype)
        x = torch.cat((context[:,None].expand(-1,horizon,-1), base_action, t[None,:,None].expand(batch,-1,-1)), dim=-1)
        hidden = self.trunk(x)
        direction = self.direction_head(hidden)
        conditioning = direction if self.mode == 'conditioned' else torch.zeros_like(direction)
        delta = torch.tanh(self.head(torch.cat((hidden, conditioning),dim=-1)))*self.caps
        return base_action+delta, delta, direction


def losses(model, context, base_action, expert, action_valid, direction_target, direction_valid, auxiliary_weight=.1):
    """Labels enter losses ONLY. Invalid tails excluded for BC and auxiliary tasks."""
    predicted, delta, direction = model(context, base_action)
    bc = (((predicted-expert)/model.caps).square().mean(-1)*action_valid).sum(-1)/action_valid.sum(-1).clamp(min=1)
    offsets = torch.tensor([0,3,7,15,31], device=direction.device)
    if base_action.shape[1] != 32 or direction_target.shape != (len(context),5,14) or direction_valid.shape != (len(context),5):
        raise ValueError('Auxiliary contract expects 32 native actions and five recorded endpoints')
    error = nn.functional.smooth_l1_loss(direction[:,offsets],direction_target,reduction='none').mean(-1)
    aux = (error*direction_valid).sum(-1)/direction_valid.sum(-1).clamp(min=1)
    total = bc.mean() + (0 if model.mode == 'bc' else auxiliary_weight*aux.mean())
    return total, dict(bc=bc.mean(), auxiliary=aux.mean(), delta=delta)
