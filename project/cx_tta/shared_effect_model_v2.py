"""Zero-anchored shared response with odd and even action features, same parameter budget."""
import torch
from torch import nn
from .shared_effect_model import fold_indices


class SharedEffect(nn.Module):
    def __init__(self,context_dim,action_dim,mode='conditioned',width=32):
        super().__init__();self.mode=mode
        if mode not in ('conditioned','action_only','no_action') or width%2:raise ValueError('Invalid mode/width')
        if mode!='action_only':self.context=nn.Sequential(nn.Linear(context_dim,width),nn.Tanh())
        if mode!='no_action':
            self.action_odd=nn.Sequential(nn.Linear(action_dim,width//2,bias=False),nn.Tanh())
            self.action_even=nn.Sequential(nn.Linear(action_dim,width//2,bias=False),nn.Tanh())
        self.head=nn.Linear(width*(2 if mode=='conditioned' else 1),7,bias=mode=='no_action')
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:nn.init.zeros_(self.head.bias)

    def forward(self,context,action):
        if self.mode=='no_action':return self.head(self.context(context))
        # Both branches vanish at zero. Squared features permit unilateral/asymmetric contact.
        a=torch.cat((self.action_odd(action),self.action_even(action.square())),dim=-1)
        return self.head(a if self.mode=='action_only' else torch.cat((a,a*self.context(context)),dim=-1))
