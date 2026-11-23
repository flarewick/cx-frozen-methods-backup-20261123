"""Small shared GRIP candidate completion head; original arm adapter frozen."""
import torch
from torch import nn
from .structured_action_adapter import ARM_INDICES,FORWARD_KEYS
class SharedPhaseGripperAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(83,32),nn.SiLU(),nn.Linear(32,2))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    @staticmethod
    def observable_context(frozen,x,base):
        frozen._validate(x,base)
        if frozen.training or any(p.requires_grad for p in frozen.parameters()):
            raise ValueError('OriginalA observable trunk must be frozen/eval')
        with torch.no_grad():
            cur,_=frozen.attention(x,'','conditional');past,_=frozen.attention(x,'past_','conditional')
            merged=torch.cat([cur,x['text'].flatten(1),past,x['past_text'].flatten(1),x['state'],x['past_state'],
                x['ee_pose'].flatten(1),x['past_ee_pose'].flatten(1),x['has_history']],1)
            latent=frozen.trunk(merged);native=base.clone();native[...,list(ARM_INDICES)]/=torch.pi
            context=torch.cat([latent[:,None].expand(-1,32,-1),native,frozen.time_encoding[None].expand(len(base),-1,-1)],-1)
        return context
    def forward(self,context):
        if context.ndim!=3 or context.shape[1:]!=(32,83) or not torch.isfinite(context).all():
            raise ValueError('Only current observable A-context/base/time allowed')
        return torch.tanh(self.head(context))*.2

def masked_gripper_loss(pred,target,valid,weights):
    if pred.shape!=target.shape or pred.ndim!=3 or pred.shape[1:]!=(32,2) or valid.shape!=pred.shape[:2] or weights.shape!=(len(pred),):
        raise ValueError('Native GRIP target/mask/weight shapes')
    if not ((valid==0)|(valid==1)).all() or (weights<0).any() or not torch.isfinite(weights).all() or not torch.isfinite(pred).all():
        raise ValueError('Finite values andtruevalid prefix required')
    active=valid.bool()&(weights[:,None]>0)
    if not torch.isfinite(target[active]).all():raise ValueError('Nonfinite active teacher')
    desired=torch.where(active[...,None],target,torch.zeros_like(target))/.2
    error=nn.functional.smooth_l1_loss(pred/.2,desired,reduction='none').mean(-1)
    row=(error*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return (row*weights).sum()

