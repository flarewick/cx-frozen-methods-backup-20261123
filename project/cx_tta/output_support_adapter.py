"""S2o: output support and target clipping, with unchanged state allocation."""
import numpy as np
import torch
from torch.nn import functional as F

from .structured_action_adapter import (
    StructuredActionAdapter,ARM_INDICES,ACTION_HORIZON,action_direction_loss,_finite_tensor,
)

CONDITIONS={
    'A_bounded_clipped':('bounded',True),
    'B_open_clipped':('open',True),
    'C_bounded_raw':('bounded',False),
    'D_open_raw':('open',False),
}


class OutputSupportAdapter(StructuredActionAdapter):
    def __init__(self,channels,geometry_mean,condition):
        if condition not in CONDITIONS:raise ValueError('Unknown factorial cell')
        super().__init__(channels,geometry_mean)
        self.output_condition=condition

    def forward(self,x,base,enabled=True):
        if CONDITIONS[self.output_condition][0]=='bounded':
            return super().forward(x,base,enabled)
        if not isinstance(enabled,bool):raise ValueError('Enabled must be bool')
        self._validate(x,base)
        cur,cw=self.attention(x,'','conditional');past,pw=self.attention(x,'past_','conditional')
        merged=torch.cat([cur,x['text'].flatten(1),past,x['past_text'].flatten(1),x['state'],
            x['past_state'],x['ee_pose'].flatten(1),x['past_ee_pose'].flatten(1),x['has_history']],dim=1)
        latent=self.trunk(merged);geometry=self.geometry_head(latent)+self.reg_mean
        if enabled:
            action_context=base.clone();action_context[...,list(ARM_INDICES)]/=torch.pi
            raw=self.action_head(torch.cat([latent[:,None,:].expand(-1,ACTION_HORIZON,-1),action_context,
                self.time_encoding[None].expand(len(base),-1,-1)],dim=-1))
            delta=raw*self.caps;actions=base+delta
        else:delta=torch.zeros_like(base);actions=base
        return dict(actions=actions,delta=delta,geometry=geometry,current_attention=cw,past_attention=pw)


def support_loss(delta,base,target,valid,weights,scale,clip_target):
    if not isinstance(clip_target,bool):raise ValueError('Clipping choice must be bool')
    if clip_target:return action_direction_loss(delta,base,target,valid,weights,scale)
    for name,value in (('delta',delta),('base',base),('weights',weights),('scale',scale)):
        _finite_tensor(value,name)
    if base.ndim!=3 or base.shape[1:]!=(32,14) or delta.shape!=base.shape or target.shape!=base.shape:
        raise ValueError('Native tensor shape')
    n=len(base)
    if valid.shape!=(n,32) or not ((valid==0)|(valid==1)).all():raise ValueError('Native valid mask')
    if weights.shape!=(n,) or (weights<0).any() or scale.shape!=(14,) or (scale<=0).any():
        raise ValueError('Weight/normalization shape')
    for value in (delta,target,valid,weights,scale):
        if value.device!=base.device:raise ValueError('Loss device mismatch')
    for value in (delta,target,weights,scale):
        if not value.is_floating_point() or value.dtype!=base.dtype:raise ValueError('Loss dtype mismatch')
    active=valid.bool()&(weights[:,None]>0)
    if not torch.isfinite(target[active]).all():raise ValueError('Nonfinite active raw label')
    safe=torch.where(active[...,None],target,base)
    desired=(safe-base)/scale
    per_step=F.smooth_l1_loss(delta/scale,desired,reduction='none').mean(-1)
    per_row=(per_step*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return (per_row*weights).sum()


def native_domain_stats(actions,limits):
    actions=np.asarray(actions);limits=np.asarray(limits)
    if actions.shape[-1]!=14 or limits.shape!=(14,2) or not np.isfinite(actions).all():
        raise ValueError('Finite native actions and joint limits required')
    bad=(actions<limits[:,0])|(actions>limits[:,1])
    return dict(arm_outside_URDF_coordinates=int(bad[...,list(ARM_INDICES)].sum()),
        nominal_gripper_outside_0_1_coordinates=int(bad[...,[6,13]].sum()),
        gripper_handling='Original native per-physics-tick np.clip; no policy-boundary clipping')


def validate_native_arms(actions,limits):
    stats=native_domain_stats(actions,limits)
    if stats['arm_outside_URDF_coordinates']:
        raise ValueError('Arm goal outside common URDF domain; refuse execution without silent projection')
    return stats
