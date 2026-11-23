"""Small shared effect readout; oracle object selection remains offline-only."""
import numpy as np
import torch
from torch import nn


class SharedEffect(nn.Module):
    def __init__(self,context_dim,action_dim,mode='conditioned',width=32):
        super().__init__();self.mode=mode
        if mode not in ('conditioned','action_only','no_action'):raise ValueError('Unknown mode')
        if mode!='action_only':self.context=nn.Sequential(nn.Linear(context_dim,width),nn.Tanh())
        if mode!='no_action':self.action=nn.Sequential(nn.Linear(action_dim,width,bias=False),nn.Tanh())
        self.head=nn.Linear(width*(2 if mode=='conditioned' else 1),7,bias=mode=='no_action')
        # Start from the zero-effect prior, rather than hallucinated candidate effects.
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:nn.init.zeros_(self.head.bias)

    def forward(self,context,action):
        if self.mode=='no_action':return self.head(self.context(context))
        a=self.action(action)
        if self.mode=='action_only':return self.head(a)
        return self.head(torch.cat((a,a*self.context(context)),dim=-1))


def project_relation(pose,k,extrinsic,shape,object_geometry):
    """Observed object centroid vs known FK tip in padded image coordinates.

    object_geometry derives from GT masks in this oracle diagnostic.
    Negative/near-zero camera depth is explicitly masked.
    """
    point=extrinsic@np.r_[pose[:3,3],1.];z=point[2]
    if z<=1e-6:return np.zeros(5)
    pixel=k@point[:3];pixel=pixel[:2]/pixel[2]
    height,width=shape;ratio=max(width/224,height/224);rh,rw=int(height/ratio),int(width/ratio)
    uv=(pixel*np.array([rw/width,rh/height])+[(224-rw)//2,(224-rh)//2])/224
    visible=object_geometry[0]
    return np.r_[(np.array(object_geometry[2:4])-uv)*visible,z,1.,visible]


def fold_indices(records,held,unit):
    if unit not in ('task','group'):raise ValueError('Bad split unit')
    train=np.array([i for i,r in enumerate(records) if r[unit]!=held],int)
    test=np.array([i for i,r in enumerate(records) if r[unit]==held],int)
    if not len(train) or not len(test):raise ValueError('Empty split')
    if {records[i]['group'] for i in train}&{records[i]['group'] for i in test}:raise ValueError('Scene leakage')
    return train,test
