"""Shared observable adapter and differentiable native control-link FK."""
import numpy as np
import torch
from torch import nn
from .residual import SharedResidualAdapter

class PoseAdapter(SharedResidualAdapter):
    """Same architecture, original bounded residual or explicit new action center.

    The proposal variant deliberately changes command support. It is not an
    original-cap result, and is not LoRA injected into a pretrained backbone.
    Forward inputs remain current/causal observable context and base actions.
    """
    def __init__(self,context_dim,proposal=False):
        super().__init__(context_dim=context_dim,action_caps=[.1]*6+[.2]+[.1]*6+[.2]);self.proposal=proposal

    def forward(self,context,base_action):
        batch,horizon,_=base_action.shape
        t=torch.linspace(0,1,horizon,device=base_action.device,dtype=base_action.dtype)
        x=torch.cat((context[:,None].expand(-1,horizon,-1),base_action,t[None,:,None].expand(batch,-1,-1)),dim=-1)
        raw=self.head(self.trunk(x));delta=(raw if self.proposal else raw.tanh())*self.caps
        return base_action+delta

class TorchFK(nn.Module):
    def __init__(self,robot):
        super().__init__();chain=[];node=robot.tip
        while node in robot.joints:
            parent,name,kind,origin,axis=robot.joints[node]
            if kind not in ('fixed','revolute','continuous'):raise ValueError('Only serial revolute FK supported')
            chain.append((name,kind,origin,axis));node=parent
        self.indices=[]
        for i,(name,kind,origin,axis) in enumerate(reversed(chain)):
            self.register_buffer(f'origin_{i}',torch.tensor(origin,dtype=torch.float64))
            self.register_buffer(f'axis_{i}',torch.tensor(axis,dtype=torch.float64));self.indices.append(None if kind=='fixed' else robot.names.index(name))
        self.register_buffer('root',torch.tensor(robot.root,dtype=torch.float64))

    def forward(self,q):
        out=self.root.to(q).expand(*q.shape[:-1],4,4)
        for i,index in enumerate(self.indices):
            out=out@getattr(self,f'origin_{i}').to(q)
            if index is None:continue
            axis=getattr(self,f'axis_{i}').to(q);x,y,z=axis.unbind()
            zero=x*0;K=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
            angle=q[...,index,None,None];I=torch.eye(3,device=q.device,dtype=q.dtype)
            rot=I+angle.sin()*K+(1-angle.cos())*(K@K)
            top=torch.cat((rot,torch.zeros(*q.shape[:-1],3,1,device=q.device,dtype=q.dtype)),-1)
            bottom=torch.tensor([0,0,0,1],device=q.device,dtype=q.dtype).expand(*q.shape[:-1],1,4)
            out=out@torch.cat((top,bottom),-2)
        return out

def parity(robot):
    """Meaningful numerical convention and derivative checks before pose training."""
    fk=TorchFK(robot);rng=np.random.default_rng(920005);q=torch.tensor(rng.uniform(-2,2,(12,6)),dtype=torch.float64,requires_grad=True)
    torchposes=fk(q);native=np.stack([robot.pose(x) for x in q.detach().numpy()]);difference=float(np.max(np.abs(torchposes.detach().numpy()-native)))
    jac=torch.autograd.functional.jacobian(lambda v:fk(v)[:3,:3],q[0])
    eps=1e-6;x=q[0].detach().numpy();finite=np.stack([(robot.pose(x+eps*np.eye(6)[i])[:3,:3]-robot.pose(x-eps*np.eye(6)[i])[:3,:3])/(2*eps) for i in range(6)],-1)
    gradient_error=float(np.max(np.abs(jac.numpy()-finite)))
    if difference>1e-10 or gradient_error>1e-8:raise ValueError('Differentiable FK parity failed')
    return dict(poses=12,maximum_matrix_error=difference,maximum_derivative_error=gradient_error)
