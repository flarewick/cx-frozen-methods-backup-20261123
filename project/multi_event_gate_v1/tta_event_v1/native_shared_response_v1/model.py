"""Zero-matched local physical response and original ARM readout share fast128."""
import torch
from torch import nn
from torch.nn import functional as F

class LocalSharedResponse(nn.Module):
    def __init__(self, arm_weight, context_mean, context_scale, action_scale):
        super().__init__()
        values=dict(arm_weight=arm_weight,context_mean=context_mean,context_scale=context_scale,action_scale=action_scale)
        shapes=dict(arm_weight=(12,64),context_mean=(53,),context_scale=(53,),action_scale=(194,))
        for name,value in values.items():
            value=torch.as_tensor(value,dtype=torch.float32).detach().clone()
            if tuple(value.shape)!=shapes[name] or not torch.isfinite(value).all():raise ValueError('Fixed local response schema')
            self.register_buffer(name,value)
        if not bool((self.context_scale>0).all()and(self.action_scale>0).all()):raise ValueError('Positive fixed training scales')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.response=nn.Sequential(nn.Linear(122,64),nn.SiLU(),nn.Linear(64,388))

    def zero_fast(self):
        return tuple(torch.zeros(64,device=self.arm_weight.device,dtype=self.arm_weight.dtype,requires_grad=True)for _ in range(2))

    def forward(self,hidden,context,command,kind,kernel_prediction,fast):
        if hidden.shape!=(len(context),32,64)or context.shape[1:]!=(53,)or command.shape!=(len(context),194)or kind.shape!=(len(context),5)or kernel_prediction.shape!=(len(context),2):raise ValueError('Actual issuing-query hidden and exact local schema required')
        if any(not torch.isfinite(v).all()for v in(hidden,context,command,kind,kernel_prediction,*fast)):raise ValueError('Nonfinite local feedback')
        gain,offset=fast
        if gain.shape!=(64,)or offset.shape!=(64,):raise ValueError('Shared fast128 only')
        h=hidden.mean(1);x=(context-self.context_mean)/self.context_scale
        base=torch.cat((h,x,kind),1)
        changed=torch.cat((h+h*gain+offset,x,kind),1)
        increment=(self.response(changed)-self.response(base)).reshape(-1,2,194)
        return kernel_prediction+(increment*(command/self.action_scale)[:,None]).sum(-1)/194

    def direction(self,hidden,arm_logits,reference,fast):
        gain,offset=fast
        increment=F.linear(hidden*gain+offset,self.arm_weight)
        return(reference+.1*(torch.tanh(arm_logits+increment)-torch.tanh(arm_logits+torch.zeros_like(increment)))).clamp(-.1,.1)

    def two_steps(self,hidden,context,command,kind,kernel_prediction,target,*,fast=None,create_graph=False,online=False):
        if online and any(p.requires_grad for p in self.response.parameters()):raise ValueError('Online response weights must be frozen')
        if fast is None:fast=self.zero_fast()
        else:fast=tuple(v.detach().clone().requires_grad_(True)for v in fast)
        if target.shape!=kernel_prediction.shape or not torch.isfinite(target).all():raise ValueError('Actual finite arrived response target')
        losses=[];norms=[]
        for _ in range(2):
            prediction=self(hidden,context,command,kind,kernel_prediction,fast)
            loss=((prediction-target)/.003125).square().mean()
            grads=torch.autograd.grad(loss,fast,create_graph=create_graph)
            norm=torch.sqrt(sum(g.square().sum()for g in grads)+1e-24)
            if not torch.isfinite(norm):raise ValueError('Nonfinite feedback gradient')
            scale=torch.clamp(1/norm,max=1.)
            fast=tuple(p-.05*scale*g for p,g in zip(fast,grads))
            losses.append(float(loss.detach()));norms.append(float(norm.detach()))
        return fast,dict(losses=losses,gradient_norms=norms,steps=2,lr=.05,clip=1.)

    def trust(self,hidden,logits,reference,fast,anchors=()):
        with torch.no_grad():
            candidate=self.direction(hidden,logits,reference,fast)
            norm=float(torch.sqrt(sum(v.square().sum()for v in fast)))
            shift=float((candidate-reference).abs().max())
            a=candidate.flatten();b=reference.flatten();an=float(a.norm());bn=float(b.norm())
            cosine=1. if an==bn==0 else(0. if min(an,bn)==0 else float(F.cosine_similarity(a[None],b[None],eps=1e-12)[0]))
            drift=max([float((self.direction(h,l,r,fast)-r).abs().max())for h,l,r in anchors]+[0.])
            accepted=bool(torch.isfinite(candidate).all()and all(torch.isfinite(v).all()for v in fast)and norm<=.25 and shift<=.02 and cosine>=.8 and drift<=.02)
        return accepted,dict(fast_norm=norm,action_shift_max=shift,direction_cosine=cosine,anchor_drift_max=drift)
