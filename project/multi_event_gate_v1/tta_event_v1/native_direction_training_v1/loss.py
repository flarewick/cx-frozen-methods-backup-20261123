"""Exactly two differentiable fixed inner steps and original valid-only ARM loss."""
import torch
import torch.nn.functional as F

def packet_loss(model,p,*,create_graph):
 fast,inner=model.two_steps(p['hidden'],p['context'],p['command'],p['kind'],p['kernel_prediction'],p['target'],create_graph=create_graph)
 direction=model.direction(p['next_hidden'],p['next_logits'],p['next_reference'],fast)
 candidate=direction[0,p['valid_offsets']];target=p['teacher_target']
 assert candidate.shape==target.shape and target.shape[1]==12 and len(target)>0
 loss=F.smooth_l1_loss(candidate/.1,target/.1,reduction='mean')
 baseline=F.smooth_l1_loss(p['next_reference'][0,p['valid_offsets']]/.1,target/.1,reduction='mean')
 assert torch.isfinite(loss)and torch.isfinite(baseline)
 accepted,trust=model.trust(p['next_hidden'],p['next_logits'],p['next_reference'],fast)
 return loss,dict(loss=float(loss.detach()),zero_update_loss=float(baseline.detach()),inner=inner,trust=trust,
  trust_would_accept=accepted,valid_next_commands=len(target),packet_rows=len(p['target']),offline_only=True,applied_to_actor=False)
