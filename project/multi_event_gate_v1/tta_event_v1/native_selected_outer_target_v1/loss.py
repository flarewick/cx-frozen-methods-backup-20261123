"""Two unchanged inner steps; supervise the direction at the next actual selected action."""
import torch
import torch.nn.functional as F

def packet_loss(model,p,*,create_graph):
 fast,inner=model.two_steps(p['hidden'],p['context'],p['command'],p['kind'],p['kernel_prediction'],p['target'],create_graph=create_graph)
 direction=model.direction(p['action_hidden'],p['action_logits'],p['action_reference'],fast)
 candidate=direction[0,p['action_valid_offsets']];target=p['action_teacher_target']
 assert candidate.shape==target.shape and target.shape[1]==12 and len(target)>0
 loss=F.smooth_l1_loss(candidate/.1,target/.1,reduction='mean')
 baseline=F.smooth_l1_loss(p['action_reference'][0,p['action_valid_offsets']]/.1,target/.1,reduction='mean')
 assert torch.isfinite(loss)and torch.isfinite(baseline)
 current_ok,current_trust=model.trust(p['next_hidden'],p['next_logits'],p['next_reference'],fast)
 action_ok,action_trust=model.trust(p['action_hidden'],p['action_logits'],p['action_reference'],fast)
 return loss,dict(loss=float(loss.detach()),zero_update_loss=float(baseline.detach()),inner=inner,current_query_trust_would_accept=current_ok,selected_action_trust_would_accept=action_ok,current_query_trust=current_trust,selected_action_trust=action_trust,valid_action_commands=len(target),packet_rows=len(p['target']),actual_feedback_query=p['actual_feedback_query'],offline_action_query=p['offline_action_query'],offline_outer_labels_only=True,applied_to_actor=False)
