"""Couple auxiliary fast response to the hidden changes visible to actual ARM readout."""
import torch
from torch.nn import functional as F
from native_shared_response_v1.model import LocalSharedResponse

class ActionVisibleResponse(LocalSharedResponse):
 model_kind='ARM_rowspace_fast_increment_v1'
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  with torch.no_grad():
   weight=self.arm_weight.double();_,singular,vh=torch.linalg.svd(weight,full_matrices=False)
   if int(torch.linalg.matrix_rank(weight))!=12:raise ValueError('Original full-row-rank ARM12 readout required')
   projection=(vh.T@vh).to(dtype=self.arm_weight.dtype)
  self.register_buffer('ARM_visible_projection',projection)
  # A distinct immutable state namespace prevents the old unprojected
  # qualification/loader from accepting these weights as its old model.
  self.response.register_buffer('ARM_visible_interface_version',torch.tensor(1,dtype=torch.int64))
 def forward(self,hidden,context,command,kind,kernel_prediction,fast):
  if hidden.shape!=(len(context),32,64)or context.shape[1:]!=(53,)or command.shape!=(len(context),194)or kind.shape!=(len(context),5)or kernel_prediction.shape!=(len(context),2):raise ValueError('Actual issuing-query hidden and exact local schema required')
  if any(not torch.isfinite(v).all()for v in(hidden,context,command,kind,kernel_prediction,*fast)):raise ValueError('Nonfinite local feedback')
  gain,offset=fast
  if gain.shape!=(64,)or offset.shape!=(64,):raise ValueError('Shared fast128 only')
  h=hidden.mean(1);x=(context-self.context_mean)/self.context_scale
  # Keep all original hidden conditioning. Only the fast perturbation is
  # projected; W P = W, so invisible ARM offsets cannot fit the proxy alone.
  delta=F.linear(h*gain+offset,self.ARM_visible_projection)
  base=torch.cat((h,x,kind),1);changed=torch.cat((h+delta,x,kind),1)
  increment=(self.response(changed)-self.response(base)).reshape(-1,2,194)
  return kernel_prediction+(increment*(command/self.action_scale)[:,None]).sum(-1)/194

def from_legacy(model):
 result=ActionVisibleResponse(model.arm_weight,model.context_mean,model.context_scale,model.action_scale)
 state=dict(result.response.state_dict());state.update(model.response.state_dict());result.response.load_state_dict(state,strict=True)
 return result.eval()
