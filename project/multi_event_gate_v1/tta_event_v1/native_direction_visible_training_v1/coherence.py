"""Post-fit structural checks on every original selected-target example; no updates."""
import torch

def audit(model,examples):
 W=model.arm_weight.double();_,_,vh=torch.linalg.svd(W,full_matrices=True);N=(vh[12:].T@vh[12:]).float();P=model.ARM_visible_projection
 assert torch.allclose(W.float()@P,W.float(),atol=1e-6,rtol=0)
 assert int(model.response.ARM_visible_interface_version)==1
 fractions=[];null_shifts=[];directions=[]
 for e in examples.values():
  fast=model.zero_fast();prediction=model(e['hidden'],e['context'],e['command'],e['kind'],e['kernel_prediction'],fast)
  assert torch.equal(prediction,e['kernel_prediction'])
  loss=((prediction-e['target'])/.003125).square().mean();gain,offset=torch.autograd.grad(loss,fast)
  total=float(gain.square().sum()+offset.square().sum());fraction=float((N@offset).square().sum())/total if total else 0.
  assert fraction<1e-10;fractions.append(fraction)
  with torch.no_grad():
   assert torch.equal(model.direction(e['action_hidden'],e['action_logits'],e['action_reference'],model.zero_fast()),e['action_reference'])
   # Fixed W-derived null vector, not a data-selected probe.
   v=(vh[12]*.05).float();probe=(torch.zeros(64),v)
   changed=model(e['hidden'],e['context'],e['command'],e['kind'],e['kernel_prediction'],probe)
   null_shifts.append(float((changed-e['kernel_prediction']).abs().max()))
   directions.append(float((model.direction(e['action_hidden'],e['action_logits'],e['action_reference'],probe)-e['action_reference']).abs().max()))
 assert len(fractions)==174 and max(directions)<=1e-7
 return dict(passed=True,original_selected_packets=174,zero_fast_kernel_and_direction_exact=True,projection_and_marker_fixed=True,max_ARM_invisible_offset_gradient_energy_fraction=max(fractions),postfit_null_response_max_report_only=max(null_shifts),postfit_ARM_null_direction_max=max(directions),optimizer_steps=0,new_independent_scenes=0,TTA_utility_proven=False)
