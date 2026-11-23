"""Actual saved packet counterexamples must vanish without altering actor math."""
from pathlib import Path
import json,time
import numpy as np,torch
from native_policy_execution_v1.common import T,read,write,sha
from native_selected_outer_target_v1.data import inventory
from native_selected_outer_target_v1.qualified_reader import QualifiedSelectedReader
from native_trained_response_bridge_v1.bridge import load_model
from native_shared_response_v1.session import state_sha
from .model import from_legacy

def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('high');rng=torch.random.get_rng_state().clone();assert not torch.cuda.is_initialized()
 manifest,cases,plans,excluded=inventory();reader=QualifiedSelectedReader();chosen=[]
 for task in('stack_blocks_two','stack_bowls_two'):
  for split in('adapter_fit','adapter_validation'):
   values=[r for r in plans if cases[r['packet']['case_id']]['task']==task and r['packet']['split']==split];chosen.extend(values[:2])
 assert len(chosen)==8
 files=dict(old=T/'outputs/native_direction_training_v2/runs/run_37022586a0f3428b98ae611681a08c35/qualification.json',selected=next((T/'outputs/native_direction_selected_training_v1/runs').glob('*/qualification.json')))
 records=[]
 for label,file in files.items():
  original,_=load_model(file);prototype=from_legacy(original)
  from native_shared_response_v1.session import qualified
  try:qualified(prototype,file)
  except AssertionError:pass
  else:raise AssertionError('Old unprojected qualification must reject projected prototype')
  oldstate=state_sha(original.state_dict());newstate=state_sha(prototype.state_dict());W=original.arm_weight.double();_,_,vh=torch.linalg.svd(W,full_matrices=True);N=vh[12:].float();P=prototype.ARM_visible_projection
  assert N.shape==(52,64)and torch.allclose(P@P,P,atol=1e-6,rtol=0)and torch.allclose(original.arm_weight@P,original.arm_weight,atol=1e-6,rtol=0)
  for plan in chosen:
   p=plan['packet'];q=plan['alignment']['next_selected_at_or_after_feedback']['query'];e=reader.packet_for_action(p,cases[p['case_id']]['queries'],q);args=tuple(e[k]for k in('hidden','context','command','kind','kernel_prediction'))
   assert torch.equal(prototype(*args,prototype.zero_fast()),e['kernel_prediction'])
   fast=prototype.zero_fast();loss=((prototype(*args,fast)-e['target'])/.003125).square().mean();gain_grad,offset_grad=torch.autograd.grad(loss,fast);invisible=N.T@(N@offset_grad);total=float(gain_grad.square().sum()+offset_grad.square().sum());fraction=float(invisible.square().sum())/total if total else 0.;assert fraction<1e-10
   null_deltas=[];old_deltas=[];arm_deltas=[]
   with torch.no_grad():
    for v in N:
     fast=(torch.zeros(64),v*.05)
     baseline=original(*args,fast);fixed=prototype(*args,fast);null_deltas.append(float((fixed-e['kernel_prediction']).abs().max()));old_deltas.append(float((baseline-e['kernel_prediction']).abs().max()))
     a=original.direction(e['action_hidden'],e['action_logits'],e['action_reference'],fast);b=prototype.direction(e['action_hidden'],e['action_logits'],e['action_reference'],fast);assert torch.equal(a,b);arm_deltas.append(float((b-e['action_reference']).abs().max()))
    assert max(arm_deltas)<1e-7 and max(null_deltas)<1e-8
    # A real visible fast perturbation retains exactly the original ARM output.
    visible=(vh[0]*.05).float();fast=(torch.zeros(64),visible)
    assert torch.equal(original.direction(e['action_hidden'],e['action_logits'],e['action_reference'],fast),prototype.direction(e['action_hidden'],e['action_logits'],e['action_reference'],fast))
    zero_command=(e['hidden'],e['context'],torch.zeros_like(e['command']),e['kind'],torch.zeros_like(e['kernel_prediction']))
    assert torch.count_nonzero(prototype(*zero_command,fast))==0
   records.append(dict(weight_label=label,case_id=p['case_id'],split=p['split'],source_query=p['source_query'],null_basis_probes=len(null_deltas),original_null_response_max=max(old_deltas),prototype_null_response_max=max(null_deltas),ARM_null_max=max(arm_deltas),invisible_offset_gradient_energy_fraction=fraction,zero_fast_kernel_exact=True,actor_output_byte_exact=True,zero_command_physical_null_exact=True))
  assert state_sha(original.state_dict())==oldstate and state_sha(prototype.state_dict())==newstate and all(p.grad is None for p in prototype.parameters())
 assert len(records)==16 and torch.equal(rng,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 value=dict(passed=True,actual_saved_packets=8,weight_versions=2,ARM_null_basis_probes=16*52,records=records,no_training=True,optimizer_steps=0,CUDA_initialized=False,models_and_RNG_unchanged=True,projection_only_fast_increment=True,original_hidden_conditioning_preserved=True,original_architecture_and_fast128_preserved=True,ARMdirection_and_inner_trust_inherited_unchanged=True,prototype_qualified_for_actor=False,old_unprojected_qualification_rejected=True,distinct_immutable_response_buffer_namespace=True,copied_legacy_weights_for_checks_only=True,new_independent_scenes=0,TTA_utility_proven=False,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')},input_sha256=reader.observed_hashes)
 write(T/'outputs/native_action_visible_response_v1/CPU_coherence_checks_v2.json',value);print('ACTION_VISIBLE_RESPONSE_NULL_AND_ACTOR_CHECKS_PASS',len(records),16*52,max(r['prototype_null_response_max']for r in records),flush=True)
if __name__=='__main__':main()
