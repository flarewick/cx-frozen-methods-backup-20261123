"""Real174 future selected-state label bindings; gradients only, no optimizer or rollout."""
from pathlib import Path
import collections,json
import torch
from .data import T,read,sha,inventory,SelectedReader
from .loss import packet_loss
from native_direction_training_v2.train import model_for
from native_policy_execution_v1.common import write
C=Path(__file__).resolve().parent;O=T/'outputs/native_selected_outer_target_v1'
def main():
 assert not O.exists(),'Preserve prior checks'
 torch.set_num_threads(2);torch.set_float32_matmul_precision('high');before=torch.random.get_rng_state().clone()
 manifest,cases,plans,excluded=inventory();reader=SelectedReader();counts=collections.Counter();examples={};proofs=[]
 for plan in plans:
  p=plan['packet'];case=cases[p['case_id']];query=plan['alignment']['next_selected_at_or_after_feedback']['query']
  assert query in next(r['selected_queries']for r in read(T/'outputs/native_outer_alignment_audit_v1/report.json')['cases']if r['case_id']==p['case_id'])
  value=reader.packet_for_action(p,case['queries'],query);counts[case['task']+'/'+case['split']]+=1
  if p['split']=='adapter_fit'and case['task']not in examples:examples[case['task']]=(p,value)
  proofs.append(dict(case_id=p['case_id'],split=p['split'],source_query=p['source_query'],feedback_query=p['query'],offline_action_query=query,valid_selected_action_commands=len(value['action_valid_offsets'])))
 assert len(proofs)==174 and len(excluded)==87 and set(examples)=={'stack_blocks_two','stack_bowls_two'}
 auth=read(T/'outputs/native_feedback_confirm_v5/direction_training_authorization.json')
 with torch.random.fork_rng(devices=[]):model,_,_=model_for(auth)
 gradients=[]
 for task,(p,value)in examples.items():
  model.zero_grad(set_to_none=True);loss,report=packet_loss(model,value,create_graph=True);loss.backward()
  grads=[param.grad for param in model.response.parameters()];assert all(g is not None and torch.isfinite(g).all()for g in grads)
  norm=float(torch.sqrt(sum(g.square().sum()for g in grads)));assert norm>0
  null=dict(value,target=value['kernel_prediction'].clone());fast,_=model.two_steps(null['hidden'],null['context'],null['command'],null['kind'],null['kernel_prediction'],null['target'],create_graph=False)
  assert all(torch.count_nonzero(f)==0 for f in fast)
  with torch.no_grad():assert torch.equal(model.direction(null['action_hidden'],null['action_logits'],null['action_reference'],fast),null['action_reference'])
  gradients.append(dict(task=task,case_id=p['case_id'],outer_gradient_norm=norm,physical_null_future_direction_exact=True,**report))
 try:reader.packet_for_action(plans[0]['packet'],cases[plans[0]['packet']['case_id']]['queries'],None)
 except ValueError:pass
 else:raise AssertionError('No-future packet received fictitious target')
 assert torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 write(O/'CPU_interface_checks.json',dict(passed=True,original_identities_retained=100,all261_packets_accounted=True,eligible_outer_packets=174,retained_no_future_packets=excluded,counts=dict(counts),actual_state_teacher_command_label_bindings=proofs,gradient_checks=gradients,weights_fitted=0,optimizer_steps=0,active_checkpoint_unchanged=True,online_binding_unchanged=True,inner_recipe_unchanged=True,CUDA_initialized=False,new_independent_scenes=0,TTA_utility_proven=False,source_sha256={str(p):sha(p)for p in C.glob('*.py')}))
 print('SELECTED_OUTER_TARGET_REAL_BINDINGS_GRADIENT_NULL_PASS',dict(counts),len(proofs),len(excluded),flush=True)
if __name__=='__main__':main()
