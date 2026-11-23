"""Fixed1000 new response fit; all inner inputs unchanged, only outer action target aligned."""
import argparse,json,time,uuid
import numpy as np,torch
from .common import *
from .plan import prerequisites
from native_selected_outer_target_v1.data import inventory
from native_selected_outer_target_v1.qualified_reader import QualifiedSelectedReader
from native_selected_outer_target_v1.loss import packet_loss
from .factory import model_for
from native_shared_response_v1.session import state_sha
TASKS=('stack_blocks_two','stack_bowls_two')
def key(p):return p['case_id'],p['source_query'],p['query']
def evaluate(model,plans,examples):
 rows=[]
 for plan in plans:
  p=plan['packet'];loss,report=packet_loss(model,examples[key(p)],create_graph=False)
  rows.append(dict(case_id=p['case_id'],source_query=p['source_query'],**report))
 assert rows
 return dict(packets=len(rows),mean_loss=float(np.mean([r['loss']for r in rows])),mean_zero_update_loss=float(np.mean([r['zero_update_loss']for r in rows])),current_query_trust_accept_fraction=float(np.mean([r['current_query_trust_would_accept']for r in rows])),selected_action_trust_accept_fraction=float(np.mean([r['selected_action_trust_would_accept']for r in rows])),records=rows,validation_report_only=True,checkpoint_selection=False,native_rollout=False,TTA_utility_proven=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args()
 torch.set_num_threads(2);torch.set_float32_matmul_precision('high');assert not torch.cuda.is_initialized()
 prereq=prerequisites();manifest,cases,plans,excluded=inventory()
 fit={task:[plan for plan in plans if cases[plan['packet']['case_id']]['task']==task and plan['packet']['split']=='adapter_fit']for task in TASKS}
 for group in fit.values():group.sort(key=lambda x:(x['packet']['original_expert_index'],x['packet']['source_query'],x['packet']['query']));assert len(group)>=8
 validation=[plan for task in TASKS for plan in plans if cases[plan['packet']['case_id']]['task']==task and plan['packet']['split']=='adapter_validation']
 assert [len(fit[t])for t in TASKS]==[76,60]and len(validation)==38 and len(excluded)==87
 print('SELECTED_OUTER_FIXED_TRAINING_READY',json.dumps(dict(recipe=RECIPE,FIT={t:len(fit[t])for t in TASKS},VAL=len(validation),retained_no_outer_loss=len(excluded),original_identities=100,CUDA=False)),flush=True)
 if not a.run:return
 if(O/'runs').exists():raise ValueError('Prior training exists; inspect exact completed steps instead of resetting budget')
 authfile=T/'outputs/native_feedback_confirm_v5/direction_training_authorization.json';auth=read(authfile)
 model,config,kernel=model_for(auth);reader=QualifiedSelectedReader();examples={}
 for plan in plans:
  record=plan['packet'];examples[key(record)]=reader.packet_for_action(record,cases[record['case_id']]['queries'],plan['alignment']['next_selected_at_or_after_feedback']['query'])
 sources={str(p):sha(p)for p in C.glob('*.py')}
 for directory,names in((T/'native_selected_outer_target_v1',('data.py','loss.py','qualified_reader.py','cpu_checks.py','gate_checks.py')),(T/'native_shared_response_v1',('model.py','session.py')),(T/'native_action_visible_response_v1',('model.py',)),(T/'native_visible_response_bridge_v1',('bridge.py',)),(T/'native_direction_training_v2',('train.py','data.py')),(T/'native_direction_training_v1',('data.py',))):
  for name in names:sources[str(directory/name)]=sha(directory/name)
 inputs=dict(prereq['input_sha256']);inputs.update(reader.observed_hashes)
 old_qualification=next((T/'outputs/native_direction_selected_training_v1/runs').glob('*/qualification.json'))
 for file in(authfile,T/'outputs/native_direction_cache_contract_v2/qualified_manifest.json',Path(config['training'])/'adapter_seed0.pt',kernel,old_qualification,Path(read(old_qualification)['training_checkpoint'])):inputs[str(file)]=sha(file)
 protocol=dict(frozen=True,model_kind=model.model_kind,projection_only_fast_increment=True,original_static_hidden_conditioning_preserved=True,recipe=RECIPE,outer_target='first actual selected intervention at/after feedback; teacher SAME state absolute ARM minus once-base clip±.1, returned offsets only',inner='unchanged observed packet/issuing hidden, two differentiable SGD .05 clip1, physical prediction sigma.003125',online='unchanged128 fast/trust .25/.02/cos.8/anchor.02; no future outer labels online',sampler='RandomState17, blocks then bowls,8/task without replacement per batch',AdamW_betas=[.9,.999],AdamW_eps=1e-8,AdamW_amsgrad=False,outer_gradient_clipping=None,FIT_packets={t:len(fit[t])for t in TASKS},VAL_packets=38,original_identities=100,original_FIT_VAL_split=[80,20],retained_no_future_packets=excluded,eligible_fit_cases={t:len({r['packet']['case_id']for r in fit[t]})for t in TASKS},source_sha256=sources,input_sha256=inputs,old_GPU_mask_source_resource_qualification_passed=manifest['source_resource_qualification_passed'],old_GPU_mask_source_actual_host_exit_code=manifest['source_actual_host_exit_code'],new_training_authority='User persistent goal authorizes local TTA optimization; no new rollout/independent efficacy permission inferred from CPU metrics',initialization='same original seed17 architecture, from scratch; not continuation of previous1000',fixed_final_step=1000,validation_steps=[0,1000],no_model_selection=True,new_independent_scenes=0,TTA_utility_proven=False)
 run=O/'runs'/('run_'+uuid.uuid4().hex);write(run/'protocol.json',protocol)
 buffers={k:v.detach().clone()for k,v in model.named_buffers()};initial=state_sha(model.response.state_dict());optimizer=torch.optim.AdamW(model.response.parameters(),lr=.0003,weight_decay=.0001,betas=(.9,.999),eps=1e-8,amsgrad=False)
 rng=np.random.RandomState(17);start=time.monotonic();completed=0;phase='validation0';trace=(run/'steps.jsonl').open('x')
 try:
  write(run/'validation_step0000.json',evaluate(model,validation,examples))
  # Same evaluation labels for the previous frozen candidate; reporting only.
  from native_trained_response_bridge_v1.bridge import load_model
  oldmodel,_=load_model(old_qualification)
  write(run/'previous_candidate_same_selected_VAL.json',evaluate(oldmodel,validation,examples));del oldmodel
  for step in range(1,1001):
   chosen=[fit[task][int(i)]for task in TASKS for i in rng.choice(len(fit[task]),size=8,replace=False)];assert len(chosen)==16 and len({key(r['packet'])for r in chosen})==16
   optimizer.zero_grad(set_to_none=True);reports=[];phase='backward'
   for plan in chosen:
    loss,report=packet_loss(model,examples[key(plan['packet'])],create_graph=True);(loss/16).backward();reports.append(report)
   assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.response.parameters())
   norm=float(torch.sqrt(sum(p.grad.square().sum()for p in model.response.parameters())));phase='step_started';optimizer.step();completed+=1;phase='step_completed'
   assert all(torch.isfinite(p).all()for p in model.response.parameters())and all(torch.equal(v,buffers[k])for k,v in model.named_buffers())
   row=dict(step=step,batch=[list(key(r['packet']))for r in chosen],loss=float(np.mean([r['loss']for r in reports])),outer_gradient_norm=norm,seconds=time.monotonic()-start)
   trace.write(json.dumps(row,allow_nan=False)+'\n');trace.flush()
   if step%100==0:
    recovery=run/('recovery_step%04d.pt'%step)
    with recovery.open('xb')as f:torch.save(dict(step=step,response=model.response.state_dict(),optimizer=optimizer.state_dict(),protocol_sha256=sha(run/'protocol.json'),deployable=False),f)
    print('SELECTED_OUTER_FIXED_STEP',step,row['loss'],row['seconds'],flush=True)
  assert completed==1000;phase='final_validation';final=evaluate(model,validation,examples);write(run/'validation_step1000.json',final)
  for p,h in {**sources,**inputs}.items():assert sha(p)==h,('Changed selected outer training source/input',p)
  for e in examples.values():
   with torch.no_grad():
    assert torch.equal(model(e['hidden'],e['context'],e['command'],e['kind'],e['kernel_prediction'],model.zero_fast()),e['kernel_prediction'])
    assert torch.equal(model.direction(e['action_hidden'],e['action_logits'],e['action_reference'],model.zero_fast()),e['action_reference'])
  from .coherence import audit as audit_coherence
  coherence=audit_coherence(model,examples);write(run/'postfit_coherence.json',coherence)
  checkpoint=run/'response_step1000.pt'
  with checkpoint.open('xb')as f:torch.save(dict(step=1000,response=model.response.state_dict(),buffers={k:v for k,v in model.named_buffers()},protocol_sha256=sha(run/'protocol.json'),deployable=False),f)
  write(run/'result.json',dict(complete=True,model_kind=model.model_kind,projection_only_fast_increment=True,actual_updates=1000,initial_response_sha256=initial,response_state_sha256=state_sha(model.response.state_dict()),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),protocol_sha256=sha(run/'protocol.json'),FIT_packets=136,VAL_packets=38,retained_no_outer_loss_packets=87,original_cases=100,fixed_buffers_unchanged=True,postfit_action_visible_coherence_passed=coherence['passed'],zero_fast_prediction_and_action_exact=True,new_independent_scenes=0,CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.monotonic()-start,qualification_pending_actual_host_exit0=True,TTA_utility_proven=False))
  print('SELECTED_OUTER_FIXED1000_FIT_COMPLETE_AWAIT_ACTUAL_HOST0',run,flush=True)
 except BaseException as exc:
  write(run/'failure.json',dict(type=type(exc).__name__,reason=str(exc),phase=phase,completed_optimizer_updates=completed,new_budget_reset_NOT_authorized=True));raise
 finally:trace.close()
if __name__=='__main__':main()
