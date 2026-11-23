from native_fit_support_review_v1.support import minimum_fit,AUTH
"""Fixed additional1000 Btheta steps; immutable old093 and no online teacher input."""
from pathlib import Path
import argparse,json,time,uuid
import numpy as np,torch
from .data import load_all
from native_object_expert_train_v1.common import T,read,sha
from native_feedback_after_train_v1.loss import packet_loss
from native_visible_response_bridge_v1.bridge import load_model
from native_shared_response_v1.session import state_sha
TASKS=('stack_blocks_two','stack_bowls_two','beat_block_hammer','turn_switch','adjust_bottle','grab_roller','open_laptop')
O=T/'outputs/native_onpolicy_calibration_fit_v7'
def write(p,value):
 p=Path(p);assert p.resolve().is_relative_to(O.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def evaluate(model,pools):
 groups={}
 for task in TASKS:
  records=[]
  for example in pools[task]:
   _,report=packet_loss(model,example['tensors'],create_graph=False);records.append(dict(key=example['key'],**report))
  assert records
  groups[task]=dict(packets=len(records),mean_loss=float(np.mean([r['loss']for r in records])),mean_zero_update_loss=float(np.mean([r['zero_update_loss']for r in records])),records=records)
 return dict(groups=groups,report_only=True,no_checkpoint_selection=True,native_rollout=False,TTA_utility_proven=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 data,evidence=load_all() # Mandatory full data gates precede model AND optimizer creation.
 fit={t:[e for r in data if r['task']==t and r['split']=='adapter_fit'for e in r['examples']]for t in TASKS}
 val={t:[e for r in data if r['task']==t and r['split']=='adapter_validation'for e in r['examples']]for t in TASKS}
 for task in TASKS:
  fit[task].sort(key=lambda e:tuple(e['key']));val[task].sort(key=lambda e:tuple(e['key']));assert len(fit[task])>=minimum_fit() and val[task]
 print('SEVEN_TASK_BTHETA_FIXED_TRAINING_DATA_READY',json.dumps({t:dict(FIT=len(fit[t]),VAL=len(val[t]))for t in TASKS}),flush=True)
 if not a.run:return
 if any((T/'outputs'/version/'runs').exists()for version in('native_onpolicy_calibration_fit_v1','native_onpolicy_calibration_fit_v2','native_onpolicy_calibration_fit_v3','native_onpolicy_calibration_fit_v4','native_onpolicy_calibration_fit_v5','native_onpolicy_calibration_fit_v6','native_onpolicy_calibration_fit_v7')):raise ValueError('Prior fitting exists: never reset the additional1000-step budget')
 plan_path=T/'outputs/native_feedback_after_train_v1/data_plan.json';plan=read(plan_path);recipe=plan['calibration_recipe_intent'];assert recipe['updates']==1000 and recipe['packets_per_task']==2 and recipe['balanced_tasks']==7 and recipe['batch_packets']==14
 initial_q=T/plan['initial_response'];model,old_qualification=load_model(initial_q);assert sha(old_qualification['training_checkpoint'])==plan['initial_checkpoint_sha256']
 model.response.requires_grad_(True);buffers={k:v.detach().clone()for k,v in model.named_buffers()};initial_state=state_sha(model.response.state_dict())
 sources=dict(old_qualification['source_sha256'])
 sources.update({str(f):sha(f)for directory in (Path(__file__).parent,T/'native_onpolicy_calibration_data_v60')for f in directory.glob('*.py')})
 for f in(T/'native_feedback_after_train_v1/loss.py',T/'native_visible_response_bridge_v1/bridge.py',T/'native_action_visible_response_v1/model.py',T/'native_direction_visible_training_v1/factory.py',T/'native_shared_response_v1/model.py',T/'native_shared_response_v1/session.py'):sources[str(f)]=sha(f)
 bindings=dict(evidence['input_sha256'])
 for f in (plan_path,initial_q,Path(old_qualification['training_checkpoint']),T/'native_feedback_after_train_v1/loss.py',T/'native_action_visible_response_v1/model.py',T/'native_shared_response_v1/model.py'):bindings[str(f)]=sha(f)
 protocol=dict(minimum_distinct_FIT_cases=minimum_fit(),exploratory_support_exception=minimum_fit()==7,support_authorization_file=str(AUTH)if minimum_fit()==7 else None,frozen=True,initialization='Continue original093 Btheta only; original retained',original_qualification=str(initial_q),initial_response_state_sha256=initial_state,recipe=recipe,AdamW_betas=[.9,.999],AdamW_eps=1e-8,AdamW_amsgrad=False,sampler='RandomState17 fixed TASKS order, two distinct FIT packets each, total14',outer_target=plan['outer_supervision'],actual_full_states_and_independent_recovery_labels=True,short_interface_probes_never_fitting_data=True,online_method=plan['frozen_online_method'],new_tasks_are_offline_training_tasks=True,source_sha256=sources,input_sha256=bindings,evidence={k:v for k,v in evidence.items()if k!='input_sha256'},fixed_final_checkpoint=1000,validation_steps=[0,1000],native_efficacy_NOT_proven=True,new_qualification_required_before_deployment=True)
 run=O/'runs'/('run_'+uuid.uuid4().hex);write(run/'protocol.json',protocol)
 optimizer=torch.optim.AdamW(model.response.parameters(),lr=.0003,weight_decay=.0001,betas=(.9,.999),eps=1e-8,amsgrad=False);rng=np.random.RandomState(17);completed=0;phase='validation0';start=time.monotonic();trace=(run/'steps.jsonl').open('x')
 try:
  write(run/'validation_step0000.json',evaluate(model,val))
  for step in range(1,1001):
   batch=[fit[t][int(i)]for t in TASKS for i in rng.choice(len(fit[t]),2,replace=False)];assert len(batch)==14 and len({tuple(e['key'])for e in batch})==14
   optimizer.zero_grad(set_to_none=True);losses=[];phase='backward'
   for e in batch:
    loss,report=packet_loss(model,e['tensors'],create_graph=True);(loss/14).backward();losses.append(report['loss'])
   assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.response.parameters())
   norm=float(torch.sqrt(sum(p.grad.square().sum()for p in model.response.parameters())));phase='optimizer_step_started';optimizer.step();completed+=1;phase='optimizer_step_completed'
   assert all(torch.isfinite(p).all()for p in model.response.parameters())and all(torch.equal(v,buffers[k])for k,v in model.named_buffers())
   row=dict(step=step,loss=float(np.mean(losses)),outer_gradient_norm=norm,batch=[e['key']for e in batch],seconds=time.monotonic()-start);trace.write(json.dumps(row,allow_nan=False)+'\n');trace.flush()
   if step%100==0:
    with (run/('recovery_step%04d.pt'%step)).open('xb')as f:torch.save(dict(step=step,response=model.response.state_dict(),optimizer=optimizer.state_dict(),protocol_sha256=sha(run/'protocol.json'),deployable=False),f)
    print('SEVEN_TASK_BTHETA_FIXED_STEP',step,row['loss'],flush=True)
  assert completed==1000;phase='validation1000';write(run/'validation_step1000.json',evaluate(model,val))
  for f,digest in {**sources,**bindings}.items():assert sha(f)==digest,('Changed frozen fitting input/source',f)
  for r in data:
   for e in r['examples']:
    x=e['tensors']
    with torch.no_grad():
     assert torch.equal(model(x['hidden'],x['context'],x['command'],x['kind'],x['kernel_prediction'],model.zero_fast()),x['kernel_prediction'])
     assert torch.equal(model.direction(x['action_hidden'],x['action_logits'],x['action_reference'],model.zero_fast()),x['action_reference'])
  from .coherence import audit as audit_coherence
  all_examples={tuple(e['key']):e['tensors']for r in data for e in r['examples']};coherence=audit_coherence(model,all_examples);write(run/'postfit_coherence.json',coherence)
  checkpoint=run/'response_step1000.pt'
  with checkpoint.open('xb')as f:torch.save(dict(step=1000,response=model.response.state_dict(),buffers=dict(model.named_buffers()),protocol_sha256=sha(run/'protocol.json'),deployable=False),f)
  write(run/'result.json',dict(complete=True,actual_updates=1000,initial_response_state_sha256=initial_state,response_state_sha256=state_sha(model.response.state_dict()),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),model_kind=model.model_kind,fixed_buffers_unchanged=True,zero_fast_prediction_and_action_exact=True,postfit_action_visible_coherence_passed=coherence['passed'],protocol_sha256=sha(run/'protocol.json'),CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),new_offline_training_tasks=list(TASKS[2:]),data=evidence,qualification_pending_actual_host_exit0=True,independent_native_validation_pending=True,TTA_utility_proven=False,seconds=time.monotonic()-start))
  print('FIXED_SEVEN_TASK_BTHETA_FIT_DONE_AWAIT_ACTUAL_EXIT0',str(run),flush=True)
 except BaseException as exc:
  write(run/'failure.json',dict(type=type(exc).__name__,reason=str(exc),phase=phase,completed_optimizer_steps=completed,budget_reset_NOT_authorized=True));raise
 finally:trace.close()
if __name__=='__main__':main()
