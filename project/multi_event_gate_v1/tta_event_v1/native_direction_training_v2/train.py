"""Fixed1000 CPU B_theta updates, enabled only by the complete original100 cache."""
import argparse,json,time,uuid
from pathlib import Path
import numpy as np
import torch
from .data import T,ARM,TASKS,CANDIDATE,KERNEL,read,sha,full_manifest,PacketReader
from native_direction_training_v1.loss import packet_loss
from native_shared_response_v1.model import LocalSharedResponse
from native_shared_response_v1.session import state_sha
from native_delayed_query_v1.predictor import FrozenPredictor
from cx_tta_feedback_confirm5.query import build_head
O=T/'outputs/native_direction_training_v2'
RECIPE=dict(seed=17,optimizer='AdamW',updates=1000,learning_rate=.0003,weight_decay=.0001,packets_per_batch=16,packets_per_task=8)
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def key(p):return p['case_id'],p['source_query'],p['query']
def model_for(auth):
 config=read(T/'outputs/native_feedback_confirm_v5/protocol.json')
 with torch.random.fork_rng(devices=[]):head,_=build_head(config['training'])
 predictor=FrozenPredictor();s=predictor.state
 kernel=T/'outputs/native_phase_feedback_v1/frozen_auxiliary/predictor.npz';assert sha(kernel)==KERNEL
 model=LocalSharedResponse(head.action_head[2].weight.detach().numpy()[ARM],s['context_mean'].copy(),s['context_scale'].copy(),s['action_scale'].copy()).eval()
 assert all(name.startswith('response.')for name,p in model.named_parameters()if p.requires_grad)
 return model,config,kernel

def evaluate(model,metadata,examples):
 rows=[]
 for p in metadata:
  loss,report=packet_loss(model,examples[key(p)],create_graph=False)
  rows.append(dict(case_id=p['case_id'],source_query=p['source_query'],query=p['query'],**report))
 assert rows
 return dict(packets=len(rows),mean_loss=float(np.mean([r['loss']for r in rows])),mean_zero_update_loss=float(np.mean([r['zero_update_loss']for r in rows])),
  trust_accept_fraction=float(np.mean([r['trust_would_accept']for r in rows])),records=rows,
  only_original_validation_expert_states=True,checkpoint_selection=False,policy_rollout_performed=False,TTA_utility_proven=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
 torch.set_num_threads(2);torch.set_float32_matmul_precision('high');assert not torch.cuda.is_initialized()
 # No directories/models/optimizer are created until full original100 support has passed.
 data,cases,pools=full_manifest(a.manifest)
 authfile=T/'outputs/native_feedback_confirm_v5/direction_training_authorization.json';auth=read(authfile)
 assert auth['candidate_identity']==CANDIDATE and auth['direction_training_allowed']and auth['actual_host_return0']and auth['resource_qualification_verified']
 for field,hkey in(('independent_feedback_report','independent_feedback_report_sha256'),('actual_host_exit_file','actual_host_exit_sha256')):assert sha(auth[field])==auth[hkey]
 feedback=read(auth['independent_feedback_report']);assert feedback['complete']and feedback['independent_feedback_gate_passed']
 assert all(auth['fixed_direction_recipe'][k]==v for k,v in RECIPE.items())and auth['fixed_direction_recipe']['trained_parameters']=='B_theta only'
 checksfile=T/'outputs/native_direction_training_v1/CPU_interface_checks.json';checks=read(checksfile);assert checks['passed']and checks['actual_packets']==12 and checks['valid_next_commands']==379 and checks['optimizer_steps']==0
 for filename in('data.py','loss.py','cpu_checks.py'):
  f=T/'native_direction_training_v1'/filename;assert checks['source_sha256'][str(f)]==sha(f)
 gatefile=O/'CPU_reader_gate_checks.json';gate=read(gatefile);assert gate['passed']and gate['optimizer_steps']==0
 for file,h in gate['source_sha256'].items():assert sha(file)==h
 plan=dict(original_cases=100,original_fit_cases=80,original_validation_cases=20,
  packet_counts={task:{split:len(v)for split,v in groups.items()}for task,groups in pools.items()},recipe=auth['fixed_direction_recipe'],
  device='cpu',torch_threads=2,seed=17,optimizer_steps=0,TTA_utility_proven=False)
 print('FIXED_DIRECTION_TRAINING_READY',json.dumps(plan),flush=True)
 if not a.run:return
 if(O/'runs').exists():raise ValueError('A training run exists; inspect its exact update count before any resumption, never repeat the fixed budget blindly')
 model,config,kernel=model_for(auth);reader=PacketReader();examples={}
 for packet in data['packets']:examples[key(packet)]=reader.packet(packet,cases[packet['case_id']]['queries'])
 fit={task:pools[task]['adapter_fit']for task in TASKS};validation=[p for task in TASKS for p in pools[task]['adapter_validation']]
 fixed_buffers={name:v.detach().clone()for name,v in model.named_buffers()};initial_state=state_sha(model.response.state_dict())
 sources={str(f):sha(f)for f in Path(__file__).parent.glob('*.py')}
 sources.update({str(T/'native_direction_training_v1'/name):sha(T/'native_direction_training_v1'/name)for name in ('data.py','loss.py','cpu_checks.py')})
 for f in(T/'native_shared_response_v1/model.py',T/'native_shared_response_v1/session.py'):sources[str(f)]=sha(f)
 inputs={str(a.manifest.resolve()):sha(a.manifest),str(authfile):sha(authfile),str(checksfile):sha(checksfile),str(gatefile):sha(gatefile),str(kernel):sha(kernel),
  str(Path(config['training'])/'adapter_seed0.pt'):sha(Path(config['training'])/'adapter_seed0.pt'),**reader.observed_hashes}
 protocol=dict(frozen=True,recipe=auth['fixed_direction_recipe'],data=plan,source_sha256=sources,input_sha256=inputs,
  candidate_identity=CANDIDATE,batch_sampler='NumPy RandomState17, fixed blocks then bowls,8 uniform packets/task without replacement within each batch',
  inner_state='fresh zero fast per sampled meta packet, exactly two differentiable SGD steps',outer_loss='unmodified valid-only SmoothL1(candidate/.1,teacher_residual/.1); mean of16 packet losses',
  online_trust_NOT_bypassed=True,trust_during_offline_training='reported only; online FeedbackSession remains fully enforced',
  AdamW_betas=[.9,.999],AdamW_eps=1e-8,AdamW_amsgrad=False,outer_gradient_clipping=None,
  validation_at_steps=[0,1000],final_checkpoint_step=1000,no_checkpoint_or_hyperparameter_selection=True,
  CPU_only=True,TTA_utility_proven=False)
 run=O/'runs'/('run_'+uuid.uuid4().hex);write(run/'protocol.json',protocol)
 optimizer=torch.optim.AdamW(model.response.parameters(),lr=.0003,weight_decay=.0001,betas=(.9,.999),eps=1e-8,amsgrad=False)
 rng=np.random.RandomState(17);started=time.monotonic();step=0;completed_updates=0;phase='before_training';trace=(run/'steps.jsonl').open('x')
 try:
  write(run/'validation_step0000.json',evaluate(model,validation,examples))
  for step in range(1,1001):
   selected=[fit[task][int(i)]for task in TASKS for i in rng.choice(len(fit[task]),size=8,replace=False)]
   assert len(selected)==16 and len({key(x)for x in selected})==16
   optimizer.zero_grad(set_to_none=True);reports=[]
   for packet in selected:
    loss,report=packet_loss(model,examples[key(packet)],create_graph=True);(loss/16).backward();reports.append(report)
   assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.response.parameters())
   norm=float(torch.sqrt(sum(p.grad.square().sum()for p in model.response.parameters())))
   phase='optimizer_step_started';optimizer.step();completed_updates+=1;phase='optimizer_step_completed'
   assert all(torch.isfinite(p).all()for p in model.response.parameters())and all(torch.equal(fixed_buffers[k],v)for k,v in model.named_buffers())
   row=dict(step=step,batch=[list(key(p))for p in selected],mean_loss=float(np.mean([r['loss']for r in reports])),outer_gradient_norm=norm,
     trust_accept_fraction=float(np.mean([r['trust_would_accept']for r in reports])),seconds=time.monotonic()-started)
   trace.write(json.dumps(row,allow_nan=False)+'\n');trace.flush()
   if step%100==0:
    # Recovery receipts are fixed-step snapshots, never candidates selected by validation/outcomes.
    state=rng.get_state();checkpoint=dict(step=step,response=model.response.state_dict(),optimizer=optimizer.state_dict(),
      numpy_algorithm=state[0],numpy_keys=torch.tensor(state[1].astype(np.int64)),numpy_position=state[2],numpy_has_gauss=state[3],numpy_cached_gauss=state[4],
      protocol_sha256=sha(run/'protocol.json'),deployable=False)
    file=run/('recovery_step%04d.pt'%step)
    with file.open('xb')as f:torch.save(checkpoint,f)
    print('FIXED_DIRECTION_STEP',step,row['mean_loss'],row['seconds'],flush=True)
  assert completed_updates==1000
  phase='final_validation';final=evaluate(model,validation,examples);write(run/'validation_step1000.json',final)
  for path,h in {**sources,**inputs}.items():assert sha(path)==h,('source_changed_during_training',path)
  for example in examples.values():
   with torch.no_grad():
    assert torch.equal(model(example['hidden'],example['context'],example['command'],example['kind'],example['kernel_prediction'],model.zero_fast()),example['kernel_prediction'])
    assert torch.equal(model.direction(example['next_hidden'],example['next_logits'],example['next_reference'],model.zero_fast()),example['next_reference'])
  assert not torch.cuda.is_initialized();checkpoint=run/'response_step1000.pt'
  with checkpoint.open('xb')as f:torch.save(dict(response=model.response.state_dict(),buffers=fixed_buffers,step=1000,protocol_sha256=sha(run/'protocol.json'),candidate_identity=CANDIDATE),f)
  qualification=dict(frozen=False,awaiting_actual_host_exit0=True,direction_trained=True,independent_feedback_gate_passed=True,
    independent_feedback_report=auth['independent_feedback_report'],independent_feedback_report_sha256=auth['independent_feedback_report_sha256'],
    training_checkpoint=str(checkpoint),training_checkpoint_sha256=sha(checkpoint),training_protocol=str(run/'protocol.json'),training_protocol_sha256=sha(run/'protocol.json'),
    response_state_sha256=state_sha(model.response.state_dict()),source_sha256=sources,original100_cache_manifest=str(a.manifest.resolve()),original100_cache_sha256=sha(a.manifest),
    actual_updates=1000,trained_parameters='B_theta only',initial_response_sha256=initial_state,validation_report_only=True,
    independent_action_and_task_benefit_still_required=True,TTA_utility_proven=False)
  write(run/'qualification_pending.json',qualification)
  write(run/'result.json',dict(complete=True,actual_updates=1000,seconds=time.monotonic()-started,qualification_pending_sha256=sha(run/'qualification_pending.json'),
    validation_mean_loss=final['mean_loss'],validation_zero_update_loss=final['mean_zero_update_loss'],CPU_only=True,online_updates=0,TTA_utility_proven=False))
 except BaseException as exc:
  write(run/'failure.json',dict(type=type(exc).__name__,reason=str(exc),current_loop_step=step,completed_optimizer_updates=completed_updates,optimizer_step_may_be_partial=phase=='optimizer_step_started',phase=phase,
   no_automatic_retry=True,deployable_checkpoint_qualification=False));raise
 finally:trace.close()
if __name__=='__main__':main()
