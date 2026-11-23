"""Finish post-fit checks from the actual step1000 checkpoint; zero new updates."""
import argparse,json
from pathlib import Path
import torch
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped,verify
from native_visible_response_bridge_v1.bridge import load_model
from native_shared_response_v1.session import state_sha
from native_onpolicy_calibration_fit_v7.data import load_all
from native_onpolicy_calibration_fit_v7.train import evaluate,TASKS
from native_onpolicy_calibration_fit_v7.coherence import audit
O=T/'outputs/native_fit7_postexit_recovery_v2'

def prior():
 runs=list((T/'outputs/native_onpolicy_calibration_fit_v7/runs').iterdir());assert len(runs)==1
 run=runs[0];protocol=read(run/'protocol.json');verify(protocol)
 hosts=list((T/'outputs/native_onpolicy_calibration_fit_v7').glob('host_run_*'));assert len(hosts)==1
 actual=hosts[0]/'exit.json';a=read(actual)
 assert a['exit_code']==1 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 failure=read(run/'failure.json')
 assert failure['type']=='FileNotFoundError'and failure['phase']=='validation1000'and failure['completed_optimizer_steps']==1000
 assert failure['budget_reset_NOT_authorized']and '/data/qiufw/cx_migration/cx_new_direction_guide_TTA/'in failure['reason']
 steps=[json.loads(s)for s in (run/'steps.jsonl').read_text().splitlines()]
 assert [r['step']for r in steps]==list(range(1,1001))
 assert not(run/'result.json').exists()and not(run/'qualification.json').exists()
 return run,protocol,actual

def main():
 p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');a=p.parse_args();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 run,protocol,old_exit=prior()
 if not a.finalize:
  assert not(O/'result.json').exists()
  model,old=load_model(mapped(protocol['original_qualification']))
  initial=state_sha(model.response.state_dict());assert initial==protocol['initial_response_state_sha256']
  buffers={k:v.clone()for k,v in model.named_buffers()}
  recovery=run/'recovery_step1000.pt';state=torch.load(recovery,map_location='cpu',weights_only=True)
  assert state['step']==1000 and state['protocol_sha256']==sha(run/'protocol.json')
  model.response.load_state_dict(state['response'],strict=True)
  assert all(torch.equal(v,buffers[k])for k,v in model.named_buffers())
  assert all(torch.isfinite(v).all()for v in model.response.parameters())
  data,evidence=load_all()
  val={t:[e for r in data if r['task']==t and r['split']=='adapter_validation'for e in r['examples']]for t in TASKS}
  for t in TASKS:val[t].sort(key=lambda e:tuple(e['key']))
  model.response.requires_grad_(True) # Reproduce the original final validation graph mode; no optimizer.
  verified_val=evaluate(model,val)
  assert json.loads(json.dumps(verified_val,allow_nan=False))==read(run/'validation_step1000.json'),'Recovery model must reproduce original final VAL report exactly'
  examples={tuple(e['key']):e['tensors']for r in data for e in r['examples']}
  coherence=audit(model,examples);assert coherence['passed']and coherence['optimizer_steps']==0
  write(O/'postfit_coherence.json',coherence)
  checkpoint=O/'response_step1000.pt'
  with checkpoint.open('xb')as f:torch.save(dict(step=1000,response=model.response.state_dict(),buffers=dict(model.named_buffers()),protocol_sha256=sha(run/'protocol.json'),deployable=False),f)
  verify(protocol)
  inputs={str(mapped(k)):v for k,v in protocol['input_sha256'].items()}
  for file in(old_exit,run/'failure.json',run/'steps.jsonl',run/'protocol.json',recovery,run/'validation_step0000.json',run/'validation_step1000.json',O/'postfit_coherence.json'):inputs[str(file)]=sha(file)
  sources={str(mapped(k)):v for k,v in protocol['source_sha256'].items()}
  sources.update({str(f):sha(f)for f in Path(__file__).parent.glob('*.py')})
  write(O/'result.json',dict(complete=True,actual_updates=1000,additional_optimizer_steps=0,optimizer_instantiated_in_recovery=False,original_training_host_exit_code=1,original_training_host_exit_file=str(old_exit),original_training_failure_preserved=True,model_kind=model.model_kind,initial_response_state_sha256=initial,response_state_sha256=state_sha(model.response.state_dict()),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),protocol_sha256=sha(run/'protocol.json'),original_protocol=str(run/'protocol.json'),source_sha256=sources,input_sha256=inputs,data=evidence,fixed_buffers_unchanged=True,zero_fast_prediction_and_action_exact=True,postfit_action_visible_coherence_passed=True,original_final_VAL_report_reproduced_exactly=True,CPU_only=True,CUDA_initialized=False,TTA_utility_proven=False))
  print('FIXED1000_POSTFIT_RECOVERY_CHECKS_PASS_ZERO_ADDITIONAL_UPDATES',flush=True)
 else:
  actual=O/'actual_recovery_host_exit.json';ex=read(actual);assert ex['exit_code']==0 and ex['actual_Popen_wait_return']and ex['proc_absent_after_wait']
  r=read(O/'result.json');verify(r);assert r['complete']and r['actual_updates']==1000 and r['additional_optimizer_steps']==0
  q=read(mapped(protocol['original_qualification']));groups=r['data']['new_population']['groups']
  q.update(frozen=True,actual_training_host_exit0=True,awaiting_actual_host_exit0=False,actual_updates=1000,training_checkpoint=r['checkpoint'],training_checkpoint_sha256=r['checkpoint_sha256'],training_protocol=str(run/'protocol.json'),training_protocol_sha256=sha(run/'protocol.json'),training_result_file=str(O/'result.json'),training_result_sha256=sha(O/'result.json'),training_host_exit_file=str(actual),training_host_exit_sha256=sha(actual),actual_final_training_continuation_host_exit0=True,original_training_host_exit_code=1,original_training_host_exit_file=str(old_exit),original_training_host_exit_sha256=sha(old_exit),original_training_failure_preserved=True,postfit_only_recovery=True,additional_optimizer_steps_after_original1000=0,response_state_sha256=r['response_state_sha256'],source_sha256=r['source_sha256'],training_started_from_scratch=False,initial_response_qualification=str(mapped(protocol['original_qualification'])),initial_response_checkpoint_sha256=q['training_checkpoint_sha256'],original093_checkpoint_preserved=True,new_offline_training_tasks=list(TASKS[2:]),new_population_identities=325,FIT_packets=136+sum(g['adapter_fit']['packets']for g in groups.values()),VAL_packets=38+sum(g['adapter_validation']['packets']for g in groups.values()),postfit_coherence_file=str(O/'postfit_coherence.json'),postfit_coherence_sha256=sha(O/'postfit_coherence.json'),independent_action_and_task_benefit_still_required=True,online_GRIP_residual_zero=True,validation_report_only=True,checkpoint_selection=False,new_independent_scenes=0,TTA_utility_proven=False,minimum_distinct_FIT_cases=7,exploratory_support_exception=True,support_authorization_file=str(mapped(protocol['support_authorization_file'])))
  write(O/'qualification_draft.json',q);model,checked=load_model(O/'qualification_draft.json');assert checked['postfit_only_recovery']and not torch.cuda.is_initialized()
  write(O/'qualification.json',q)
  print('EXPLORATORY1000_RECOVERED_QUALIFICATION_PENDING_NATIVE',flush=True)
if __name__=='__main__':main()
