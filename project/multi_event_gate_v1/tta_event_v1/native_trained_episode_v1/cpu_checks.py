"""CPU scheduling fixture: all five modes, unknown semantics, real capture API."""
from pathlib import Path
import hashlib,json
import numpy as np,torch
from .episode import execute,CONDITIONS
from native_policy_execution_v1 import capture as capture_module
from native_policy_execution_v1.common import T,read,sha,write
from native_trained_response_bridge_v1.bridge import load_model
from native_delayed_query_v1.predictor import FrozenPredictor
from cx_tta.multi_event import EventConfig
from actuation_telemetry import ACTIVE as TRACE
from cx_tta_clock.clock import ACTIVE as CLOCK
class Task:
 def __init__(self):self.take_action_cnt=0;self.step_lim=96;self.eval_success=False;self.command=np.zeros(14,np.float32)
 def take_action(self,command):
  if self.take_action_cnt>=self.step_lim:return
  self.take_action_cnt+=1;self.command=command.copy()
  for _ in range(2):
   row=dict(arm_position_target=np.r_[command[:6],command[7:13]].astype(float),arm_velocity_target=np.zeros(12),gripper_logical_command=command[[6,13]].astype(float))
   TRACE.get().append(row,float(np.float32(.004)));CLOCK.get().after_step(command,float(np.float32(.004)))
 def observe(self):
  pixel=int(hashlib.sha256(self.command.tobytes()).digest()[0]);image=np.full((240,320,3),pixel,np.uint8)
  return dict(rgb=(image,image.copy(),image.copy()),command_state=self.command.copy(),physical_sha256=hashlib.sha256(self.command.tobytes()+str(self.take_action_cnt).encode()).hexdigest(),rng_sha256='0'*64)
class Actor:
 def __init__(self,case):self.instruction=case['instruction'];self.refs=case['queries'];self.calls=0;self.executed=0
 def query(self,current,record,folder):
  q=self.calls;assert q==record['query'];self.calls+=1;p=self.refs[q]['policy']
  with np.load(p['policy_file'],allow_pickle=False)as z:values={k:z[k].copy()for k in z.files}
  file=folder/('CPU_saved_policy%03d.npz'%q)
  with file.open('xb')as f:np.savez_compressed(f,**values)
  policy=dict(policy_file=str(file),policy_sha256=sha(file),query=q,actor_observation_tick=record['global_tick'],CPU_cached_query_fixture=True,base_infer_calls=0,actual_internal_noise_calls=0)
  return dict(base=values['base'],raw_delta=values['raw_delta'],legal={k[len('legal_'):]:v for k,v in values.items()if k.startswith('legal_')}),policy,values
 def finish_chunk(self,actions,receipt,trace_file,folder):
  with np.load(trace_file,allow_pickle=False)as z:assert np.array_equal(actions,z['source_commands'])and len(z['dt_s'])==receipt['committed_ticks']
  self.executed+=1;return dict(CPU_receipt_fixture=True,commands_returned=sum(r['status']=='returned'for r in receipt['receipts']),end_tick=receipt['global_start_tick']+receipt['committed_ticks'])
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('high');before=torch.random.get_rng_state().clone();qualification=next((T/'outputs/native_direction_training_v2/runs').glob('*/qualification.json'));model,_=load_model(qualification);predictor=FrozenPredictor()
 case=next(c for c in read(T/'outputs/native_expert_full_feedback_v1/index.json')['cases']if c['old_pilot_reused']);config=EventConfig(**read(T/'outputs/native_expert_full_cache_v8/protocol.json')['event_config'])
 out=T/'outputs/native_trained_episode_v1/CPU_scheduling_fixture';assert not out.exists();results={};reference=None
 original=capture_module.discard_observation;capture_module.discard_observation=lambda task:None
 def forbidden(*args):raise AssertionError('Unknown semantics must not create a source')
 try:
  for condition in CONDITIONS:
   task=Task();actor=Actor(case);obs=task.observe();initial=dict(obs,instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest(),available=False)
   result=execute(task,actor,task.observe,condition=condition,event_config=config,response_model=model,qualification=qualification,predictor=predictor,source_factory=forbidden,seed_available=False,expected_initial=initial,instruction=case['instruction'],folder=out/condition,cpu_fixture=True)
   assert result['CPU_fixture_only']and result['native_episode_complete']and not result['success']and result['native_action_count']==96 and len(result['queries'])==actor.calls==actor.executed==3
   assert result['completed_sources']==0 and result['remaining_live_sources']==0 and result['feedback']['fast_reset_on_close']
   action_values=[]
   for q in result['queries']:
    with np.load(q['actions_file'],allow_pickle=False)as z:action_values.append(z['actions'].tobytes())
   if reference is None:reference=action_values
   else:assert action_values==reference
   results[condition]=dict(queries=actor.calls,native_command_calls=task.take_action_cnt,physical_ticks=result['physical_ticks'],unknown_feedback_fallback_exact=True)
  task=Task();actor=Actor(case);bad=task.observe();bad['command_state']=bad['command_state']+1;bad.update(instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest(),available=False)
  try:execute(task,actor,task.observe,condition='M_event_full',event_config=config,response_model=model,qualification=qualification,predictor=predictor,source_factory=forbidden,seed_available=False,expected_initial=bad,instruction=case['instruction'],folder=out/'initial_mismatch',cpu_fixture=True)
  except ValueError:pass
  else:raise AssertionError('Wrong initial seed-cache binding admitted')
  assert actor.calls==task.take_action_cnt==0
 finally:capture_module.discard_observation=original
 assert torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 write(out/'report.json',dict(passed=True,conditions=results,all_five_unknown_semantics_fallbacks_byte_exact=True,initial_binding_rejected_before_actor_draw=True,CPU_scheduling_fixture_only=True,cached_old_policy_arrays_used=True,new_VLA_forward_calls=0,real_native_simulator_rollouts=0,actual_online_actor_updates=0,TTA_utility_proven=False,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')}))
 print('WHOLE_EPISODE_FIVE_MODE_SCHEDULING_FIXTURE_PASS',5,96,flush=True)
if __name__=='__main__':main()
