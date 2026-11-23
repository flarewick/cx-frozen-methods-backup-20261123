"""Actual TRAIN target replay through passive hook; synthetic images are explicit."""
from pathlib import Path
import ast,types
import numpy as np,torch
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped
from native_feedback_application_v1.checks import run_application
from .readouts import CurrentReadouts,KIND
from .admission import choose

def main():
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 application=run_application() # Exact original max3/cooldown/zero/shadow/trust behavior.
 from native_live_response_source_v1.pool import SourcePool
 pool=SourcePool(lambda *args:None);initial=np.zeros(14,np.float32);commands=np.zeros((32,14),np.float32);commands[:,0]=np.linspace(0,1,32)
 kind,reason=choose(pool,0,None,False,seed_available=True,initial=initial,commands=commands);assert kind==KIND and not pool.queue.events
 pool.live[0]=object();selected=dict(kinds=['close_left'])
 assert choose(pool,1,selected,True,seed_available=True,initial=initial,commands=commands)[1]=='prior_original_window_still_live'
 pool.live.clear();pool.queue.events={i:{}for i in range(3)}
 assert choose(pool,4,selected,True,seed_available=True,initial=initial,commands=commands)[1]=='original_max3_feedback_source_budget'
 assert choose(pool,0,None,False,seed_available=False,initial=initial,commands=commands)[0]is None
 pool.queue.events.clear();assert choose(pool,0,None,False,seed_available=True,initial=initial,commands=np.zeros_like(commands))[0]is None
 # Actual original native targets and command indices, only the image reader is a fixture.
 evidence=read(T/'outputs/native_arm_phase_feedback_v1/CPU_actual_TRAIN20_phase_checks.json');first=evidence['records'][0];cid=first['case_id']
 roots=[p for p in(T/'outputs/native_onpolicy_collection_v1/fit_000/batches/000/runs').glob('*')if(p/'cases'/cid).exists()];assert len(roots)==1
 folder=roots[0]/'cases'/cid/'M_event_full';result=read(folder/'result.json');q=result['queries'][0]
 tracefile=mapped(q['execution']['actual_trace_file']);meta=read(folder/'execution000.json')
 with np.load(tracefile,allow_pickle=False)as z:trace={k:z[k].copy()for k in('arm_position_target','gripper_logical_command','source_commands','dt_s')}
 with np.load(mapped(q['observation']['file']),allow_pickle=False)as z:state=z['command_state'].copy()
 recorder=types.SimpleNamespace(rows=[],dt=[],failed=False);arrivals=[];reads=[]
 def image(tick,elapsed):
  assert tick==len(recorder.rows);reads.append(tick)
  return dict(rgb=(np.zeros((2,2,3),np.uint8),)*3,images={'0':'synthetic_CPU_fixture'},command_state=state.copy(),physical_sha256='0'*64,rng_sha256='1'*64)
 def arrived(tick,elapsed,obs,record):
  assert tick<=len(record.rows)and obs['physical_sha256']=='0'*64;arrivals.append(tick)
 hook=CurrentReadouts(recorder,image,kind=KIND,initial=dict(command_state=state,images={'0':'synthetic_CPU_fixture'}),base=trace['source_commands'],on_arrival=arrived)
 for command in meta['receipts']:
  hook.command_begin(command['offset'])
  for tick in range(command['start_tick'],command['end_tick']):
   recorder.rows.append({k:trace[k][tick]for k in('arm_position_target','gripper_logical_command')});recorder.dt.append(float(trace['dt_s'][tick]));hook.after(tick)
 tick=len(recorder.rows);terminal=hook.terminal_read(tick,complete_native_chunk=True)
 hook.completion.finish_origin('origin',arrived_tick=tick,commands_returned=32,native_censored=False);hook.completion.observe_rgb('origin',observation_tick=tick,decision_tick=tick,frame_id=hook.frame(terminal))
 row=hook.completion.closed[0];assert row['requested95_tick']==first['requested95_tick']and row['status']=='feedback_ready'
 assert arrivals==sorted(set(arrivals))and arrivals[-1]==tick and row['requested95_tick']in arrivals
 # The native capture loop differs only by passing the actually issued index.
 old=(T/'native_feedback_application_v1/capture.py').read_text();new=(Path(__file__).parent/'capture.py').read_text().replace('   if hook is not None:hook.command_begin(i)\n','')
 assert ast.dump(ast.parse(old))==ast.dump(ast.parse(new))
 oldepisode=(T/'native_feedback_application_v1/episode.py').read_text();newepisode=(Path(__file__).parent/'episode.py').read_text()
 for line in("   proposal,policy,values=actor.query(current,observed,folder)","   actions,delta,event=composer.step(proposal['base'],proposal['raw_delta'],proposal['legal'],adapted_ARM=direction)","   actions,delta,event,application_receipt=application.step(composer,proposal['base'],proposal['raw_delta'],direction,actions,delta,event,update,condition=condition)"):
  assert oldepisode.count(line)==newepisode.count(line)==1
 write(T/'outputs/native_arm_phase_feedback_v1/CPU_integration_checks.json',dict(passed=True,original_fast_application=application,source_overlap_and_max3_and_missing_semantics_rejected=True,actual_qualified_TRAIN_motor_targets_through_new_readout_hook=True,synthetic_image_fixture_only=True,actual_phase_tick=row['requested95_tick'],actual_native_ticks=tick,exactly_once_arrival_ticks=True,original_native_capture_AST_exact_except_actual_index=True,once_actor_query_original_gate_and_application_unchanged=True,CUDA_initialized=False,no_live_RGB_or_teacher_or_TTA_utility_qualification=True,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()},input_sha256={str(tracefile):sha(tracefile)}))
 print('ARM_PHASE_ACTUAL_TRAIN_HOOK_AND_ORIGINAL_FAST_CONTRACTS_CPU_PASS',flush=True)
if __name__=='__main__':main()
