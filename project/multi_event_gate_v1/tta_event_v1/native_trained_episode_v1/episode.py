"""Original natural task budget with once-base, frozen event control and arrived TTA."""
from pathlib import Path
import hashlib,json,time
import numpy as np,torch
from native_phase_feedback_v1.feedback_queue import FeedbackQueue
from native_live_response_source_v1.pool import SourcePool
from native_live_response_source_v1.export import save_source
from native_response_action_v1.compose import EventComposer,ARM
from native_trained_response_bridge_v1.bridge import DirectionBridge
from native_policy_execution_v1.capture import capture
from native_policy_execution_v1.common import write,sha
from native_delayed_query_v1.boundary import native_continue
from .controls import wrong_commands
CONDITIONS=('M_event_full','zero_update','shadow_update','TTA','wrong_feedback')

def execute(task,actor,observe,*,condition,event_config,response_model,qualification,predictor,source_factory,seed_available,expected_initial,instruction,folder,retire=None,check_cancel=lambda:None,cpu_fixture=False):
 if condition not in CONDITIONS:raise ValueError('Fixed explicit episode condition required')
 folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False)
 if not isinstance(seed_available,bool)or int(task.step_lim)<=0 or int(task.take_action_cnt)!=0:raise ValueError('Actual semantic availability and original native budget required')
 composer=EventComposer(event_config);queue=FeedbackQueue();bridge=None
 if condition in ('shadow_update','TTA','wrong_feedback'):
  bridge=DirectionBridge(response_model,qualification,mode='shadow'if condition=='shadow_update'else'TTA',queue=queue)
 pool=SourcePool(source_factory,queue=queue,retire=retire);issuing={};source_kinds={};queries=[];physical_tick=0;elapsed_s=0.;started=time.monotonic();stop_reason=None
 try:
  if actor.instruction!=instruction or expected_initial['instruction_sha256']!=hashlib.sha256(instruction.encode()).hexdigest()or expected_initial['available']!=seed_available:raise ValueError('Semantic cache/actor instruction or availability mismatch')
  initial=observe()
  if len(initial['rgb'])!=3 or any(not np.array_equal(a,b)for a,b in zip(initial['rgb'],expected_initial['rgb'],strict=True))or not np.array_equal(initial['command_state'],expected_initial['command_state']):raise ValueError('Initial semantic-cache/current actor RGB or command mismatch')
  current=initial
  while native_continue(task):
   check_cancel();q=len(queries)
   if q>=(int(task.step_lim)+31)//32:raise ValueError('Native action counter failed to respect original step budget')
   obsfile=folder/('observation%03d.npz'%q)
   with obsfile.open('xb')as f:np.savez_compressed(f,**{'cam'+str(i):im for i,im in enumerate(current['rgb'])},command_state=current['command_state'])
   observed=dict(file=str(obsfile),sha256=sha(obsfile),query=q,global_tick=physical_tick)
   prefix=dict(query=q,tick=physical_tick,episode_elapsed_s=elapsed_s,rgb=current['rgb'][0],command_state=current['command_state'])
   pool.arrive_query(prefix)
   proposal,policy,values=actor.query(current,observed,folder)
   if not cpu_fixture:assert policy['base_infer_calls']==policy['actual_internal_noise_calls']==1 and policy['actor_observation_tick']==physical_tick
   proof=dict(query=q,tick=physical_tick,observation_sha256=observed['sha256'],policy_sha256=policy['policy_sha256'])
   proof_sha=hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest()
   actual=dict(query=q,tick=physical_tick,query_proof_sha256=proof_sha,**{k:values[k]for k in ('hidden','logits','raw_delta')})
   control=[];update=dict(reason='frozen_full_no_update',accepted=False)
   if bridge is not None:
    rows,control=wrong_commands(pool)if condition=='wrong_feedback'else(pool.rows(),[])
    direction,update=bridge.step(current=actual,rows_by_source=rows,issuing_queries=issuing,predictor=predictor)
   else:
    lease=queue.at_query(query=q,tick=physical_tick,query_proof_sha256=proof_sha)
    update.update(released_packet_rows=0 if lease is None else len(lease.rows))
    direction=None
    if condition=='zero_update':
     with torch.no_grad():direction=response_model.direction(torch.tensor(values['hidden'][None]),torch.tensor(values['logits'][:,ARM][None]),torch.tensor(values['raw_delta'][:,ARM][None]),response_model.zero_fast()).numpy()[0]
   actions,delta,event=composer.step(proposal['base'],proposal['raw_delta'],proposal['legal'],adapted_ARM=direction)
   pool.retire_after_query(q)
   selected=event['decision']['selected_event']
   single=selected is not None and len(selected['kinds'])==1
   if event['decision']['active']and single and seed_available:
    pool.admit();source_kinds[q]=selected['kinds'][0]
   plan=pool.capture_plan(q,physical_tick);kind=source_kinds[q]if plan['origin_source_query']is not None else None
   trace,receipt,dense=capture(task,actions,reader=observe,initial=current,kind=kind,collect_dense=plan['collect_dense'],on_arrival=plan['on_arrival'],check_cancel=check_cancel)
   tracefile=folder/('execution%03d.npz'%q)
   with tracefile.open('xb')as f:np.savez_compressed(f,**trace)
   if dense is not None:
    densefile=folder/('execution%03d_arrivals.npz'%q)
    with densefile.open('xb')as f:np.savez_compressed(f,**dense)
    receipt.update(dense_file=str(densefile),dense_sha256=sha(densefile))
   receipt.update(query=q,global_start_tick=physical_tick,actual_trace_file=str(tracefile),actual_trace_sha256=sha(tracefile))
   write(folder/('execution%03d.json'%q),receipt)
   actor_execution=actor.finish_chunk(actions,receipt,tracefile,folder)
   pool.close_origin(q,trace,receipt)
   commandfile=folder/('actions%03d.npz'%q)
   with commandfile.open('xb')as f:np.savez_compressed(f,base=proposal['base'],reference=proposal['raw_delta'],actions=actions,applied_delta=delta)
   queries.append(dict(query=q,tick=physical_tick,observation=observed,policy=policy,actions_file=str(commandfile),actions_sha256=sha(commandfile),event=event,response=update,wrong_feedback_control=control,execution=actor_execution,reason=receipt['reason']))
   issuing[q]=dict(query=q,tick=physical_tick,hidden=values['hidden'].copy());physical_tick+=receipt['committed_ticks'];elapsed_s+=float(np.sum(trace['dt_s']))
   if not receipt['complete_source_chunk']:
    stop_reason=receipt['reason'];break
   current=observe()
  if stop_reason is None:stop_reason='native_success_terminal'if bool(task.eval_success)else'native_step_limit'
  sources=pool.finish()
  for i,summary in enumerate(sources['completed_sources']):save_source(folder/('source%02d'%i),summary)
  feedback=bridge.finish()if bridge is not None else dict(queue=queue.finish(),updates=[],fast_reset_on_close=True)
  result=dict(complete=True,condition=condition,native_episode_complete=not native_continue(task),success=bool(task.eval_success),native_action_count=int(task.take_action_cnt),original_native_step_limit=int(task.step_lim),physical_ticks=physical_tick,physical_elapsed_s=elapsed_s,reason=stop_reason,queries=queries,feedback=feedback,completed_sources=len(sources['completed_sources']),remaining_live_sources=sources['remaining_live_sources'],semantic_seed_available=seed_available,qualification_file=str(qualification),qualification_sha256=sha(qualification),seconds=time.monotonic()-started,CPU_fixture_only=cpu_fixture,task_success_diagnostic_only=True,TTA_utility_proven=False)
  write(folder/'result.json',result);return result
 except BaseException as exc:
  write(folder/'failure.json',dict(type=type(exc).__name__,reason=str(exc),condition=condition,complete=False,queries_written=len(queries),CPU_fixture_only=cpu_fixture))
  try:pool.finish()
  except BaseException:pass
  try:
   if bridge is not None:bridge.finish()
   else:queue.finish()
  except BaseException:pass
  raise
