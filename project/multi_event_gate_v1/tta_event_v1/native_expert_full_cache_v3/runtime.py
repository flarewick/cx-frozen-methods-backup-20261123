"""One GPU owner and six persistent native simulation lanes; only original96 cases."""
import argparse,importlib,os,queue,threading,time,traceback
import numpy as np
from .paths import read,write,sha
from .capture import capture
from .policy import ActualQuery
from native_expert_smoke_v2.observe import observe
from native_expert_smoke_v2.setup import replay_prefix

def execute(spec,root,manifest,prepared,barrier):
 import torch
 from cx_tta_feedback_confirm5.lanes import Coordinator,Lane,gripper_only
 from cx_tta_feedback_confirm5.throughput import TaskPool
 from cx_tta.collect_smoke import make_config
 from cx_tta.evaluate_b2 import seed_all
 from cx_tta.multi_event import EventDetector,EventConfig
 from cx_tta.empirical_onset_gate import candidate_packet
 from cx_tta import recovery_contract as recovery
 from native_initial_frame_v3.observe import physical
 from adapter_cache_cpu import move_to_gpu
 from native_physics import load_native
 from collection_native_step import load as load_step
 from native_expert_planner_v2.runtime import diagnostic
 from .cache_maintenance import CacheMaintenance
 assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==manifest['gpu_uuid']
 torch.set_num_threads(2);base=move_to_gpu(prepared['policy']);base._model.eval().requires_grad_(False)
 sampler=base._model.sample_actions
 while hasattr(sampler,'_torchdynamo_orig_callable'):sampler=sampler._torchdynamo_orig_callable
 base._sample_actions=sampler;assert torch.get_float32_matmul_precision()=='high'
 write(barrier/'cuda_ready.json',dict(inner_pid=os.getpid(),gpu_uuid=manifest['gpu_uuid']))
 began=time.monotonic()
 while not(barrier/'owned_pid_permit.json').exists():
  if time.monotonic()-began>45:raise TimeoutError('No verified single GPU owner permit')
  time.sleep(.03)
 c=Coordinator(cuda=True,native=load_native(),native_step=load_step());c.rng.install()
 saved={name:getattr(recovery,name)for name in('rng_capture','rng_restore')}
 for name,fn in saved.items():setattr(recovery,name,c.routed(fn))
 import envs.robot.robot as robot
 from envs.utils.create_actor import UnStableError
 original_planner=robot.CuroboPlanner;robot.CuroboPlanner=gripper_only(original_planner)
 session=prepared['session'];cache=CacheMaintenance(torch,c,base,session)
 pools=[TaskPool()for _ in range(6)];threads=[];finished=queue.Queue();errors=queue.Queue();results={};start=time.monotonic()
 from .recovery import recovered_rows
 results.update(recovered_rows(spec,root));recovered_indices=set(results)
 prepared['GPU_keepalive']=dict(coordinator=c,model=base,cache=cache,pools=pools)
 tasks=('stack_blocks_two','stack_bowls_two');jobs={task:queue.Queue()for task in tasks}
 for index,item in enumerate(spec['scenes']):
  if index not in recovered_indices:jobs[item['task']].put(index)
 for task in tasks:
  for _ in range(3):jobs[task].put(None)
 classes={name:getattr(importlib.import_module('envs.'+name),name)for name in tasks}
 def worker(number):
  lane=Lane(str(number));taskname=tasks[number%2];warmed=False;task=None;head,robots=prepared['heads'][number]
  try:
   with c.rng.bind(lane):
    while not c.cancel.is_set():
     index=jobs[taskname].get()
     if index is None:return
     item=spec['scenes'][index];assert item['task']==taskname
     folder=root/('case%03d'%index);folder.mkdir();began=time.monotonic();chunks=[];observations=[];events=[];query_rows=[]
     cfg=make_config(prepared['runtime'],argparse.Namespace(task=taskname,task_config='demo_randomized'),session)
     cfg.update(eval_mode=False,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
     task=pools[number].get(taskname,lambda:c.call(lane,lambda:c.prepare_task(classes[taskname](),lane)))
     raw=lambda:c.call(lane,observe,task);seed=c.routed(seed_all)
     def save_obs(value,q,tick):
      file=folder/('query%03d.npz'%q)
      with file.open('xb')as f:np.savez_compressed(f,**{'cam'+str(i):v for i,v in enumerate(value['rgb'])},command_state=value['command_state'])
      row={k:v for k,v in value.items()if k not in('rgb','command_state')};row.update(file=str(file),sha256=sha(file),query=q,global_tick=tick)
      observations.append(row);return row
     try:
      if not warmed:
       try:replay_prefix(task,item,[],cfg,raw,seed)
       finally:task.close_env(clear_cache=False)
       warmed=True
      current=replay_prefix(task,item,[],cfg,raw,seed);save_obs(current,0,0)
      initial=c.call(lane,diagnostic,task,item['object_names'])
      with np.load(item['first_saved_identity_file'],allow_pickle=False)as z:
       comparisons={k:bool(np.array_equal(v,z['first_saved_'+k]))for k,v in initial.items()}
       comparisons['command']=bool(np.array_equal(current['command_state'],z['first_saved_command']))
      identity_file=folder/'initial_identity.npz'
      with identity_file.open('xb')as f:np.savez_compressed(f,**initial,command=current['command_state'])
      write(folder/'initial_identity.json',dict(offline_diagnostic_only=True,source_file=item['first_saved_identity_file'],source_sha256=item['first_saved_identity_sha256'],actual_file=str(identity_file),actual_sha256=sha(identity_file),equal=comparisons))
      if not all(comparisons.values()):
       row=dict(available=False,reason='original_initial_identity_mismatch_retained',identity_comparisons=comparisons,observations=observations)
      else:
       query=ActualQuery(c,base,head,robots,prepared['projection'],cache,name='original'+str(item['index']),instruction=item['instruction'])
       detector=EventDetector(EventConfig(**spec['event_config']));previous_event=False;source_ordinal=0
       with np.load(item['file'],allow_pickle=False)as z:commands=z['absolute_source_commands'].copy()
       tick=0
       for q,source_start in enumerate(range(0,len(commands),32)):
        c.check_cancel()
        if task.eval_success or task.take_action_cnt>=task.step_lim:break
        before_physical=observations[-1]['physical_sha256'];before_rng=observations[-1]['rng_sha256']
        proposal,policy_row=query.query(current,observations[-1],folder)
        assert c.call(lane,physical,task)==before_physical
        assert c.call(lane,lambda:recovery.rng_fingerprint(recovery.rng_capture()))==before_rng,'Model call changed native environment RNG'
        policy_row.update(environment_physics_RNG_unchanged=True);query_rows.append(policy_row)
        event,evidence=detector.step(q,candidate_packet(proposal['legal'],proposal['raw_delta']))
        kind=None;ordinal=None
        if event is not None and len(event['kinds'])==1:
         kind=event['kinds'][0];ordinal=source_ordinal;source_ordinal+=1
        events.append(dict(query=q,event=event,evidence=evidence,source_ordinal=ordinal,
         within_original_max3=ordinal is not None and ordinal<3,unsupported_multi_kind=event is not None and len(event['kinds'])!=1))
        selected=commands[source_start:source_start+32]
        arrays,receipt,dense=capture(task,selected,max_ticks=spec['max_ticks_per_chunk'],reader=raw,initial=current,
          kind=kind,collect_dense=bool(event is not None or previous_event))
        previous_event=event is not None;file=folder/('chunk%03d.npz'%q)
        with file.open('xb')as f:np.savez_compressed(f,**arrays)
        receipt.update(query=q,source_start=source_start,source_count=len(selected),global_start_tick=tick,file=str(file),sha256=sha(file))
        if dense is not None:
         df=folder/('arrived%03d.npz'%q)
         with df.open('xb')as f:np.savez(f,**dense)
         receipt.update(dense_file=str(df),dense_sha256=sha(df))
        write(folder/('chunk%03d.json'%q),receipt);chunks.append(receipt);tick+=receipt['committed_ticks']
        current=raw();save_obs(current,q+1,tick)
        print('ORIGINAL96_NATIVE_CHUNK',index,item['case_id'],q,receipt['committed_ticks'],receipt['reason'],flush=True)
        if receipt['reason']!='source_chunk_returned':break
       row=dict(available=True,chunks=chunks,observations=observations,queries=query_rows,events=events,
         matched_initial_object_robot_command_exact=True,initialization_eval_mode=False,source_commands=len(commands),
         actual_native_action_count=int(task.take_action_cnt),native_step_limit=int(task.step_lim),native_success_diagnostic_only=bool(task.eval_success),
         committed_ticks=tick,source_exhausted=bool(chunks and chunks[-1]['source_start']+len(chunks[-1]['receipts'])==len(commands)),
         sampled_base_NOT_executed=True,actual_policy_draws=len(query_rows),no_semantic_tracking_yet=True)
     except UnStableError as exc:
      row=dict(available=False,reason='native_unstable_initialization_retained',error=str(exc),observations=observations,chunks=chunks,queries=query_rows)
     finally:task.close_env(clear_cache=False)
     row.update(index=index,original_index=item['index'],case_id=item['case_id'],split=item['split'],lane=number,seconds=time.monotonic()-began)
     write(folder/'capture.json',row);finished.put((index,row));jobs[taskname].task_done()
     print('ORIGINAL96_NATIVE_CASE',index,item['case_id'],row['available'],round(row['seconds'],3),flush=True)
  except BaseException:
   failure=traceback.format_exc();write(session/('lane%d_failure.json'%number),dict(traceback=failure));errors.put(failure);c.cancel.set()
 try:
  for i in range(6):
   thread=threading.Thread(target=worker,args=(i,),daemon=True);threads.append(thread);thread.start()
  while len(results)<96:
   if not errors.empty():raise RuntimeError(errors.get())
   c.service.process_one(timeout=.01)
   while True:
    try:i,row=finished.get_nowait();assert i not in results;results[i]=row
    except queue.Empty:break
  for thread in threads:thread.join(timeout=3)
  assert not any(thread.is_alive()for thread in threads)
  write(root/'GPU_native_complete.json',dict(complete=True,protocol_identity=spec['identity'],rows=[results[i]for i in range(96)],
    requested=96,new_GPU_cases=68,recovered_terminal_cases=28,recovered_native_available=27,recovered_initialization_failures=1,recovered_indices=sorted(recovered_indices),
    actual_new_policy_draws=sum(r.get('actual_policy_draws',0)for i,r in results.items()if i not in recovered_indices),
    recovered_policy_draws=sum(results[i].get('actual_policy_draws',0)for i in recovered_indices),
    source_attempt_normal_exit_NOT_claimed=True,seconds=time.monotonic()-start,actual_policy_draws=sum(r.get('actual_policy_draws',0)for r in results.values()),
    available=sum(r['available']for r in results.values()),four_completed_original_cases_NOT_repeated=True,
    full_original_expert_cache_ready=False,training_steps=0,online_updates=0,TTA_utility_proven=False))
 finally:
  c.cancel.set()
  for thread in threads:thread.join(timeout=2)
  cache.uninstall()
  for name,fn in saved.items():setattr(recovery,name,fn)
  robot.CuroboPlanner=original_planner;c.rng.uninstall()
