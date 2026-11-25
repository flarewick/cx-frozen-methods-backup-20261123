"""One GPU owner services actual VLA/SAM2; bounded resident CPU native episode workers."""
import argparse,importlib,os,queue,threading,time,traceback,weakref
import numpy as np
from native_policy_execution_v1.common import T,read,write,sha
from .episode import execute as episode,CONDITIONS
from native_best100_projection_guard_v1.source import SourceStream
from .observe_checked_v23 import observe
from .matched_actor import AuditedActor as OriginalAuditedActor
from .official_open_laptop_arm_v143 import restore as restore_official_arm

def execute(spec,root,manifest,prepared,barrier,*,exit_success):
 import torch
 total_cases=len(spec['registered_cases']);lanes=spec['simulation_lanes'];assert 1<=lanes<=8 and total_cases>=lanes
 from .lanes import Coordinator,Lane,gripper_only
 from cx_tta_feedback_confirm5.throughput import replay_prefix
 from cx_tta_feedback_confirm5.model_math import sam2_profile
 from .cache import CacheMaintenance
 from cx_tta.collect_smoke import make_config
 from cx_tta.evaluate_b2 import seed_all
 from cx_tta.multi_event import EventConfig
 from cx_tta import recovery_contract as recovery
 from envs.utils.create_actor import UnStableError
 from native_initial_frame_v3.observe import physical
 from native_completed_memory_v1.reclaim import reclaim
 from native_case_release_v1.release import release as release_case
 from adapter_cache_cpu import move_to_gpu
 from native_physics import load_native
 from collection_native_step import load as load_step
 assert spec['scope']=='shared6_v118_alpha0p5_GT_v1'and spec['cases_per_batch']==total_cases
 assert spec['conditions']in [['M_event_full','TTA'],['TTA']]and spec['simulation_lanes']==lanes
 assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==manifest['gpu_uuid']
 torch.set_num_threads(2);base=move_to_gpu(prepared['base']);base._model.eval().requires_grad_(False)
 mask_model=prepared['mask_model'].to('cuda').eval().requires_grad_(False)
 from .compiled_sampler import install as install_compiled_sampler
 compiled_sampler=install_compiled_sampler(base,root)
 from .owner_precision import install
 check_precision=install(base,torch)
 assert check_precision()
 write(barrier/'cuda_ready.json',dict(inner_pid=os.getpid(),gpu_uuid=manifest['gpu_uuid']))
 began=time.monotonic()
 while not(barrier/'owned_pid_permit.json').exists():
  if time.monotonic()-began>45:raise TimeoutError('No actual single GPU owner permit')
  time.sleep(.03)
 assert read(barrier/'owned_pid_permit.json')['gpu_uuid']==manifest['gpu_uuid']
 c=Coordinator(cuda=True,native=load_native(),native_step=load_step());c.rng.install()
 models=(base._model,mask_model,*[head for head,_ in prepared['heads']],prepared['response_model'])
 def versions():return [[(name,p.data_ptr(),p._version,str(p.dtype),tuple(p.shape))for name,p in model.named_parameters()]for model in models]
 frozen_versions=versions()
 import cv2
 cv2.setNumThreads(2)
 import sam2.sam2_video_predictor as official
 official.tqdm=lambda iterable,**kwargs:iterable
 import envs.robot.robot as robot
 original_planner=robot.CuroboPlanner
 from .screen import Schedule as OriginalSchedule
 Schedule=OriginalSchedule
 schedule=Schedule(c,prepared,spec,root,original_planner)
 from .prefetch import PrefetchClient
 from cx_tta_feedback_confirm5.throughput import discard_observation
 from cx_tta_feedback_confirm5.render_upload import install as install_render
 prepared['seed_store']=PrefetchClient(root/'semantic')
 for row in(spec.get('cached_preparation')or{}).get('requests',[]):prepared['seed_store'].register_verified(row)
 def prefetch_initial(task,instruction,name,seed,index):
  install_render(task);task.set_instruction(instruction=instruction);discard_observation(task);initial=observe(task)
  assert initial['native_action_count']==0
  key=prepared['seed_store'].submit(initial,instruction)
  write(root/'initial_prefetch'/(name+':'+str(seed)+'.json'),dict(content_key=key,only_RGB_own_command_language_submitted=True,passive_physics_action_count_RNG_unchanged=True,no_CPU_answer_wait=True))
 schedule.on_initial=prefetch_initial
 # Screen each lane only after its real-query startup gate opens.
 # Separate per-task order locks preserve exactly the original seed stream.
 from .ordered_screen import OrderedScreen
 # Cached prefixes avoid late GPU planner allocation; finish only missing expert work first.
 prescheduled=[schedule.next(spec['tasks'][n%2])for n in range(lanes)]if spec.get('cached_preparation')else None
 scheduled=[None]*total_cases
 ordered_screen=OrderedScreen(schedule.next,[x['task']for x in spec['registered_cases']],c.cancel)
 case_to_lane={}
 def select(number,index):
  assert prescheduled is None,'Cached finite-lane preparation is not a resident protocol'
  item=ordered_screen.next(index)
  if item is None:return None
  if False:
   previous=__import__('pathlib').Path(spec['previous_batch002_accepted_root'])
   assert item==read(previous/(item['case_id']+'.json')),'Resource-only scheduling changed fixed scene/instruction identity'
  scheduled[index]=item;case_to_lane[item['case_id']]=number
  write(root/'screening_selected'/('%d.json'%index),dict(case=item,lane=number,prior_batch002_identity_exact=False,guard_clock_not_reset=True))
  assert c.call(schedule.lanes[item['task']],check_precision)
  return item
 saved={name:getattr(recovery,name)for name in('rng_capture','rng_restore')}
 for name,fn in saved.items():setattr(recovery,name,c.routed(fn))
 cache=CacheMaintenance(torch,c,base,prepared['session'],auxiliary_models=(mask_model,))
 # Shared model is read-only; FeedbackSession deep-copies it per episode/branch.
 prepared['GPU_keepalive']=dict(coordinator=c,base=base,mask_model=mask_model,cache=cache)
 import actuation_telemetry as telemetry
 from native_telemetry_bridge_v1.bridge import ReadoutBridge
 from native_telemetry_batch_v1.capture import capture as batch_capture
 original_capture=telemetry.capture
 bridge=ReadoutBridge(original_capture,lambda robot:batch_capture(robot,prepared['telemetry_native']))
 def guarded_readout(robot):
  with c.scene_gate.read():return bridge.capture(robot)
 telemetry.capture=guarded_readout
 from .startup_two_then_fill_v23 import StagedLanes,actor_class
 startup=StagedLanes(c.cancel,lambda row:write(root/('startup_release_'+str(len(row['released_lanes']))+'.json'),row))
 AuditedActor=actor_class(OriginalAuditedActor,startup,case_to_lane)
 errors=queue.Queue();finished=queue.Queue();threads=[];results={};started=time.monotonic()
 def run_case(number,item):
  lane=Lane(str(number));head,robots=prepared['heads'][number]
  branches=['M_event_full','zero_update','shadow_update','TTA']if item['episode_index']==0 or item['case_id']in spec.get('extra_control_case_ids',[])else spec['conditions']
  with c.rng.bind(lane):
   cls=getattr(importlib.import_module('envs.'+item['task']),item['task'])
   cfg=make_config(prepared['runtime'],argparse.Namespace(task=item['task'],task_config='demo_randomized'),prepared['session'])
   cfg.update(eval_mode=spec['phase']=='EVAL',save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
   folder=root/'cases'/item['case_id'];folder.mkdir(parents=True)
   write(folder/'case.json',dict(case_id=item['case_id'],role='frozen_official_EVAL_no_fit',conditions=list(branches),initialization_eval_mode=True,task_seed_excluded_from_online_forward=True))
   rows=[];initial_identity=None
   for branch in branches:
    c.check_cancel();task=c.call(lane,lambda:c.prepare_task(cls(),lane));task._rwc_registered_instruction=item['instruction'];raw=lambda:c.call(lane,observe,task)
    routed_seed=c.routed(seed_all);branchfolder=folder/branch;bridge.begin_case(item['case_id']+'/'+branch,number)
    c.call(lane,cache.begin_case,lane,item['case_id']+'/'+branch)
    cleanup=[];actor=None;factory=None;native_diagnostic=None
    try:
     # A successful step-zero replay is eligible as the live initial state
     # after the exact existing physical/RGB/command/RNG checks.
     from .reconstruct_q0_v14 import reconstruct
     def replay():
      try:
       return replay_prefix(task,item,[],cfg,raw,routed_seed)
      except UnStableError as exc:
       marker=folder/(branch+'_scheduled_warmup_failure.json')
       if marker.exists():raise
       write(marker,dict(type=type(exc).__name__,reason=str(exc),no_policy_actions=True,
                         phase='first_exact_step_zero_replay',same_seed_retry=True))
       task.close_env(clear_cache=False)
       return replay_prefix(task,item,[],cfg,raw,routed_seed)
     initial,identity=reconstruct(task,item,folder,branch,initial_identity,replay,
                                  lambda:c.call(lane,recovery.rng_capture))
     arm=c.call(lane,restore_official_arm,task,item['task'])
     if arm is not None:
      write(folder/(branch+'_official_open_laptop_arm.json'),dict(arm=arm,derived_from_initial_pose_by_original_play_once_rule=True))
     if initial_identity is None:
      initial_identity=identity
     expected,masks,seed_report=prepared['seed_store'].at_initial(initial,item['instruction'],query=0,tick=0)
     def diagnostic_owner():
      return dict(physical_sha256=physical(task),rng_sha256=recovery.rng_fingerprint(recovery.rng_capture()))
     native_diagnostic=lambda:c.call(lane,diagnostic_owner)
     from .warmup_reuse_checked_v23 import warmup
     c.call(lane,warmup,c,lane,base,initial,item['instruction'],root,head,robots,prepared['projection'],item['case_id'])
     actor=AuditedActor(c,base,head,robots,prepared['projection'],cache,name=item['case_id']+'_'+branch,instruction=item['instruction'],native_diagnostic=native_diagnostic)
     def tracker(fn,*args,**kwargs):
      def operation():
       with sam2_profile():return fn(*args,**kwargs)
      return c.call(lane,operation)
     def factory(history,queue,ticket):
      return SourceStream(mask_model,masks,history,item['instruction'],robots=robots,cfg=prepared['feedback_hardware'][0],camera=prepared['feedback_hardware'][1],call=tracker,queue=queue,ticket=ticket)
     def retire(box,refs):
      def operation():
       box.clear();proof=reclaim(refs);cleanup.append(proof)
      c.call(lane,operation)
     result=episode(task,actor,raw,condition=branch,event_config=EventConfig(**spec['event_config']),response_model=prepared['response_model'],qualification=spec['trained_qualification'],predictor=prepared['predictor'],source_factory=factory,seed_available=seed_report['available'],expected_initial=expected,instruction=item['instruction'],folder=branchfolder,retire=retire,check_cancel=c.check_cancel)
     write(branchfolder/'semantic_initial_receipt.json',dict(**seed_report,initial_diagnostic_identity=identity))
     write(branchfolder/'completed_source_memory.json',dict(proofs=cleanup))
     readouts=bridge.finish_case();assert readouts['native_calls']==result['physical_ticks'];write(branchfolder/'readout_bridge.json',readouts)
     c.call(lane,cache.finish_case,lane,branchfolder)
     rows.append(dict(condition=branch,result_file=str(branchfolder/'result.json'),result_sha256=sha(branchfolder/'result.json'),success=result['success'],native_episode_complete=result['native_episode_complete'],queries=len(result['queries'])))
    finally:task.close_env(clear_cache=False)
    # No callback or task reference survives into the next branch's setup.
    task_ref=weakref.ref(task);scene_ref=weakref.ref(task.scene);task=None;raw=None;actor=None;factory=None;native_diagnostic=None
    released=c.call(lane,lambda:release_case(c,task_ref,scene_ref,torch=torch,model=base._model))
    write(branchfolder/'completed_task_release.json',released)
    print('TRAINED_DEVELOPER_NATIVE_BRANCH',item['case_id'],branch,result['success'],result['native_episode_complete'],flush=True)
   record=dict(case_id=item['case_id'],all_requested_conditions_retained=True,initialized=True,branches=rows,initial_identity=initial_identity,independent_result=True)
   write(folder/'bundle.json',record);finished.put((item['case_id'],record))

 from .resident_queue import ResidentQueue
 resident=ResidentQueue(total_cases,lanes,c.cancel,root)
 def worker(number):
  try:
   if not startup.wait(number):return
   index=resident.first(number)
   while index is not None:
    item=select(number,index)
    if item is None:raise RuntimeError('Fixed registered resident case disappeared')
    resident.begin(number,index,item['case_id'])
    run_case(number,item)
    resident.finish(number,index,item['case_id'])
    startup.release('actual_case_terminal',lane=number)
    index=resident.next()
  except BaseException:
   text=traceback.format_exc();write(prepared['session']/('lane%d_failure.json'%number),dict(traceback=text));errors.put(text);c.cancel.set()

 try:
  for number in range(lanes):
   thread=threading.Thread(target=worker,args=(number,),daemon=True);threads.append(thread);thread.start()
  while len(results)<total_cases:
   if not errors.empty():raise RuntimeError(errors.get())
   c.service.process_one(timeout=.01)
   while True:
    try:number,record=finished.get_nowait();assert number not in results;results[number]=record
    except queue.Empty:break
  for thread in threads:thread.join(timeout=3)
  assert not any(t.is_alive()for t in threads)and c.service.requests.empty()
  assert frozen_versions==versions(),'Frozen model parameter identity or mutation counter changed'
  assert all(scheduled)and len({item['case_id']for item in scheduled})==total_cases
  resident_receipt=resident.receipt();resident_receipt.update(inner_gpu_PID=os.getpid(),frozen_parameter_pointers_and_mutation_counters_unchanged=True);write(root/'resident_model_reuse.json',resident_receipt)
  schedule.finish()
  assert base._sample_actions is compiled_sampler
  write(root/'denoise_graph_runtime.json',dict(original_official_compiled_sampler_retained=True,original_outer_sampler_and_once_noise_unchanged=True,original_model_parameters_and_base_outputs_checked=True,first_resumed_batch_whole_zero_shadow_controls=spec['batch_index']==2))
  write(root/'GPU_episodes_complete.json',dict(complete=True,protocol_identity=spec['identity'],cases=[results[k]for k in sorted(results)],seconds=time.monotonic()-started,scope=spec['scope'],frozen_parameter_pointers_and_mutation_counters_unchanged=True,phase=spec['phase'],registered_scenes=total_cases,concurrent_scene_slots=lanes,single_model_owner_reused=True,TTA_utility_proven=False))
 finally:
  c.cancel.set()
  for thread in threads:thread.join(timeout=2)
  telemetry.capture=original_capture;cache.uninstall()
  for name,fn in saved.items():setattr(recovery,name,fn)
  robot.CuroboPlanner=original_planner;c.rng.uninstall()

 # Avoid unwinding large CUDA/physics local ownership after a complete batch.
 # Any exception in the scientific work or finally cleanup bypasses this call.
 exit_success()
 raise RuntimeError('Success exit callback returned unexpectedly')
