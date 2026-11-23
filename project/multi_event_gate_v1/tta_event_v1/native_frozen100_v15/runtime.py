"""One GPU owner services actual VLA/SAM2; four CPU native episode workers."""
import argparse,importlib,os,queue,threading,time,traceback,weakref
import numpy as np
from native_policy_execution_v1.common import T,read,write,sha
from .episode import execute as episode,CONDITIONS
from native_best100_projection_guard_v1.source import SourceStream
from native_expert_smoke_v2.observe import observe
from native_trained_runtime_v1.actor import AuditedActor as OriginalAuditedActor

def execute(spec,root,manifest,prepared,barrier,*,exit_success):
 import torch
 from cx_tta_feedback_confirm5.lanes import Coordinator,Lane,gripper_only
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
 lanes=spec['simulation_lanes'];assert lanes in(8,10)
 assert spec['scope']=='frozen_best_TTA_cross_task_official_seed0_paired100'and spec['cases_per_batch']==lanes
 assert spec['conditions']==['baseline','M_event_full','TTA']and spec['simulation_lanes']==lanes
 assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==manifest['gpu_uuid']
 torch.set_num_threads(2);base=move_to_gpu(prepared['base']);base._model.eval().requires_grad_(False)
 mask_model=prepared['mask_model'].to('cuda').eval().requires_grad_(False)
 sampler=base._model.sample_actions
 while hasattr(sampler,'_torchdynamo_orig_callable'):sampler=sampler._torchdynamo_orig_callable
 base._sample_actions=sampler
 from native_denoise_graph_v1.graph import DenoiseGraph
 denoise_graph=DenoiseGraph(base._model);base._model.denoise_step=denoise_graph
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
 from .python_scheduling import install as install_python_scheduling
 install_python_scheduling(root/'python_thread_scheduling.json')
 from native_encoder_graph_v2.runtime import PrefixGraphs
 prefix_graphs=PrefixGraphs(base._model)
 from .setup_fairness import install as install_setup_fairness
 install_setup_fairness(c,root/'setup_queue_admission.jsonl')
 models=(base._model,mask_model,*[head for head,_ in prepared['heads']],prepared['response_model'])
 def versions():return [[(name,p.data_ptr(),p._version,str(p.dtype),tuple(p.shape))for name,p in model.named_parameters()]for model in models]
 frozen_versions=versions()
 from .scene_cache import install as install_scene_cache
 install_scene_cache(c,torch,models,root/'scene_allocator_reclamation.jsonl')
 import cv2
 cv2.setNumThreads(2)
 import sam2.sam2_video_predictor as official
 official.tqdm=lambda iterable,**kwargs:iterable
 import envs.robot.robot as robot
 original_planner=robot.CuroboPlanner
 from .screen import Schedule as OriginalSchedule
 from .cached_screen import schedule_class
 Schedule=schedule_class(OriginalSchedule)
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
  write(root/'initial_prefetch'/(name+':'+str(seed)+'.json'),dict(content_key=key,only_RGB_own_command_language_submitted=True,passive_physics_action_count_RNG_unchanged=True,no_CPU_answer_wait=True,diagnostic_identity={k:initial[k]for k in('physical_sha256','rng_sha256')}))
 schedule.on_initial=prefetch_initial
 # Screen each lane only after its real-query startup gate opens.
 # Separate per-task order locks preserve exactly the original seed stream.
 from .ordered_screen import OrderedScreen
 # Cached prefixes avoid late GPU planner allocation; finish only missing expert work first.
 prescheduled=[schedule.next(spec['tasks'][n%2])for n in range(lanes)]
 from .completed_planners import release_completed
 release_completed(schedule,c,torch,models,root/'completed_expert_GPU_release.json',prescheduled,lanes)
 scheduled=[None]*lanes
 ordered_screen=OrderedScreen(schedule.next,spec['tasks'],c.cancel)
 case_to_lane={}
 def select(number):
  item=prescheduled[number]if prescheduled is not None else ordered_screen.next(number)
  if item is None:return None
  scheduled[number]=item;case_to_lane[item['case_id']]=number
  write(root/'screening_selected'/('%d.json'%number),dict(case=item,lane=number,guard_clock_not_reset=True))
  assert c.call(schedule.lanes[item['task']],check_precision)
  return item
 saved={name:getattr(recovery,name)for name in('rng_capture','rng_restore')}
 for name,fn in saved.items():setattr(recovery,name,c.routed(fn))
 cache=CacheMaintenance(torch,c,base,prepared['session'],auxiliary_models=(mask_model,))
 from .forward_cache import install as install_forward_cache
 install_forward_cache(c,torch,models,base,cache,root/'forward_allocator_reclamation.jsonl')
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
 from .startup import StagedLanes,actor_class
 startup=StagedLanes(c.cancel,lambda row:write(root/('startup_release_'+str(len(row['released_lanes']))+'.json'),row),lanes=lanes)
 write(root/'initial_real_lanes_admitted.json',dict(lanes=[0,1],remaining_wait_for_actual_first_query=True,only_original_real_episode_workers=True,no_extra_queries_or_GPU_compute=True,guard_clock_not_reset=True,unused_allocator_scene_hook_active=True))
 AuditedActor=actor_class(OriginalAuditedActor,startup,case_to_lane)
 errors=queue.Queue();finished=queue.Queue();threads=[];results={};started=time.monotonic()
 def run_case(number,item):
  lane=Lane(str(number));head,robots=prepared['heads'][number]
  branches=['baseline','M_event_full','zero_update','shadow_update','TTA']if spec['batch_index']==0 and item['episode_index']==spec['initial_screen_state'][item['task']]['accepted'] else spec['conditions']
  with c.rng.bind(lane):
   cls=getattr(importlib.import_module('envs.'+item['task']),item['task'])
   cfg=make_config(prepared['runtime'],argparse.Namespace(task=item['task'],task_config='demo_randomized'),prepared['session'])
   cfg.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
   folder=root/'cases'/item['case_id'];folder.mkdir(parents=True)
   write(folder/'case.json',dict(case_id=item['case_id'],role='frozen_official_EVAL_no_fit',conditions=list(branches),initialization_eval_mode=True,task_seed_excluded_from_online_forward=True))
   rows=[];initial_identity=None
   for branch in branches:
    c.check_cancel();task=c.call(lane,lambda:c.prepare_task(cls(),lane));raw=lambda:c.call(lane,observe,task)
    routed_seed=c.routed(seed_all);branchfolder=folder/branch;bridge.begin_case(item['case_id']+'/'+branch,number)
    c.call(lane,cache.begin_case,lane,item['case_id']+'/'+branch)
    cleanup=[];actor=None;factory=None;native_diagnostic=None
    try:
     from .initial_integrity import prepare_initial
     from .semantic import content_key
     meta=read(root/'initial_prefetch'/(item['case_id']+'.json'));diagnostic=meta.get('diagnostic_identity')
     if diagnostic is None and spec['batch_index']==8:
      from .prefix_reference import REFERENCE
      previous=read(REFERENCE/'cases'/item['case_id']/'baseline/query000_environment_audit.json');diagnostic={k:previous[k]for k in('physical_sha256','rng_sha256')}
     initial=prepare_initial(task,item,cfg,raw,routed_seed,expected_diagnostic=diagnostic,expected_key=meta['content_key'],instruction=item['instruction'],replay=replay_prefix,key_fn=content_key,unstable=UnStableError,output=folder/(branch+'_initialization_integrity.json'))
     expected,masks,seed_report=prepared['seed_store'].at_initial(initial,item['instruction'],query=0,tick=0)
     identity=dict(physical_sha256=initial['physical_sha256'],rng_sha256=initial['rng_sha256'],RGB_sha256=[__import__('hashlib').sha256(im.tobytes()).hexdigest()for im in initial['rgb']],command_state_sha256=__import__('hashlib').sha256(initial['command_state'].tobytes()).hexdigest())
     if initial_identity is None:initial_identity=identity
     elif identity!=initial_identity:raise ValueError('Matched branches did not start at identical actual initial physics/RNG/RGB/state')
     def diagnostic_owner():
      return dict(physical_sha256=physical(task),rng_sha256=recovery.rng_fingerprint(recovery.rng_capture()))
     native_diagnostic=lambda:c.call(lane,diagnostic_owner)
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

 def worker(number):
  try:
   if not startup.wait(number):return
   item=select(number)
   if item is None:return
   run_case(number,item)
   startup.release('actual_case_terminal',lane=number)
  except BaseException:
   text=traceback.format_exc();write(prepared['session']/('lane%d_failure.json'%number),dict(traceback=text));errors.put(text);c.cancel.set()

 try:
  for number in range(lanes):
   thread=threading.Thread(target=worker,args=(number,),daemon=True);threads.append(thread);thread.start()
  while len(results)<lanes:
   if not errors.empty():raise RuntimeError(errors.get())
   c.service.process_one(timeout=.01)
   while True:
    try:number,record=finished.get_nowait();assert number not in results;results[number]=record
    except queue.Empty:break
  for thread in threads:thread.join(timeout=3)
  assert not any(t.is_alive()for t in threads)and c.service.requests.empty()
  assert frozen_versions==versions(),'Frozen model parameter identity or mutation counter changed'
  assert all(scheduled)and len({item['case_id']for item in scheduled})==lanes
  schedule.finish()
  assert denoise_graph.calls>0
  write(root/'prefix_graph_runtime.json',prefix_graphs.proof())
  write(root/'denoise_graph_runtime.json',dict(actual_denoise_calls=denoise_graph.calls,original_outer_sampler_and_once_noise_unchanged=True,original_model_parameters_and_base_outputs_checked=True,first_resumed_batch_whole_zero_shadow_controls=spec['batch_index']==2))
  write(root/'GPU_episodes_complete.json',dict(complete=True,protocol_identity=spec['identity'],cases=[results[k]for k in sorted(results)],seconds=time.monotonic()-started,scope=spec['scope'],frozen_parameter_pointers_and_mutation_counters_unchanged=True,new_independent_scenes=lanes,TTA_utility_proven=False))
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
