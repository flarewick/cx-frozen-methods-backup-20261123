"""One GPU owner services actual VLA/SAM2; four CPU native episode workers."""
import argparse,importlib,os,queue,threading,time,traceback,weakref
import numpy as np
from native_policy_execution_v1.common import T,read,write,sha
from native_trained_episode_v1.episode import execute as episode,CONDITIONS
from native_live_response_source_v1.source import SourceStream
from native_expert_smoke_v2.observe import observe
from .actor import AuditedActor

def execute(spec,root,manifest,prepared,barrier):
 import torch
 from cx_tta_feedback_confirm5.lanes import Coordinator,Lane,gripper_only
 from cx_tta_feedback_confirm5.throughput import replay_prefix
 from cx_tta_feedback_confirm5.model_math import sam2_profile
 from .cache import CacheMaintenance
 from cx_tta.collect_smoke import make_config
 from cx_tta.evaluate_b2 import seed_all
 from cx_tta.multi_event import EventConfig
 from cx_tta import recovery_contract as recovery
 from native_initial_frame_v3.observe import physical
 from native_completed_memory_v1.reclaim import reclaim
 from native_case_release_v1.release import release as release_case
 from adapter_cache_cpu import move_to_gpu
 from native_physics import load_native
 from collection_native_step import load as load_step
 assert spec['scope']=='four_existing_developer_cases_only'and len(spec['scenes'])==4
 assert tuple(spec['conditions'])==CONDITIONS and spec['simulation_lanes']==4
 assert {s['seed']for s in spec['scenes']}<={7100000,7100001}
 assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==manifest['gpu_uuid']
 torch.set_num_threads(2);base=move_to_gpu(prepared['base']);base._model.eval().requires_grad_(False)
 mask_model=prepared['mask_model'].to('cuda').eval().requires_grad_(False)
 sampler=base._model.sample_actions
 while hasattr(sampler,'_torchdynamo_orig_callable'):sampler=sampler._torchdynamo_orig_callable
 base._sample_actions=sampler
 assert torch.get_float32_matmul_precision()=='high'
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
 original_planner=robot.CuroboPlanner;robot.CuroboPlanner=gripper_only(original_planner)
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
 errors=queue.Queue();finished=queue.Queue();threads=[];results={};started=time.monotonic()
 def worker(number):
  lane=Lane(str(number));item=spec['scenes'][number];head,robots=prepared['heads'][number]
  try:
   with c.rng.bind(lane):
    cls=getattr(importlib.import_module('envs.'+item['task']),item['task'])
    cfg=make_config(prepared['runtime'],argparse.Namespace(task=item['task'],task_config='demo_randomized'),prepared['session'])
    cfg.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
    folder=root/('case%03d'%number);folder.mkdir()
    write(folder/'case.json',dict(case_id=item['case_id'],role='old_training_developer_integration_only',conditions=list(CONDITIONS),initialization_eval_mode=True,task_seed_excluded_from_online_forward=True))
    rows=[];initial_identity=None
    for branch in CONDITIONS:
     c.check_cancel();task=c.call(lane,lambda:c.prepare_task(cls(),lane));raw=lambda:c.call(lane,observe,task)
     routed_seed=c.routed(seed_all);branchfolder=folder/branch;bridge.begin_case(item['case_id']+'/'+branch,number)
     c.call(lane,cache.begin_case,lane,item['case_id']+'/'+branch)
     cleanup=[];actor=None;factory=None;native_diagnostic=None
     try:
      # Fresh native task warmup is reset before this branch's actual q0.
      try:replay_prefix(task,item,[],cfg,raw,routed_seed)
      finally:task.close_env(clear_cache=False)
      initial=replay_prefix(task,item,[],cfg,raw,routed_seed)
      expected,masks,seed_report=prepared['seed_store'].at_initial(initial,item['instruction'],query=0,tick=0)
      identity=dict(physical_sha256=initial['physical_sha256'],rng_sha256=initial['rng_sha256'],RGB_sha256=[__import__('hashlib').sha256(im.tobytes()).hexdigest()for im in initial['rgb']],command_state_sha256=__import__('hashlib').sha256(initial['command_state'].tobytes()).hexdigest())
      if initial_identity is None:initial_identity=identity
      elif identity!=initial_identity:raise ValueError('Matched branches did not start at identical actual initial physics/RNG/RGB/state')
      def diagnostic_owner():
       return dict(physical_sha256=physical(task),rng_sha256=recovery.rng_fingerprint(recovery.rng_capture()))
      native_diagnostic=lambda:c.call(lane,diagnostic_owner)
      actor=AuditedActor(c,base,head,robots,prepared['projection'],cache,name='developer_'+str(number)+'_'+branch,instruction=item['instruction'],native_diagnostic=native_diagnostic)
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
    record=dict(case_id=item['case_id'],all_requested_conditions_retained=True,branches=rows,initial_identity=initial_identity,independent_result=False)
    write(folder/'bundle.json',record);finished.put((number,record))
  except BaseException:
   text=traceback.format_exc();write(prepared['session']/('lane%d_failure.json'%number),dict(traceback=text));errors.put(text);c.cancel.set()
 try:
  for number in range(4):
   thread=threading.Thread(target=worker,args=(number,),daemon=True);threads.append(thread);thread.start()
  while len(results)<4:
   if not errors.empty():raise RuntimeError(errors.get())
   c.service.process_one(timeout=.01)
   while True:
    try:number,record=finished.get_nowait();assert number not in results;results[number]=record
    except queue.Empty:break
  for thread in threads:thread.join(timeout=3)
  assert not any(t.is_alive()for t in threads)and c.service.requests.empty()
  assert frozen_versions==versions(),'Frozen model parameter identity or mutation counter changed'
  write(root/'GPU_episodes_complete.json',dict(complete=True,protocol_identity=spec['identity'],cases=[results[i]for i in range(4)],seconds=time.monotonic()-started,scope='developer_integration_only',frozen_parameter_pointers_and_mutation_counters_unchanged=True,new_independent_scenes=0,TTA_utility_proven=False))
 finally:
  c.cancel.set()
  for thread in threads:thread.join(timeout=2)
  telemetry.capture=original_capture;cache.uninstall()
  for name,fn in saved.items():setattr(recovery,name,fn)
  robot.CuroboPlanner=original_planner;c.rng.uninstall()
