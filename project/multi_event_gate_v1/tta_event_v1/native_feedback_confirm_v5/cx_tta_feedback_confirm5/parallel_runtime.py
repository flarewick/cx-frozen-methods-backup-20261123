"""Persistent four native lanes; CPU parent validates, GPU owner stays useful."""
import argparse,importlib,os,queue,signal,threading,time,traceback
import numpy as np
from .paths import PROJECT,read,write,output,digest
from .runtime import Observer
from .views import batched_encode
from .throughput import TaskPool,replay_prefix
from .lanes import Coordinator,Lane,gripper_only
from .schedule import instrument
from .stream_schedule import StreamSchedule
from .task_affinity import assigned_task
from .actor_trace import ActorTrace
from parent_validation import request
def execute(spec,root,manifest,journal,prepared=None):
 if(spec.get('runtime_status')!='FROZEN_INDEPENDENT_TRAIN_CONFIRMATION'or prepared is None
     or prepared.get('source_bound_GPU_permit')is not True):
  raise ValueError('Source-bound CPU preflight and guarded GPU permit required')
 spec=dict(spec,gpu_uuid_runtime=manifest['gpu_uuid'])
 import torch
 from cx_tta.collect_smoke import make_config
 from cx_tta.evaluate_b2 import seed_all
 from adapter_cache_cpu import move_to_gpu
 from native_physics import load_native
 from collection_native_step import load as load_step
 from cx_tta import recovery_contract as recovery
 assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==manifest['gpu_uuid']
 torch.set_num_threads(2)
 session=prepared['session'];barrier=prepared['barrier'];ipc=prepared['ipc'];nonce=prepared['nonce']
 initial=len(journal['cases']);start=time.monotonic();status_lock=threading.Lock()
 lane_status={};captured={};threads=[];c=None;original_rng={};robot=None;old_planner=None;restore_prefix=None;trace=None;cache=None
 pools=[TaskPool()for _ in range(4)]
 # Strong references survive execute() and disappear only at GPU child's os._exit.
 prepared['GPU_live_keepalive']=dict(pools=pools)
 def note(phase,lane=None,**fields):
  with status_lock:
   if lane is not None:lane_status[lane]=dict(phase=phase,**fields)
   value=dict(identity=manifest['identity'],session=session.name,phase=phase,seconds=time.monotonic()-start,
      captured=initial+len(captured),requested=len(spec['scenes']),validated=initial+len(list((ipc/'acks').glob('*.json'))),
      lanes=lane_status.copy(),**fields)
   write(root/'progress.json',value);print('STREAM_STATUS',__import__('json').dumps(value),flush=True)
 def interrupted(*_):raise InterruptedError('No incomplete case is accepted')
 signal.signal(signal.SIGTERM,interrupted)
 try:
  note('moving_preloaded_frozen_model')
  base=move_to_gpu(prepared['base']);base._model.eval().requires_grad_(False)
  mask_model=prepared['mask_model'].to('cuda').eval().requires_grad_(False)
  import cv2
  cv2.setNumThreads(2)
  import sam2.sam2_video_predictor as official
  official.tqdm=lambda iterable,**kwargs:iterable
  sampler=base._model.sample_actions
  while hasattr(sampler,'_torchdynamo_orig_callable'):sampler=sampler._torchdynamo_orig_callable
  base._sample_actions=sampler
  write(barrier/'cuda_ready.json',dict(inner_pid=os.getpid(),gpu_uuid=manifest['gpu_uuid']))
  began=time.monotonic()
  while not(barrier/'owned_pid_permit.json').exists():
   if time.monotonic()-began>45:raise TimeoutError('No verified GPU owner permit')
   time.sleep(.03)
  assert read(barrier/'owned_pid_permit.json')['gpu_uuid']==manifest['gpu_uuid']
  encode=torch.compiler.disable(lambda views,prompt:batched_encode(base,views,prompt))
  with np.load(spec['projection'],allow_pickle=False)as z:projection={k:z[k].copy()for k in z.files}
  c=Coordinator(cuda=True,native=load_native(),native_step=load_step());c.rng.install()
  from .cache_maintenance import CacheMaintenance
  cache=CacheMaintenance(torch,c,base,session,cpu_only_fixture=False,auxiliary_models=(mask_model,));original_encode=encode
  encode=torch.compiler.disable(lambda views,prompt:cache.encode(original_encode,views,prompt))
  trace=ActorTrace(base,c.rng)
  prepared['GPU_live_keepalive'].update(coordinator=c,model=base,trace=trace,cache=cache,mask_model=mask_model)
  original_rng={name:getattr(recovery,name)for name in('rng_capture','rng_restore')}
  for name,fn in original_rng.items():setattr(recovery,name,c.routed(fn))
  routed_seed=c.routed(seed_all)
  from .prefix_time import install_native
  restore_prefix=install_native()
  import envs.robot.robot as robot
  old_planner=robot.CuroboPlanner;robot.CuroboPlanner=gripper_only(old_planner)
  classes={name:getattr(importlib.import_module('envs.'+name),name)for name in{s['task']for s in spec['scenes']}}
  schedule=StreamSchedule(spec['scenes'],initial,spec['bridge_count'])
  errors=queue.Queue();finished=queue.Queue()
  def work(lane):
   pool=pools[int(lane.name)];warmed=False;head,robots=prepared['CPU_lane_heads'][int(lane.name)]
   jobs=schedule.queues[assigned_task(lane.name)]
   try:
    with c.rng.bind(lane):
     while not c.cancel.is_set():
      try:index=jobs.get(timeout=.05)
      except queue.Empty:continue
      if index is None:jobs.task_done();return
      item=spec['scenes'][index]
      if item['task']!=assigned_task(lane.name):raise ValueError('Task escaped its persistent lane')
      cfg=make_config(prepared['runtime'],argparse.Namespace(task=item['task'],task_config='demo_randomized'),session)
      cfg.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
      def factory():return c.call(lane,lambda:c.prepare_task(classes[item['task']](),lane))
      task=pool.get(item['task'],factory)
      def observer():
       obj=Observer(task,lambda views,prompt:c.call(lane,encode,views,prompt),projection,item['instruction'])
       physical=obj.physical
       def protected():
        with c.scene_gate.read():return physical()
       obj.physical=protected;return obj
      if not warmed:
       note('one_time_lane_warmup',lane=lane.name,case=item['case_id'])
       try:replay_prefix(task,item,[],cfg,observer().read,routed_seed)
       finally:task.close_env(clear_cache=False)
       warmed=True
      folder=output(session/('scene%03d_%s_seed%d_lane%s'%(index,item['task'],item['seed'],lane.name)));folder.mkdir()
      write(folder/'case.json',dict(case_id=item['case_id'],source_family='current_RGB_native',
          role='excluded_compatibility'if index<spec['bridge_count']else'independent_TRAIN_confirmation',
          legacy_full32=False,requires_sequential_reference=item.get('requires_sequential_reference',False),
          legacy_source_unavailable_reason=item.get('legacy_source_unavailable_reason'),
          coverage_start_query=item.get('coverage_start_query',0),max_measurement_ticks=5000))
      from .current_case import run_current
      c.call(lane,cache.begin_case,lane,item['case_id'])
      c.call(lane,trace.begin,lane,item['case_id'])
      run_current(task,c,cfg,item,spec,index,folder,lane,base,head,robots,projection,observer,routed_seed,note,mask_model=mask_model,seed_masks=prepared['seed_masks'][item['case_id']],feedback_geometry=(*prepared['feedback_hardware'],robots),predictor=prepared['predictor'],cache=cache)
      c.call(lane,trace.finish,lane,folder)
      c.call(lane,cache.finish_case,lane,folder)
      request(ipc,index-initial,str(folder.relative_to(root)),nonce=nonce,
          protocol_identity=spec['identity'],case_id=item['case_id'])
      finished.put((index,folder));note('complete_files_posted_to_CPU_parent',lane=lane.name,case=item['case_id'])
      jobs.task_done()
   except BaseException:
    value=traceback.format_exc();write(session/('lane'+lane.name+'_failure.json'),dict(traceback=value))
    errors.put(value);c.cancel.set()
  for i in range(4):
   t=threading.Thread(target=work,args=(Lane(str(i)),),daemon=True);threads.append(t);t.start()
  while len(captured)<len(spec['scenes'])-initial:
   if(ipc/'validation_failure.json').exists():raise ValueError('CPU parent rejected case: '+str(read(ipc/'validation_failure.json')))
   if not errors.empty():raise RuntimeError(errors.get())
   c.service.process_one(timeout=.01)
   while True:
    try:index,folder=finished.get_nowait();captured[index]=str(folder.relative_to(root))
    except queue.Empty:break
   if not schedule.tail_admitted and(ipc/'bridges_passed.json').exists():
    schedule.admit_after_bridges(read(ipc/'bridges_passed.json'),nonce,spec['identity'])
    note('CPU_bridges_passed_remaining_stream_admitted')
  for t in threads:t.join(timeout=3)
  if any(t.is_alive()for t in threads):raise RuntimeError('Persistent worker teardown incomplete')
  write(session/'capture_complete.json',dict(capture_complete=True,captured_cases=len(captured),initial_committed=initial,
      source_integrity_and_deep_case_validation_deferred_to_CPU_parent=True,
      no_global_wave_cache_purges=True,persistent_task_owners=sum(len(p.tasks)for p in pools),
      folders={str(k):v for k,v in sorted(captured.items())},online_updates=0,TTA_utility_proven=False))
  note('GPU_required_work_complete_CPU_parent_finishes_validation')
 except BaseException:
  write(session/'failure.json',dict(traceback=traceback.format_exc(),capture_complete=False));raise
 finally:
  if c is not None:
   c.cancel.set()
   for t in threads:t.join(timeout=2)
   if trace is not None:trace.uninstall()
   if cache is not None:cache.uninstall()
   for name,fn in original_rng.items():setattr(recovery,name,fn)
   c.rng.uninstall()
  if robot is not None and old_planner is not None:robot.CuroboPlanner=old_planner
  if restore_prefix is not None:restore_prefix()
  # No deep verification, source hashes, global GC, render-cache clear or model deletion here.
