from pathlib import Path

import os,sys,subprocess,time,uuid,fcntl,traceback,json,argparse

from .common import C,T,P,O,TASKS,COHORT,CANON,read,write,replace,sha,identity,mapped,verify,batch

sys.path.insert(0,str(P/'deployment'));sys.path.insert(0,str(T/'resource_guard_v1'));sys.path.insert(0,str(T/'native_nextchunk_v1'))
import user_container as container

import gpu_doctor

from native_resource_policy_v4 import admission

from native_process_sampler_v13 import guard

from native_fast_inventory_v1.inventory import inventory

from native_visible_expansion_v1.placement import plan,apply

def canonical(p):return str(CANON/Path(p).resolve().relative_to(P))

def script(mode,module,*args):return container.command(['env','CX_OBJECT_STAGE='+COHORT,'bash',canonical(C/'scripts'/('cpu.sh'if mode=='cpu'else'gpu.sh')),*(['-m',module]if mode=='cpu'else[]),*args],mount_project=True)

def wait_file(path,process,seconds):
 start=time.monotonic()
 while not path.exists():
  if process.poll()is not None:raise RuntimeError('Prepared process exited before '+str(path)+'; inspect log')
  if time.monotonic()-start>seconds:raise TimeoutError(str(path))
  time.sleep(.25)
 return read(path)

def choose():
 start=time.monotonic()
 while True:
  fleet=inventory()
  if fleet['own_gpu_count']<3:
   for gpu in (6,1,4,2,0,3):
    if gpu not in fleet['available_indices']:continue
    info=gpu_doctor.gpu_snapshot(gpu);cert=container.WORK/('gpu-doctor-'+str(gpu)+'.json')
    if not cert.exists():continue
    d=read(cert)
    if d['uuid']!=info['uuid']or d['build_mtime_ns']!=(container.WORK/'stage-bootstrap.ok').stat().st_mtime_ns:continue
    admission.require_admission(fleet,info['uuid']);return gpu,info,plan(gpu),fleet
  if time.monotonic()-start>36000:raise TimeoutError('No free certified GPU below account3 limit')
  replace(O/'live_status.json',dict(phase='CPU_only_wait_for_empty_GPU',fleet=fleet));time.sleep(10)

def terminal(root,spec):
 rows=[json.loads(s)for s in(root/'attempt/resources.jsonl').read_text().splitlines()];child=read(root/'attempt/barrier/normal_child_exit.json');result=read(root/'result.json')
 assert child['exit_code']==0 and child['normal_exit']and child['proc_absent_after_wait']and not child['cpu_only_fixture']and result['complete']
 assert child['protocol_identity']==spec['identity']and not any('stopped'in r for r in rows)
 released=[r for r in rows if'own_gpu_released'in r];assert len(released)==1 and released[0]['own_gpu_released']and not released[0]['remaining_own_gpu_pids']
 steady=[r for r in rows if r.get('seconds',0)>=280 and r.get('gpu_pids')];assert steady,'Actual scientific batch must qualify sustained resource metrics'
 for r in steady:
  assert min(r['mean_utilization'],r['time_weighted_mean_utilization'],r['rolling_300s_utilization'])>=34 and 4096<=r['memory_mib']<=24563 and r['account_gpu_count']<=3
  for pid in r['gpu_pids']:
   k=str(pid);assert r['process_utilization'][k]is not None and min(r['process_mean_utilization'][k],r['process_rolling_300s_utilization'][k])>=34 and r['process_memory_mib'][k]>=4096 and 40<=r['cpu'][k]<=1600 and r['rss_bytes'][k]<=128*2**30
 proof=read(root/'attempt/NVML_stream_actual_exit.json');assert proof['exit_code']==0 and proof['actual_wait_return']and proof['proc_absent_after_wait']and proof['request_count']>0 and proof['no_CUDA_or_GPU_work']
 from native_process_sampler_v13.nvml_query import render
 queries=[json.loads(line)for line in(root/'attempt/NVML_queries.jsonl').read_text().splitlines()];assert queries and all('error'not in q and q['timeout_seconds']==3 and q['automatic_retry']is False and q['CLI_used']is False for q in queries)
 for q in queries:render(q['raw_NVML'],q['arguments'])
 from native_process_sampler_v13.device_audit import audit as audit_device_stream
 device_audit=audit_device_stream(rows,read(root/'post_release_device_samples.json'),spec['required_gpu_uuid'],(root/'attempt/barrier/permit.json').stat().st_mtime_ns);write(root/'CPU_device_stream_audit.json',device_audit)
 assert read(root/'semantic/result.json')['complete']and read(root/'semantic_exit.json')['exit_code']==0
 assert read(root/'CPU_original_prefix_comparison.json')['passed']and read(root/'CPU_feedback_application_audit.json')['passed']
 verify(spec);out=dict(passed=True,actual_host_exit_code=0,actual_child_exit_code=0,own_GPU_released=True,steady_resource_samples=len(steady),batch_protocol_identity=spec['identity'],input_sha256={str(f):sha(f)for f in [root/'result.json',root/'CPU_episode_audit.json',root/'CPU_feedback_application_audit.json',root/'attempt/resources.jsonl',root/'attempt/barrier/normal_child_exit.json',root/'semantic/result.json',root/'semantic_exit.json',root/'attempt/NVML_queries.jsonl',root/'post_release_device_samples.json',root/'CPU_device_stream_audit.json',root/'attempt/NVML_stream_actual_exit.json']})
 write(root/'terminal_audit.json',out);return out

def execute_batch(root,spec,gpu,info):
 assert not(root/'attempt').exists(),'Existing GPU attempt must be inspected, never retried unchanged'
 semlog=(root/'semantic_host.log').open('x');glog=(root/'GPU_host.log').open('x');sem=None;process=None;start=time.monotonic()
 try:
  semcmd=script('cpu','native_object_transfer_pilot_v3.semantic','--batch',canonical(root));sem=subprocess.Popen(semcmd,stdout=semlog,stderr=subprocess.STDOUT,start_new_session=True)
  write(root/'semantic_launch.json',dict(pid=sem.pid,started_ticks=int(Path('/proc/%d/stat'%sem.pid).read_text().split(') ',1)[1].split()[19]),command=semcmd));ready=wait_file(root/'semantic/ready.json',sem,240);assert ready['passed']and not ready['CUDA_initialized']and ready['source_identity']==spec['identity']
  for row in(spec.get('cached_preparation')or{}).get('requests',[]):write(root/'semantic/requests'/(row['content_key']+'.json'),row)
  attempt=root/'attempt';attempt.mkdir();barrier=attempt/'barrier';nonce=uuid.uuid4().hex
  cmd=script('gpu',None,'--batch',canonical(root),'--barrier',canonical(barrier),'--nonce',nonce);split=cmd.index('--');cmd[split:split]=gpu_doctor.driver_options(info)
  process=subprocess.Popen(cmd,stdout=glog,stderr=subprocess.STDOUT,start_new_session=True);ticks=int(Path('/proc/%d/stat'%process.pid).read_text().split(') ',1)[1].split()[19]);write(attempt/'launch.json',dict(pid=process.pid,started_ticks=ticks,command=cmd,gpu=info,nonce=nonce))
  proof=wait_file(barrier/'ready.json',process,240);assert proof['passed']and proof['cpu_fork_child_passed']and proof['os_threads']==1 and not proof['cuda_initialized']and proof['source_sha256']==spec['source_sha256'];write(root/'source_bound_CPU_preload_proof.json',proof)
  for row in(spec.get('cached_preparation')or{}).get('requests',[]):wait_file(root/'semantic/answers'/(row['content_key']+'.json'),sem,240)
  write(root/'CPU_preparation_ready.json',dict(cached_initials_recomputed_before_GPU=len((spec.get('cached_preparation')or{}).get('requests',[])),CUDA_permit_not_written=not(barrier/'permit.json').exists()))
  lockdir=container.WORK/'gpu_locks';lockdir.mkdir(exist_ok=True);waiting=time.monotonic()
  with(lockdir/'account_admission.lock').open('a')as account,(lockdir/('gpu_'+str(gpu)+'.lock')).open('a')as card:
   while True:
    try:
     fcntl.flock(account,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(card,fcntl.LOCK_EX|fcntl.LOCK_NB);admission.check_admission(gpu_doctor.gpu_snapshot(gpu));break
    except(BlockingIOError,RuntimeError):
     fcntl.flock(account,fcntl.LOCK_UN);fcntl.flock(card,fcntl.LOCK_UN)
     if time.monotonic()-waiting>1500:raise TimeoutError('CPU admission wait expired without GPU')
     time.sleep(2)
   began=time.monotonic();started_ns=time.monotonic_ns();released=False;permitted=False
   def permit():
    admission.check_admission(gpu_doctor.gpu_snapshot(gpu));write(barrier/'permit.json',dict(gpu_uuid=info['uuid'],attempt_nonce=nonce,protocol_identity=spec['identity']))
   def sample(row):
    nonlocal released,permitted
    if len(row['gpu_pids'])>1:return 'Exactly one GPU owner required'
    if row['gpu_pids']and not released:fcntl.flock(account,fcntl.LOCK_UN);released=True
    if row['gpu_pids']and not permitted and(barrier/'cuda_ready.json').exists():write(barrier/'owned_pid_permit.json',dict(gpu_uuid=info['uuid'],host_GPU_PID=row['gpu_pids'][0]));permitted=True
    replace(O/'live_status.json',dict(phase='paired_GPU_evaluation',batch=spec['batch_index'],gpu=gpu,protocol_identity=spec['identity'],resources=row))
   guard.guarded_command(cmd,info,attempt/'resources.jsonl',gpu=gpu,deadline=7200,on_sample=sample,min_cumulative_utilization=34,max_memory_mib=24563,prepared_process=process,before_monitor=permit,child_exit_proof=barrier/'normal_child_exit.json',child_exit_identity=dict(attempt_nonce=nonce,gpu_uuid=info['uuid'],protocol_identity=spec['identity'],cpu_only_fixture=False,attempt_started_ns=started_ns))
  code=process.wait();assert code==0;write(root/'actual_GPU_parent_exit.json',dict(exit_code=code,actual_Popen_wait_return=True,pid=process.pid,started_ticks=ticks))
  from native_process_sampler_v13.metrics import process_utilization,diagnostics
  process_utilization(gpu);write(root/'post_release_device_samples.json',dict(collected_unix_us=time.time_ns()//1000,raw_NVML=diagnostics()['raw_NVML'],actual_GPU_parent_exit0_and_release_already_verified=True,no_new_GPU_work=True))
  write(root/'semantic/stop.json',dict(GPU_parent_actual_exit0=True));semcode=sem.wait(timeout=90);write(root/'semantic_exit.json',dict(exit_code=semcode,actual_Popen_wait_return=True,pid=sem.pid));assert semcode==0
  terminal(root,spec)
 except BaseException:
  write(root/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
  from native_process_sampler_v13.metrics import diagnostics as last_actual_metrics
  diagnostic=last_actual_metrics()
  if diagnostic:
   write(root/'last_successfully_collected_metric_at_failure.json',dict(metric=diagnostic,may_precede_exception=True,no_new_NVML_read_performed=True,current_attempt_permit_exists=(root/'attempt/barrier/permit.json').exists(),metric_unix_ns=diagnostic.get('unix_ns')))
  if process is not None and process.poll()is None:
   known=guard.stop_job(process);guard.record_release(process,info,start,root/'failed_release.jsonl',known)
  if sem is not None and sem.poll()is None:
   if not(root/'semantic/stop.json').exists():write(root/'semantic/stop.json',dict(reason='GPU_or_CPU_attempt_failed_no_retry'))
   try:code=sem.wait(timeout=30)
   except subprocess.TimeoutExpired:guard.stop_job(sem);code=sem.returncode
   if not(root/'semantic_exit.json').exists():write(root/'semantic_exit.json',dict(exit_code=code,actual_Popen_wait_return=True,pid=sem.pid))
  raise
 finally:semlog.close();glog.close()

def freeze():
 from .contracts import check
 check()
 if(O/'study.json').exists():
  s=read(O/'study.json');verify(s);return s
 parent_path=T/'outputs/native_frozen100_v25/mixed/study.json';parent=read(parent_path);verify(parent)
 keys=('trained_qualification','training','runtime_source','projection','event_config','actor_sampler_seed','candidate_checkpoint_sha256')
 s={k:parent[k]for k in keys};sources=dict(parent['source_sha256']);inputs=dict(parent['input_sha256'])
 inputs[str(parent_path)]=sha(parent_path)
 from .prefix_reference import reference_inputs
 inputs.update(reference_inputs())
 for f in(T/'outputs/native_feedback_application_v1/CPU_prior_native_action_audit.json',T/'outputs/native_object_feedback_v5/CPU_official_language_coverage.json',T/'outputs/native_object_transfer_pilot_v3/development/CPU_integration_checks.json',T/'outputs/native_feedback_application_v1/CPU_contract_checks_v2.json',T/'outputs/native_feedback_application_v1/CPU_actual_TRAIN_first_change_replay.json',T/'outputs/native_object_feedback_v5/CPU_development_seeds/result.json',T/'outputs/native_object_transfer_pilot_v2/development/study.json'):
  inputs[str(f)]=sha(f)
 for f in(P/'deployment/shared6_task_bookkeeping.py',P/'deployment/reports/target_shared6_task_bookkeeping_cpu_20261024.json',T/'outputs/native_object_transfer_pilot_v1/development/study.json',T/'outputs/native_object_transfer_pilot_v1/development/recovery_control.json'):
  inputs[str(f)]=sha(f)
 inputs[str(T/'outputs/native_object_transfer_pilot_v2/development/CPU_original_task_bookkeeping_checks.json')]=sha(T/'outputs/native_object_transfer_pilot_v2/development/CPU_original_task_bookkeeping_checks.json')
 sources['deployment/shared6_task_bookkeeping.py']=sha(P/'deployment/shared6_task_bookkeeping.py')
 for directory in(C,T/'native_object_feedback_v3',T/'native_object_feedback_v4',T/'native_object_feedback_v5',T/'native_feedback_application_v1'):
  for f in directory.rglob('*'):
   if f.is_file()and f.suffix in('.py','.sh','.md'):sources[str(f.relative_to(P))]=sha(f)
 for f in(T/'outputs/native_object_feedback_v3/CPU_interface_checks.json',T/'outputs/native_object_feedback_v3/CPU_actual_SAM2_checks.json'):
  value=read(f);inputs[str(f)]=sha(f)
  inputs.update(value.get('input_sha256',{}));inputs.update(value.get('language',{}).get('input_sha256',{}))
 if COHORT=='confirmation':
  development=T/'outputs/native_object_transfer_pilot_v3/development/study.json';verify(read(development));inputs[str(development)]=sha(development)
  assert read(development)['source_sha256']==sources,'Freeze candidate before independent confirmation; changes need new version'
 start=7109000 if COHORT=='development'else 7111000
 s.update(version='native_object_transfer_pilot_v3',method_candidate='TTA_OBJECT_V5_FAST_DIFFERENCE_DEV_20261102',candidate_interface_version='native_object_feedback_v5',application_version='native_feedback_application_v1',user_approved_accepted_fast_only_additional_opportunity=True,scope='new_object_feedback_independent_TRAIN_interface_pilot',scene_role='TRAIN_development'if COHORT=='development'else'independent_TRAIN_confirmation_no_tuning',stage=COHORT,tasks=list(TASKS),conditions=['baseline','M_event_full','TTA'],actual_seed_start=start,max_screen_attempts_per_task=1000,episodes_per_task=4,logical_batch_count=2,cases_per_batch=10,simulation_lanes=10,task_slots=list(TASKS)*2,batch_index=0,initial_screen_state={t:dict(next_seed=start,accepted=0)for t in TASKS},source_sha256=sources,input_sha256=inputs,perception_interface='native_object_feedback_v5',frozen093_weights_and_fast_optimizer_unchanged=True,official_EVAL_used=False,first_accepted_scene_each_task_has_whole_zero_and_shadow_controls=True,original_native_step_limits_unchanged=True,required_interface_confirmation=dict(all_five_tasks_have_real_grounding=True,all_five_tasks_have_causal_eligible_packet=True,all_five_tasks_have_accepted_update_with_later_actual_action_change=True,whole_controls_exact=True,all_actual_normal_exits_and_resource_audits=True),missing_feedback_or_zero_updates_not_removed_from_denominator=True,quarantined_unreliable_PID_sensor_UUIDs=['GPU-0f89dea0-728e-1d23-fd62-d25b9bad3e4f'],resource_limits=dict(minimum_utilization=34,VRAM_MiB=[4096,24563],CPU_percent=[40,1600],maximum_CPU_cores=16,maximum_RSS_GiB=128,maximum_account_GPUs=3,attempt_seconds=7200),no_automatic_retry_of_unknown_failure=True,original_native_success_bookkeeping_installed_before_policy_setup=True,external_interruption_count_imported=1 if COHORT=='development'else 0,previous_primary_branches_all_excluded=True)
 s['identity']=identity(s);verify(s);write(O/'study.json',s);return s

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');a=parser.parse_args();study=freeze()
 if a.freeze_only:print('OBJECT_FEEDBACK_PILOT_FROZEN',study['identity'],flush=True);return
 with(O/'campaign.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state={t:dict(next_seed=study['actual_seed_start'],accepted=0)for t in TASKS};qualified=[]
  os.environ['CX_BEST100_RUN_ID']=uuid.uuid4().hex
  for n in range(study['logical_batch_count']):
   existing=list((O/'batches'/('%03d'%n)/'runs').glob('*'));passed=[r for r in existing if(r/'terminal_audit.json').exists()]
   if passed:
    assert len(passed)==1 and read(passed[0]/'terminal_audit.json')['passed'];verify(read(passed[0]/'protocol.json'));qualified.append(passed[0]);state=read(passed[0]/'screening_complete.json')['state'];continue
   assert not existing,'Inspect every failed attempt, never retry unchanged'
   root=batch(n);gpu,info,placement,fleet=choose();root.mkdir(parents=True);write(root/'cpu_placement_plan.json',placement);write(root/'device_plan.json',dict(gpu=gpu,info=info,fresh_fleet=fleet))
   from .preparation import collect
   cached=collect(study,n,state)
   if COHORT=='development'and n==0:assert cached and cached['cached_valid_cases']==10
   write(root/'preparation_cache.json',cached)
   spec=dict(study,cached_preparation=cached,batch_index=n,initial_screen_state=state,CPU_placement=canonical(root/'cpu_placement_plan.json'),required_gpu_uuid=info['uuid'],study_identity=study['identity']);spec.pop('identity');spec['input_sha256']=dict(spec['input_sha256'],**{canonical(O/'study.json'):sha(O/'study.json'),canonical(root/'cpu_placement_plan.json'):sha(root/'cpu_placement_plan.json'),canonical(root/'device_plan.json'):sha(root/'device_plan.json')});spec['input_sha256'].update((cached or{}).get('input_sha256',{}));spec['input_sha256'][canonical(root/'preparation_cache.json')]=sha(root/'preparation_cache.json');spec['identity']=identity(spec);write(root/'protocol.json',spec)
   replace(O/'execution_state.json',dict(root=str(root),unit='cx-tta-object-v5-'+COHORT+'-engine3.service'))
   allowed=os.sched_getaffinity(0)
   try:apply(placement);execute_batch(root,spec,gpu,info)
   finally:os.sched_setaffinity(0,allowed)
   from .report import summarize
   value=summarize(root,spec);qualified.append(root);state=read(root/'screening_complete.json')['state'];replace(O/'live_status.json',dict(phase='qualified_TRAIN_batch_complete',root=str(root),qualified_batches=len(qualified),target_batches=study['logical_batch_count'],state=state))
  assert all(state[t]['accepted']==study['episodes_per_task']for t in TASKS)
  from .report import aggregate
  value=aggregate(qualified,study);replace(O/'live_status.json',dict(phase='pilot_complete',interface_qualified=value['interface_qualified'],complete_native_episodes=value['actual_native_episodes']));print('OBJECT_FEEDBACK_STAGE_COMPLETE',json.dumps(value),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  replace(O/'campaign_failure.json',dict(traceback=traceback.format_exc(),automatic_GPU_retry=False));raise
