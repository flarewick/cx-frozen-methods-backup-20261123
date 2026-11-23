from pathlib import Path

import os,sys,subprocess,time,uuid,fcntl,traceback,json,argparse

from .common import C,T,P,O,TASKS,COHORT,CANON,read,write,replace,sha,identity,mapped,verify,batch

sys.path.insert(0,str(P/'deployment'));sys.path.insert(0,str(T/'resource_guard_v1'));sys.path.insert(0,str(T/'native_nextchunk_v1'))
import user_container as container

import gpu_doctor

from native_resource_policy_v4 import admission
from native_server_rule4_v1.admission import require as require_rule4

from native_process_sampler_v13 import guard
from native_onpolicy_sampler_health_v1.health import check as check_sampler_health

from native_fast_inventory_v1.inventory import inventory

from native_visible_expansion_v1.placement import plan,apply

def canonical(p):return str(CANON/Path(p).resolve().relative_to(P))

def script(mode,module,*args):return container.command(['env','CX_OBJECT_STAGE=development','CX_EXPERT_STAGE='+COHORT,'bash',canonical(C/'scripts'/('cpu.sh'if mode=='cpu'else'gpu.sh')),*(['-m',module]if mode=='cpu'else[]),*args],mount_project=True)

def wait_file(path,process,seconds):
 start=time.monotonic()
 while not path.exists():
  if process.poll()is not None:raise RuntimeError('Prepared process exited before '+str(path)+'; inspect log')
  if time.monotonic()-start>seconds:raise TimeoutError(str(path))
  time.sleep(.25)
 return read(path)

def choose():
 start=time.monotonic();failed_metadata_devices=set()
 while True:
  fleet=inventory()
  if fleet['own_gpu_count']<3:
   for gpu in (4,0,6,1,2):
    if gpu not in fleet['available_indices']:continue
    info=gpu_doctor.gpu_snapshot(gpu);cert=container.WORK/('gpu-doctor-'+str(gpu)+'.json')
    if not cert.exists():continue
    d=read(cert)
    if d['uuid']!=info['uuid']or d['build_mtime_ns']!=(container.WORK/'stage-bootstrap.ok').stat().st_mtime_ns:continue
    admission.require_admission(fleet,info['uuid'])
    try:require_rule4(info['uuid'])
    except RuntimeError:continue
    if info['uuid']in failed_metadata_devices:continue
    proof,health=check_sampler_health(gpu,info['uuid'],O/'sampler_admission')
    if not health['passed']:
     failed_metadata_devices.add(info['uuid']);continue
    replace(O/'selected_sampler_health.json',dict(proof=str(proof),proof_sha256=sha(proof)))
    return gpu,info,plan(gpu),fleet
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
 assert read(root/'CPU_full_capture_audit.json')['passed']
 verify(spec);out=dict(passed=True,actual_host_exit_code=0,actual_child_exit_code=0,own_GPU_released=True,steady_resource_samples=len(steady),batch_protocol_identity=spec['identity'],input_sha256={str(f):sha(f)for f in [root/'result.json',root/'CPU_episode_audit.json',root/'CPU_full_capture_audit.json',root/'attempt/resources.jsonl',root/'attempt/barrier/normal_child_exit.json',root/'semantic/result.json',root/'semantic_exit.json',root/'attempt/NVML_queries.jsonl',root/'post_release_device_samples.json',root/'CPU_device_stream_audit.json',root/'attempt/NVML_stream_actual_exit.json']})
 write(root/'terminal_audit.json',out);return out

def execute_batch(root,spec,gpu,info):
 assert not(root/'attempt').exists(),'Existing GPU attempt must be inspected, never retried unchanged'
 semlog=(root/'semantic_host.log').open('x');glog=(root/'GPU_host.log').open('x');sem=None;process=None;start=time.monotonic()
 try:
  semcmd=script('cpu','native_onpolicy_coverage_v8.semantic','--batch',canonical(root));sem=subprocess.Popen(semcmd,stdout=semlog,stderr=subprocess.STDOUT,start_new_session=True)
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
     fcntl.flock(account,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(card,fcntl.LOCK_EX|fcntl.LOCK_NB);require_rule4(info['uuid']);admission.check_admission(gpu_doctor.gpu_snapshot(gpu));break
    except(BlockingIOError,RuntimeError):
     fcntl.flock(account,fcntl.LOCK_UN);fcntl.flock(card,fcntl.LOCK_UN)
     if time.monotonic()-waiting>1500:raise TimeoutError('CPU admission wait expired without GPU')
     time.sleep(2)
   began=time.monotonic();started_ns=time.monotonic_ns();released=False;permitted=False
   def permit():
    require_rule4(info['uuid']);admission.check_admission(gpu_doctor.gpu_snapshot(gpu));write(barrier/'permit.json',dict(gpu_uuid=info['uuid'],attempt_nonce=nonce,protocol_identity=spec['identity']))
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
 check();CPUproof=O/'CPU_stream_contract_checks.json';checked=read(CPUproof);assert checked['passed']and checked['concurrent_owner_full_recovery_bridge_isolation_passed']and checked['actual_nested_same_lane_RNG_test_passed'];verify(checked)
 if(O/'study.json').exists():
  value=read(O/'study.json');verify(value);return value
 prior=T/'outputs/native_onpolicy_full_v1/pilot_fit';previous=mapped(read(prior/'execution_state.json')['root']);parent=read(previous/'protocol.json');verify(parent)
 assert read(previous/'terminal_audit.json')['passed']and read(prior/'actual_host_wait_exit.json')['exit_code']==0
 initial=read(previous/'screening_complete.json')['state'];assert all(s['accepted']==4 for s in initial.values())
 value=dict(parent);value.pop('identity');value.pop('cached_preparation',None)
 value.update(version='native_onpolicy_coverage_v8',initial_screen_state=initial,episodes_per_task=8,simulation_lanes=8,cases_per_batch=20,task_slots=list(TASKS)*4,scene_role='same_fixed20_TRAIN_development_population_semantic_candidate_check_not_new_independent_scenes',coverage_expansion_only=True,interleaved_original_screen_and_full=True,no_planner_concurrency=True,no_automatic_retry_of_unknown_failure=True)
 value['source_sha256']=dict(parent['source_sha256']);value['input_sha256']=dict(parent['input_sha256'])
 for file in C.rglob('*'):
  if file.is_file()and file.suffix in('.py','.sh','.md'):value['source_sha256'][str(file.relative_to(P))]=sha(file)
 for file in(T/'native_server_rule4_v1').glob('*.py'):value['source_sha256'][str(file.relative_to(P))]=sha(file)
 for file in(previous/'protocol.json',previous/'terminal_audit.json',previous/'screening_complete.json',prior/'actual_host_wait_exit.json'):value['input_sha256'][str(file)]=sha(file)
 dispatchproof=T/'outputs/native_expert_owner_dispatch_v1/CPU_actual_dispatch_checks.json';dispatch=read(dispatchproof);assert dispatch['passed']and not dispatch['CUDA_initialized'];verify(dispatch);value['input_sha256'][str(dispatchproof)]=sha(dispatchproof)
 dispatchexit=T/'outputs/native_expert_owner_dispatch_v1/CPU_actual_host_exit_v2.json';exitproof=read(dispatchexit);assert exitproof['exit_code']==0 and exitproof['actual_Popen_wait_return']and exitproof['proc_absent_after_wait'];value['input_sha256'][str(dispatchexit)]=sha(dispatchexit)
 for file in(T/'native_expert_owner_dispatch_v1').glob('*'):
  if file.is_file():value['source_sha256'][str(file.relative_to(P))]=sha(file)
 for file in(T/'native_onpolicy_sampler_health_v1').glob('*'):
  if file.is_file():value['source_sha256'][str(file.relative_to(P))]=sha(file)
 for file in(T/'native_onpolicy_recovery_stream_v2').glob('*'):
  if file.is_file():value['source_sha256'][str(file.relative_to(P))]=sha(file)
 value['source_sha256'][str((T/'native_reference_recovery_v1/selector.py').relative_to(P))]=sha(T/'native_reference_recovery_v1/selector.py')
 cacheproof=T/'outputs/native_onpolicy_preparation_v2/CPU_actual_cache_checks.json';cp=read(cacheproof);assert cp['passed']and cp['earliest_positive_and_negative_decisions_preserved'];verify(cp)
 cachefile=T/'outputs/native_onpolicy_preparation_v2/actual_preparation_cache.json';cached=read(cachefile);verify(cached);assert cached['historical_C5_preparation_recoveries_used']==cached['historical_C5_preparation_recoveries_maximum']==2 and cached['new_automatic_recovery_attempts']==0
 value['cached_preparation']=cached;value['initial_screen_state']=initial;value['input_sha256'].update(cached['input_sha256']);value['input_sha256'][str(cacheproof)]=sha(cacheproof);value['input_sha256'][str(cachefile)]=sha(cachefile)
 for file in(T/'native_onpolicy_preparation_v2').glob('*'):
  if file.is_file():value['source_sha256'][str(file.relative_to(P))]=sha(file)
 value['perception_interface']='native_object_association_v1';value['same_fixed20_TRAIN_population_as_qualified_V7']=True
 associationproof=T/'outputs/native_object_association_v1/candidate_checks/result.json';association=read(associationproof);assert association['passed']and association['all20_fixed_TRAIN_scenes_retained']and association['unchanged_branches_bit_exact']and not association['CUDA_initialized'];verify(association)
 associationexit=T/'outputs/native_object_association_v1/CPU_candidate_actual_host_exit.json';actual=read(associationexit);assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
 for f in(associationproof,associationexit):value['input_sha256'][str(f)]=sha(f)
 for f in(T/'native_object_association_v1').glob('*'):
  if f.is_file():value['source_sha256'][str(f.relative_to(P))]=sha(f)
 value['input_sha256'][str(CPUproof)]=sha(CPUproof)
 value['identity']=identity(value);verify(value);write(O/'study.json',value);return value

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');a=parser.parse_args();study=freeze()
 if a.freeze_only:print('NEXT_FIXED_TRAIN_COVERAGE_FROZEN',study['identity'],flush=True);return
 with(O/'campaign.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not list((O/'batches/000/runs').glob('*'))
  gpu,info,placement,fleet=choose();os.environ['CX_BEST100_RUN_ID']=uuid.uuid4().hex;root=batch(0);root.mkdir(parents=True)
  write(root/'cpu_placement_plan.json',placement);write(root/'device_plan.json',dict(gpu=gpu,info=info,fresh_fleet=fleet))
  spec=dict(study,CPU_placement=canonical(root/'cpu_placement_plan.json'),required_gpu_uuid=info['uuid'],study_identity=study['identity']);spec.pop('identity');spec['input_sha256']=dict(study['input_sha256'])
  health=read(O/'selected_sampler_health.json');proof=Path(health['proof']);assert sha(proof)==health['proof_sha256']and read(proof)['passed'];verify(read(proof))
  spec['input_sha256'][str(proof)]=sha(proof)
  for file in(O/'study.json',root/'cpu_placement_plan.json',root/'device_plan.json'):spec['input_sha256'][canonical(file)]=sha(file)
  spec['identity']=identity(spec);write(root/'protocol.json',spec);replace(O/'execution_state.json',dict(root=str(root),unit='cx-tta-onpolicy-coverage-v8.service'))
  allowed=os.sched_getaffinity(0)
  try:apply(placement);execute_batch(root,spec,gpu,info)
  finally:os.sched_setaffinity(0,allowed)
  from .report import summarize
  report=summarize(root,spec);recovered=read(root/'CPU_recovery_stream_audit.json');assert recovered['passed'];report.update(recovery_stream_audit=recovered,coverage_passed=recovered['coverage_passed'],coverage_is_potential_only=False,calibration_training_allowed=False)
  write(root/'atomic_full_recovery_qualification.json',dict(passed=True,parent_terminal_passed=True,recovery_audit_passed=True,no_optimizer=True,input_sha256={str(p):sha(p)for p in(root/'terminal_audit.json',root/'CPU_recovery_stream_audit.json',root/'CPU_full_capture_audit.json')}));write(O/'pilot_complete.json',report);print('NEXT_FIXED_TRAIN_COVERAGE_COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:
  replace(O/'campaign_failure.json',dict(traceback=traceback.format_exc(),automatic_GPU_retry=False));raise
