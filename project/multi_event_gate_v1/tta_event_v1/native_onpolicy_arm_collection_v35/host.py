from pathlib import Path

import os,sys,subprocess,time,uuid,fcntl,traceback,json,argparse

from .common import C,T,P,O,TASKS,COHORT,CANON,read,write,replace,sha,identity,mapped,verify,batch

sys.path.insert(0,str(P/'deployment'));sys.path.insert(0,str(T/'resource_guard_v1'));sys.path.insert(0,str(T/'native_nextchunk_v1'))
import user_container as container

import gpu_doctor

from .cpu_admission import CPUAdmissionRace,occupied_before_permit,assert_replan
from native_account_stream_v1 import admission
from native_nvml_query_order_v3.context import context as ordered_NVML_context
from native_server_rule4_v11.admission import require as require_rule4,ongoing as ongoing_rule4,close as close_rule4,prime as prime_rule4

from native_governor_margin_v2 import margin as guard
from native_process_missing_guard_v2.empty import eligible_row as idle_sample_eligible
from native_onpolicy_sampler_health_v1.health import check as check_sampler_health

from native_account_stream_v1.transport import inventory,context as account_context

from native_visible_expansion_v1.placement import plan,apply

def canonical(p):return str(CANON/Path(p).resolve().relative_to(P))

def script(mode,module,*args):return container.command(['env','CX_OBJECT_STAGE=development','CX_EXPERT_STAGE='+COHORT,'CX_ONPOLICY_STAGE='+COHORT,'bash',canonical(C/'scripts'/('cpu.sh'if mode=='cpu'else'gpu.sh')),*(['-m',module]if mode=='cpu'else[]),*args],mount_project=True)

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
   for gpu in (1,6,5,4,2,3):
    if gpu not in fleet['available_indices']:continue
    info=gpu_doctor.gpu_snapshot(gpu);cert=container.WORK/('gpu-doctor-'+str(gpu)+'.json')
    if not cert.exists():continue
    d=read(cert)
    if d['uuid']!=info['uuid']or d['build_mtime_ns']!=(container.WORK/'stage-bootstrap.ok').stat().st_mtime_ns:continue
    admission.require_admission(fleet,info['uuid'])
    # CPU-only preparation may overlap resource observation; original rule4 stays mandatory at both permit checks.
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
   k=str(pid);assert (r['process_utilization'][k]is not None or idle_sample_eligible(r,k)) and min(r['process_mean_utilization'][k],r['process_rolling_300s_utilization'][k])>=34 and r['process_memory_mib'][k]>=4096 and 40<=r['cpu'][k]<=1600 and r['rss_bytes'][k]<=128*2**30
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
  semcmd=script('cpu','native_onpolicy_arm_collection_v35.semantic','--batch',canonical(root));sem=subprocess.Popen(semcmd,stdout=semlog,stderr=subprocess.STDOUT,start_new_session=True)
  write(root/'semantic_launch.json',dict(pid=sem.pid,started_ticks=int(Path('/proc/%d/stat'%sem.pid).read_text().split(') ',1)[1].split()[19]),command=semcmd));ready=wait_file(root/'semantic/ready.json',sem,240);assert ready['passed']and not ready['CUDA_initialized']and ready['source_identity']==spec['identity']
  for row in(spec.get('cached_preparation')or{}).get('requests',[]):write(root/'semantic/requests'/(row['content_key']+'.json'),row)
  for row in sorted((spec.get('cached_preparation')or{}).get('requests',[]),key=lambda row:row['content_key']):wait_file(root/'semantic/answers'/(row['content_key']+'.json'),sem,240)
  attempt=root/'attempt';attempt.mkdir();barrier=attempt/'barrier';nonce=uuid.uuid4().hex
  cmd=script('gpu',None,'--batch',canonical(root),'--barrier',canonical(barrier),'--nonce',nonce);split=cmd.index('--');cmd[split:split]=gpu_doctor.driver_options(info)
  process=subprocess.Popen(cmd,stdout=glog,stderr=subprocess.STDOUT,start_new_session=True);ticks=int(Path('/proc/%d/stat'%process.pid).read_text().split(') ',1)[1].split()[19]);write(attempt/'launch.json',dict(pid=process.pid,started_ticks=ticks,command=cmd,gpu=info,nonce=nonce))
  proof=wait_file(barrier/'ready.json',process,240);assert proof['passed']and proof['cpu_fork_child_passed']and proof['os_threads']==1 and not proof['cuda_initialized']and proof['source_sha256']==spec['source_sha256'];write(root/'source_bound_CPU_preload_proof.json',proof)
  write(root/'CPU_preparation_ready.json',dict(cached_initials_recomputed_before_GPU=len((spec.get('cached_preparation')or{}).get('requests',[])),CUDA_permit_not_written=not(barrier/'permit.json').exists()))
  lockdir=container.WORK/'gpu_locks';lockdir.mkdir(exist_ok=True);waiting=time.monotonic()
  from native_admission_commit_margin_v1.margin import require as require_commit_margin
  with(lockdir/'account_admission.lock').open('a')as account,(lockdir/('gpu_'+str(gpu)+'.lock')).open('a')as card:
   while True:
    try:
     fcntl.flock(account,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(card,fcntl.LOCK_EX|fcntl.LOCK_NB);first_admission=require_rule4(info['uuid']);require_commit_margin(first_admission,info['uuid']);admission.check_admission(gpu_doctor.gpu_snapshot(gpu));break
    except(BlockingIOError,RuntimeError):
     fcntl.flock(account,fcntl.LOCK_UN);fcntl.flock(card,fcntl.LOCK_UN)
     fleet=inventory()
     if next(card for card in fleet['cards']if card['index']==gpu)['pids']:occupied_before_permit(root,gpu,fleet)
     from native_retained_cpu_wait_v1.wait import observe as observe_CPU_wait
     observe_CPU_wait(root,gpu,fleet,process)
     replace(O/'live_status.json',dict(phase='CPU_only_retained_preload_wait_for_selected_GPU_or_rules',gpu=gpu,fleet=fleet))
     if time.monotonic()-waiting>1500:
      occupied_before_permit(root,gpu,fleet)
      raise TimeoutError('CPU admission wait expired without GPU')
     time.sleep(2)
   began=time.monotonic();started_ns=time.monotonic_ns();released=False;permitted=False
   def permit():
    rule4_proof=require_rule4(info['uuid']);write(root/'attempt/rule4_admission.json',rule4_proof);admission.check_admission(gpu_doctor.gpu_snapshot(gpu));write(barrier/'permit.json',dict(gpu_uuid=info['uuid'],attempt_nonce=nonce,protocol_identity=spec['identity']))
   def sample(row):
    nonlocal released,permitted
    if len(row['gpu_pids'])>1:return 'Exactly one GPU owner required'
    from native_foreign_gpu_contention_v1.guard import reason as foreign_contention
    collision=foreign_contention(row)
    if collision:return collision
    if row['gpu_pids']:
     try:rule4_proof=ongoing_rule4(info['uuid'])
     except RuntimeError as error:return 'Conservative ongoing server rule4 cannot be verified: '+str(error)
     with(root/'attempt/rule4_running.jsonl').open('a')as f:f.write(json.dumps(dict(seconds=row['seconds'],proof=rule4_proof))+'\n')
    if row['gpu_pids']and not released:fcntl.flock(account,fcntl.LOCK_UN);released=True
    if row['gpu_pids']and not permitted and(barrier/'cuda_ready.json').exists():write(barrier/'owned_pid_permit.json',dict(gpu_uuid=info['uuid'],host_GPU_PID=row['gpu_pids'][0]));permitted=True
    replace(O/'live_status.json',dict(phase='paired_GPU_evaluation',batch=spec['batch_index'],gpu=gpu,protocol_identity=spec['identity'],resources=row))
   guard.guarded_command(cmd,info,attempt/'resources.jsonl',gpu=gpu,deadline=min(7200,spec['stage_total_GPU_budget_seconds']-spec['inherited_stage_GPU_seconds']),on_sample=sample,min_cumulative_utilization=34,max_memory_mib=24563,prepared_process=process,before_monitor=permit,child_exit_proof=barrier/'normal_child_exit.json',child_exit_identity=dict(attempt_nonce=nonce,gpu_uuid=info['uuid'],protocol_identity=spec['identity'],cpu_only_fixture=False,attempt_started_ns=started_ns))
  code=process.wait();assert code==0;write(root/'actual_GPU_parent_exit.json',dict(exit_code=code,actual_Popen_wait_return=True,pid=process.pid,started_ticks=ticks))
  from native_process_sampler_v13.metrics import process_utilization,diagnostics
  process_utilization(gpu);write(root/'post_release_device_samples.json',dict(collected_unix_us=time.time_ns()//1000,raw_NVML=diagnostics()['raw_NVML'],actual_GPU_parent_exit0_and_release_already_verified=True,no_new_GPU_work=True))
  write(root/'semantic/stop.json',dict(GPU_parent_actual_exit0=True));semcode=sem.wait(timeout=90);write(root/'semantic_exit.json',dict(exit_code=semcode,actual_Popen_wait_return=True,pid=sem.pid));assert semcode==0
  from native_process_missing_guard_v2.audit import audit as audit_idle_lower_bound
  audit_idle_lower_bound(root,spec)
  from native_semantic_host_audit_v1.audit import audit as audit_CPU_thread_budget
  audit_CPU_thread_budget(root)
  from native_governor_margin_v2.audit import audit as audit_governor_margin
  audit_governor_margin(root,spec)
  terminal(root,spec)
 except BaseException:
  write(root/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
  from native_process_sampler_v13.metrics import diagnostics as last_actual_metrics
  diagnostic=last_actual_metrics()
  if diagnostic:
   write(root/'last_successfully_collected_metric_at_failure.json',dict(metric=diagnostic,may_precede_exception=True,no_new_NVML_read_performed=True,current_attempt_permit_exists=(root/'attempt/barrier/permit.json').exists(),metric_unix_ns=diagnostic.get('unix_ns')))
  if process is not None and process.poll()is None:
   known=guard.stop_job(process)
   if (root/'attempt/barrier/permit.json').exists():guard.record_release(process,info,start,root/'failed_release.jsonl',known)
   else:
    from .cpu_cleanup import record as record_CPU_cleanup
    record_CPU_cleanup(process,root,known)
  if sem is not None and sem.poll()is None:
   if not(root/'semantic/stop.json').exists():write(root/'semantic/stop.json',dict(reason='GPU_or_CPU_attempt_failed_no_retry'))
   try:code=sem.wait(timeout=30)
   except subprocess.TimeoutExpired:guard.stop_job(sem);code=sem.returncode
   if not(root/'semantic_exit.json').exists():write(root/'semantic_exit.json',dict(exit_code=code,actual_Popen_wait_return=True,pid=sem.pid))
  if isinstance(sys.exception(),CPUAdmissionRace):
   assert process is not None and process.poll()is not None
   write(root/'CPU_admission_prepared_exit.json',dict(exit_code=process.returncode,actual_Popen_wait_return=True,pid=process.pid,proc_absent_after_wait=not Path('/proc',str(process.pid)).exists()))
  raise
 finally:semlog.close();glog.close()

def freeze():
 from .collection_plan import freeze_stage
 return freeze_stage()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only', action='store_true')
    a = parser.parse_args()
    study = freeze()
    if a.freeze_only:
        print('NEXT_FIXED_TRAIN_COVERAGE_FROZEN', study['identity'], flush=True)
        return
    with (O / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not list((O / 'batches/000/runs').glob('*'))
        rule4_root = prime_rule4()
        write(O / 'rule4_monitor_identity.json', dict(root=str(rule4_root)))
        for CPU_admission_attempt in range(3):
            try:
                gpu, info, placement, fleet = choose()
                os.environ['CX_BEST100_RUN_ID'] = uuid.uuid4().hex
                root = batch(0)
                root.mkdir(parents=True)
                write(root / 'cpu_placement_plan.json', placement)
                write(root / 'device_plan.json', dict(gpu=gpu, info=info, fresh_fleet=fleet))
                spec = dict(study, CPU_placement=canonical(root / 'cpu_placement_plan.json'), required_gpu_uuid=info['uuid'], study_identity=study['identity'])
                spec.pop('identity')
                spec['input_sha256'] = dict(study['input_sha256'])
                health = read(O / 'selected_sampler_health.json')
                proof = Path(health['proof'])
                assert sha(proof) == health['proof_sha256'] and read(proof)['passed']
                verify(read(proof))
                spec['input_sha256'][str(proof)] = sha(proof)
                for file in (O / 'study.json', root / 'cpu_placement_plan.json', root / 'device_plan.json'):
                    spec['input_sha256'][canonical(file)] = sha(file)
                spec['identity'] = identity(spec)
                write(root / 'protocol.json', spec)
                replace(O / 'execution_state.json', dict(root=str(root), unit='cx-tta-onpolicy-collection-' + COHORT + '.service'))
                allowed = os.sched_getaffinity(0)
                try:
                    apply(placement)
                    execute_batch(root, spec, gpu, info)
                finally:
                    os.sched_setaffinity(0, allowed)
            except CPUAdmissionRace:
                assert_replan(root)
                continue
            break
        else:
            raise RuntimeError('Three CPU-only card races; no GPU experiment retry')
        metadata = list((root / 'attempt').glob('NVML_metadata_*_actual_exit.json'))
        assert len(metadata) >= 2
        for file in metadata:
            item = read(file)
            assert item['exit_code'] == 0 and item['actual_wait_return'] and item['proc_absent_after_wait'] and (not item['forced']) and (not item['failed_request']) and item['no_CUDA_or_GPU_work'] and (item['request_count'] > 0)
        write(root / 'CPU_metadata_transport_terminal.json', dict(passed=True, input_sha256={str(f): sha(f) for f in metadata}, all_queries_original_three_seconds=True, no_CUDA_or_GPU_work=True))
        admission_proof = root / 'attempt/rule4_admission.json'
        running_proof = root / 'attempt/rule4_running.jsonl'
        assert read(admission_proof)['passed']
        rule4_rows = [json.loads(line) for line in running_proof.read_text().splitlines()]
        assert rule4_rows and all((r['proof']['passed'] and r['proof']['no_CUDA_or_GPU_work'] for r in rule4_rows))
        from native_server_rule4_v11.audit import audit as audit_admission_scope
        audit_admission_scope(root)
        from native_retained_cpu_wait_v1.wait import audit as audit_CPU_wait
        audit_CPU_wait(root)
        write(root / 'CPU_cautious_rule4_terminal.json', dict(passed=True, actual_active_samples=len(rule4_rows), input_sha256={str(f): sha(f) for f in (admission_proof, running_proof,root/'CPU_rule4_admission_scope_audit.json')}, original_scientific_resource_thresholds_unchanged=True))
        from native_unread_capture_full_v1.audit import audit as audit_unread_capture
        audit_unread_capture(root,spec)
        from native_storage_capture_v1.audit import audit as audit_storage
        audit_storage(root,spec)
        from native_carried_next_FIT_v1.carry import audit as audit_carried_next
        audit_carried_next(root,spec)
        from .report import summarize
        report = summarize(root, spec)
        recovered = read(root / 'CPU_recovery_stream_audit.json')
        assert recovered['passed']
        report.update(recovery_stream_audit=recovered, coverage_passed=recovered['coverage_passed'], coverage_is_potential_only=False, calibration_training_allowed=False)
        write(root / 'atomic_full_recovery_qualification.json', dict(passed=True, parent_terminal_passed=True, recovery_audit_passed=True, no_optimizer=True, input_sha256={str(p): sha(p) for p in (root / 'terminal_audit.json', root / 'CPU_recovery_stream_audit.json', root / 'CPU_full_capture_audit.json', root / 'CPU_metadata_transport_terminal.json', root / 'CPU_cautious_rule4_terminal.json',root/'CPU_retained_preload_wait_audit.json',root/'CPU_unread_capture_whole_reference_audit.json',root/'CPU_storage_whole25_audit.json',*([root/'CPU_carried_FIT0_inputs_audit.json']if spec.get('carried_FIT0_input_proof')else[]))}))
        write(O / 'pilot_complete.json', report)
        print('NEXT_FIXED_TRAIN_COVERAGE_COMPLETE', json.dumps(report), flush=True)

if __name__=='__main__':
 try:
  with account_context(O/'account_admission_actual_exit.json'),ordered_NVML_context():
   main()
  close_rule4()
  if '--freeze-only'not in sys.argv:
   from native_account_stream_v1.audit import audit as audit_account
   root=Path(read(O/'execution_state.json')['root']);rule4_root=Path(read(O/'rule4_monitor_identity.json')['root'])
   audit_account(root,[O/'account_admission_actual_exit.json',root/'attempt/NVML_account_monitor_actual_exit.json',rule4_root/'NVML_account_rule4_actual_exit.json'])
 except BaseException:
  replace(O/'campaign_failure.json',dict(traceback=traceback.format_exc(),automatic_GPU_retry=False));raise
 finally:close_rule4()
