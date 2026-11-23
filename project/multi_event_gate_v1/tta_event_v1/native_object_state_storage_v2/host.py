"""Bounded software probe using the actual current admission and resource guards."""
from pathlib import Path
import os,sys,time,uuid,fcntl,subprocess,traceback,json,types
from .common import T,P,C,O,read,write,replace,sha,mapped,verify,identity
from native_onpolicy_arm_collection_v29 import host as original
# The actual immutable choose bytecode and guard functions are retained. Only
# its output namespace changes, so no frozen ARM29 output is written here.
choose=types.FunctionType(original.choose.__code__,dict(original.__dict__,O=O,replace=replace),'choose')

def main():
 proof=O/'CPU_contract_checks_v2.json';v=read(proof);verify(v);assert v['passed']
 actual=read(O/'CPU_actual_host_exit_v2.json');assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
 assert not(O/'launch.json').exists(),'No unchanged retry of a failed actual probe'
 manifest=O/'input_manifest.json';data=read(manifest);verify(data);assert data['passed']and len(data['cases'])==6
 carry=T/'outputs/native_onpolicy_full_ops_v2/ARM29_actual_first280_failure_v1.json';previous=read(carry);verify(previous)
 authorization=T/'outputs/native_onpolicy_full_ops_v2/ARM29_VAL_budget_authorization_v1.json';approved=read(authorization);verify(approved);assert approved['approved_total_seconds']==18000
 original.prime_rule4();gpu,info,placement,fleet=choose();root=O/'runs'/uuid.uuid4().hex
 write(root/'cpu_plan.json',placement)
 health=read(O/'selected_sampler_health.json');health_file=Path(health['proof']);assert sha(health_file)==health['proof_sha256'];verify(read(health_file))
 spec=dict(version=C.name,engineering_only=True,scientific_qualification_forbidden=True,
  required_gpu_uuid=info['uuid'],CPU_placement=original.canonical(root/'cpu_plan.json'),attempt_nonce=uuid.uuid4().hex,
  probe_cases=data['cases'],stage_total_GPU_budget_seconds=18000,inherited_stage_GPU_seconds=previous['cumulative_VAL_seconds'],
  source_sha256=dict(read(T/'outputs/native_onpolicy_arm_collection_v29/val_000/study.json')['source_sha256']),
  input_sha256={str(p):sha(p)for p in [proof,O/'CPU_actual_host_exit_v2.json',manifest,carry,authorization,root/'cpu_plan.json',health_file]})
 spec['source_sha256'].update({str(p):sha(p)for p in C.rglob('*')if p.is_file()and p.suffix in('.py','.sh','.md')})
 for p in [T/'outputs/video_instances_v1/models/sam2.1_hiera_tiny.pt',T/'outputs/video_instances_v1/models/verified_manifest.json']:
  spec['input_sha256'][str(p)]=sha(p)
 spec['identity']=identity(spec);verify(spec);write(root/'protocol.json',spec)
 write(O/'launch.json',dict(root=str(root),gpu=gpu,uuid=info['uuid'],protocol_identity=spec['identity'],maximum_GPU_seconds=200))
 original.apply(placement);cmd=original.container.command(['bash',original.canonical(C/'scripts/gpu.sh'),'--root',original.canonical(root)],mount_project=True)
 split=cmd.index('--');cmd[split:split]=original.gpu_doctor.driver_options(info)
 process=None;code=None;started=None;success=False
 try:
  with(root/'host.log').open('x')as log:
   process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   ready=original.wait_file(root/'barrier/ready.json',process,240)
   assert ready['passed']and ready['cpu_fork_child_passed']and ready['os_threads']==1 and not ready['CUDA_initialized']
   assert ready['source_sha256']==spec['source_sha256']
   lockdir=original.container.WORK/'gpu_locks'
   with(lockdir/'account_admission.lock').open('a')as account,(lockdir/('gpu_'+str(gpu)+'.lock')).open('a')as card:
    waiting=time.monotonic()
    while True:
     try:
      fcntl.flock(account,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(card,fcntl.LOCK_EX|fcntl.LOCK_NB)
      from native_admission_commit_margin_v1.margin import require as require_commit_margin
      first=original.require_rule4(info['uuid']);require_commit_margin(first,info['uuid']);original.admission.check_admission(original.gpu_doctor.gpu_snapshot(gpu));break
     except (RuntimeError,BlockingIOError):
      fcntl.flock(account,fcntl.LOCK_UN);fcntl.flock(card,fcntl.LOCK_UN)
      fleet=original.inventory()
      assert not next(c for c in fleet['cards']if c['index']==gpu)['pids'],'Selected GPU occupied before CUDA; CPU-only refusal'
      assert time.monotonic()-waiting<1500,'Original CPU-only admission wait expired'
      assert process.poll() is None,'CPU prepared parent exited while waiting'
      replace(O/'live_status.json',dict(root=str(root),gpu=gpu,phase='CPU_only_wait_for_original_admission',no_CUDA_permit=True))
      time.sleep(2)
    released=False;owned=False;started=time.monotonic_ns()
    def permit():
     rule=original.require_rule4(info['uuid']);write(root/'rule4_admission.json',rule)
     original.admission.check_admission(original.gpu_doctor.gpu_snapshot(gpu))
     write(root/'barrier/permit.json',dict(gpu_uuid=info['uuid'],attempt_nonce=spec['attempt_nonce'],protocol_identity=spec['identity']))
    def sample(row):
     nonlocal released,owned
     if len(row['gpu_pids'])>1:return 'Exactly one GPU owner required'
     from native_foreign_gpu_contention_v1.guard import reason
     collision=reason(row)
     if collision:return collision
     if row['gpu_pids']:
      try:rule=original.ongoing_rule4(info['uuid'])
      except RuntimeError as error:return 'Original ongoing rule4 refused: '+str(error)
      with(root/'rule4_running.jsonl').open('a')as stream:stream.write(json.dumps(dict(seconds=row['seconds'],proof=rule))+'\n')
     if row['gpu_pids']and not released:fcntl.flock(account,fcntl.LOCK_UN);released=True
     if row['gpu_pids']and not owned and(root/'barrier/cuda_ready.json').exists():
      write(root/'barrier/owned_pid_permit.json',dict(gpu_uuid=info['uuid'],host_GPU_PID=row['gpu_pids'][0]));owned=True
     replace(O/'live_status.json',dict(root=str(root),gpu=gpu,phase='bounded_SAM2_storage_parity',resources=row))
    original.guard.guarded_command(cmd,info,root/'resources.jsonl',gpu=gpu,deadline=min(200,18000-previous['cumulative_VAL_seconds']),on_sample=sample,
     min_cumulative_utilization=34,max_memory_mib=24563,prepared_process=process,before_monitor=permit,
     child_exit_proof=root/'barrier/normal_child_exit.json',child_exit_identity=dict(attempt_nonce=spec['attempt_nonce'],gpu_uuid=info['uuid'],protocol_identity=spec['identity'],cpu_only_fixture=False,attempt_started_ns=started))
   code=process.wait();assert code==0 and read(root/'CPU_parent_result.json')['passed']
   assert read(root/'barrier/normal_child_exit.json')['exit_code']==0 and read(root/'GPU_probe_result.json')['passed'];verify(spec);success=True
 except BaseException:
  write(root/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
  if process is not None and process.poll()is None:original.guard.stop_job(process)
  raise
 finally:
  if process is not None:
   code=process.wait();write(root/'actual_host_exit.json',dict(exit_code=code,actual_Popen_wait_return=True,pid=process.pid,proc_absent_after_wait=not Path('/proc',str(process.pid)).exists()))
  resource=root/'resources.jsonl';rows=[json.loads(line)for line in resource.read_text().splitlines()]if resource.exists()else[]
  if rows:assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids']
  seconds=rows[-1]['cleanup_seconds_after_start']if rows else 0
  bindings=[root/name for name in ['protocol.json','resources.jsonl','actual_host_exit.json','barrier/normal_child_exit.json','GPU_probe_result.json','failure.json']if(root/name).exists()]
  write(O/'actual_report.json',dict(passed=success,software_only=True,sustained_resource_qualification=False,
   root=str(root),actual_host_exit=code,actual_GPU_seconds=seconds,own_GPU_released=bool(rows and rows[-1]['own_gpu_released']),
   cumulative_VAL_seconds=previous['cumulative_VAL_seconds']+seconds,approved_total_seconds=18000,remaining_seconds=18000-previous['cumulative_VAL_seconds']-seconds,
   input_sha256={str(p):sha(p)for p in bindings}))
if __name__=='__main__':
 try:
  with original.account_context(O/'account_admission_actual_exit.json'),original.ordered_NVML_context():main()
 finally:original.close_rule4()
