"""Whole-cache GPU sampler proof under unchanged strict resource thresholds."""
import argparse,fcntl,json,os
from pathlib import Path
import subprocess,sys,time,uuid
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1]
sys.path.insert(0,str(P/'deployment'));sys.path.insert(0,str(T/'resource_guard_v1'))
import user_container as container
import gpu_doctor
import strict_guard as guard
def read(p):return json.loads(Path(p).read_text())
def canonical(p):return str(container.CANONICAL/'VLA-Adapter/cx_new_direction_guide_TTA'/p.relative_to(P))
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--run',action='store_true');a=p.parse_args()
 for name in ('guard_cpu_checks.json','exit_cpu_checks.json','guard_source_audit.json','sampler_cpu_checks.json','current_sampler_cpu_checks.json','current_query_cpu_checks.json'):assert read(T/'outputs/native_nextchunk_v1'/name)['passed']
 output=T/'outputs/native_nextchunk_v1/sampler_gpu_v1'
 if (output/'latest_complete.json').exists():
  prior=read(output/'latest_complete.json')
  assert read(Path(prior['root'])/'report.json')['complete']
  print('ALREADY_COMPLETE_NO_GPU_RETRY',json.dumps(prior),flush=True);return
 info=gpu_doctor.gpu_snapshot(a.gpu);gpu_doctor.check_admission(info)
 cert=read(container.WORK/('gpu-doctor-'+str(a.gpu)+'.json'));assert cert['uuid']==info['uuid'] and cert['build_mtime_ns']==(container.WORK/'stage-bootstrap.ok').stat().st_mtime_ns
 if not a.run:
  print('SAMPLER_GPU_PLAN_ONLY',json.dumps(dict(gpu=info,cached_queries=149,new_scenes=0,full_task_evaluation=False,guard='40%/300s/4GB/CPU/RSS/max3 unchanged')),flush=True);return
 locks=container.WORK/'gpu_locks';locks.mkdir(exist_ok=True)
 with (locks/'account_admission.lock').open('a') as admission,(locks/('gpu_'+str(a.gpu)+'.lock')).open('a') as card:
  fcntl.flock(admission,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(card,fcntl.LOCK_EX|fcntl.LOCK_NB)
  gpu_doctor.check_admission(gpu_doctor.gpu_snapshot(a.gpu))
  output=T/'outputs/native_nextchunk_v1/sampler_gpu_v1'
  if (output/'resource_blocked.json').exists():raise ValueError('No unchanged GPU retry after resource failure')
  nonce=uuid.uuid4().hex;root=output/'attempts'/nonce;root.mkdir(parents=True);barrier=root/'barrier'
  cmd=container.command(['bash',canonical(C/'sampler_gpu.sh'),'--barrier',canonical(barrier),'--root',canonical(root),'--gpu-uuid',info['uuid'],'--nonce',nonce],mount_project=True)
  split=cmd.index('--');cmd[split:split]=gpu_doctor.driver_options(info)
  os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-16:])
  log=(root/'console.log').open('x');process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  prep=time.monotonic();attempt_start_ns=time.monotonic_ns()
  try:
   from gpu_policy import inventory,require_admission
   while not (barrier/'ready.json').exists():
    if process.poll()is not None:raise ValueError('CPU preloading/proof failed before GPU permit')
    require_admission(inventory(),info['uuid'])
    if time.monotonic()-prep>240:raise TimeoutError('CPU preparation deadline')
    time.sleep(.5)
   ready=read(barrier/'ready.json');spec=read(output/'protocol.json')
   assert ready['one_thread'] and ready['CUDA_uninitialized'] and ready['CPU_fork_child_passed']
   assert ready['protocol_identity']==spec['identity'] and ready['source_sha256']==spec['source_sha256']
   write=lambda path,x:path.write_text(json.dumps(x,indent=2)+'\n')
   write(root/'launch.json',dict(gpu=info,protocol_identity=spec['identity'],attempt_nonce=nonce,CPU_preparation=ready))
   def permit():
    gpu_doctor.check_admission(gpu_doctor.gpu_snapshot(a.gpu));write(barrier/'permit.json',dict(gpu_uuid=info['uuid'],attempt_nonce=nonce))
   released=False;owned_permitted=False
   def sample(row):
    nonlocal released,owned_permitted
    if len(row['gpu_pids'])>1:return 'Exactly one GPU worker required'
    if row['gpu_pids'] and not released:fcntl.flock(admission,fcntl.LOCK_UN);released=True
    if row['gpu_pids'] and (barrier/'cuda_ready.json').exists() and not owned_permitted:
     write(barrier/'owned_pid_permit.json',dict(gpu_uuid=info['uuid'],verified_host_GPU_PID=row['gpu_pids'][0]));owned_permitted=True
   guard.guarded_command(cmd,info,root/'resources.jsonl',gpu=a.gpu,deadline=7200,on_sample=sample,min_cumulative_utilization=40,max_memory_mib=22528,
       prepared_process=process,before_monitor=permit,child_exit_proof=barrier/'normal_child_exit.json',
       child_exit_identity=dict(attempt_nonce=nonce,gpu_uuid=info['uuid'],protocol_identity=spec['identity'],cpu_only_fixture=False,attempt_started_ns=attempt_start_ns))
   assert read(root/'report.json')['complete']
   write(output/'latest_complete.json',dict(root=str(root),gpu_uuid=info['uuid'],protocol_identity=spec['identity'],host_exit_code=0,
     source_policy_inputs='cached_current_RGB',simulator_started=False,TTA_utility_proven=False))
   print('REAL_VLA_SAMPLER_GPU_HOST_COMPLETE',str(root),flush=True)
  except BaseException as exc:
   (root/'host_failure.json').write_text(json.dumps(dict(reason=str(exc),type=type(exc).__name__,automatic_retry=False),indent=2)+'\n')
   if any(term in str(exc).lower() for term in ('utilization','vram','memory','rss')):(output/'resource_blocked.json').write_text(json.dumps(dict(reason=str(exc),attempt=str(root),automatic_retry=False))+'\n')
   if process.poll()is None:
    known=guard.stop_job(process);guard.record_release(process,info,prep,root/'failed_release.jsonl',known)
   raise
  finally:log.close()
if __name__=='__main__':main()
