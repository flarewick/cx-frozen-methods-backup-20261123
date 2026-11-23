"""Single source-bound bounded software GPU probe; never an evaluation retry."""
from pathlib import Path
import os,sys,json,subprocess,time,uuid,fcntl,traceback
from .common import T,P,C,O,read,write,sha,verify,identity
from native_cross_task100_v5.host import canonical,wait_file,plan,apply,container,gpu_doctor,admission,guard,inventory

def main():
 assert read(O/'CPU_actual_exit.json')['actual_exit_code']==0 and read(O/'CPU_AST_checks.json')['passed']
 fleet=inventory();assert fleet['own_gpu_count']<3
 gpu=next(i for i in(5,3,4,6,0,1,2)if i in fleet['available_indices']);info=gpu_doctor.gpu_snapshot(gpu);admission.require_admission(fleet,info['uuid']);place=plan(gpu)
 cert=read(container.WORK/('gpu-doctor-'+str(gpu)+'.json'));assert cert['uuid']==info['uuid']and cert['build_mtime_ns']==(container.WORK/'stage-bootstrap.ok').stat().st_mtime_ns
 root=O/'runs'/uuid.uuid4().hex;write(root/'cpu_plan.json',place);s=read(T/'outputs/native_cross_task100_v5/study.json');verify(s);s=dict(s,version='native_batched_prefix_graph_v1',scope='bounded_saved_native_batch3_prefix_software_parity_only',CPU_placement=canonical(root/'cpu_plan.json'),GPU_deadline_seconds=240,attempt_nonce=uuid.uuid4().hex)
 s['source_sha256']=dict(s['source_sha256']);s['input_sha256']=dict(s['input_sha256']);s['input_sha256'][canonical(root/'cpu_plan.json')]=sha(root/'cpu_plan.json')
 for f in C.rglob('*'):
  if f.is_file()and f.suffix in('.py','.sh'):s['source_sha256'][str(f.relative_to(P))]=sha(f)
 for n in ['benefit_prefix_graph.py','benefit_prefix_graph_lifetime_v10.py']:
  f=P/'deployment'/n;s['source_sha256'][str(f.relative_to(P))]=sha(f)
 source=T/'outputs/native_cross_task100_v5/batches/002/runs/c818d4d069b94a8f83fdc5e242d0a284';items=[]
 for task in s['tasks']:
  for folder in sorted((source/'cases').glob(task+':*'))[:2]:
   accepted=read(source/'accepted'/(folder.name+'.json'))
   for f in sorted((folder/'M_event_full').glob('observation*.npz'))[:2]:items.append(dict(file=canonical(f),sha256=sha(f),instruction=accepted['instruction']))
 assert len(items)==8;s['probe_inputs']=items;s.pop('identity');s['identity']=identity(s);verify(s);write(root/'protocol.json',s);write(O/'probe_launch.json',dict(root=str(root),gpu=gpu,uuid=info['uuid'],protocol_identity=s['identity']));apply(place)
 cmd=container.command(['bash',canonical(C/'scripts/run.sh'),'--root',canonical(root)],mount_project=True);split=cmd.index('--');cmd[split:split]=gpu_doctor.driver_options(info)
 with(root/'host.log').open('x')as log:
  process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   ready=wait_file(root/'barrier/ready.json',process,240);assert ready['passed']and ready['cpu_fork_child_passed'];started_ns=time.monotonic_ns()
   with(container.WORK/'gpu_locks/account_admission.lock').open('a')as account,(container.WORK/'gpu_locks'/('gpu_'+str(gpu)+'.lock')).open('a')as card:
    fcntl.flock(account,fcntl.LOCK_EX);fcntl.flock(card,fcntl.LOCK_EX);admission.check_admission(gpu_doctor.gpu_snapshot(gpu));released=False;owned=False
    def permit():admission.check_admission(gpu_doctor.gpu_snapshot(gpu));write(root/'barrier/permit.json',dict(gpu_uuid=info['uuid'],nonce=s['attempt_nonce']))
    def sample(row):
     nonlocal released,owned
     if len(row['gpu_pids'])>1:return 'Exactly one GPU owner required'
     if row['gpu_pids']and not released:fcntl.flock(account,fcntl.LOCK_UN);released=True
     if row['gpu_pids']and not owned and(root/'barrier/cuda_ready.json').exists():write(root/'barrier/owned_pid_permit.json',dict(pid=row['gpu_pids'][0]));owned=True
    guard.guarded_command(cmd,info,root/'resources.jsonl',gpu=gpu,deadline=240,on_sample=sample,min_cumulative_utilization=34,max_memory_mib=24563,prepared_process=process,before_monitor=permit,child_exit_proof=root/'barrier/normal_child_exit.json',child_exit_identity=dict(attempt_nonce=s['attempt_nonce'],gpu_uuid=info['uuid'],protocol_identity=s['identity'],cpu_only_fixture=False,attempt_started_ns=started_ns))
   assert process.wait()==0 and read(root/'CPU_parent_result.json')['passed'];rows=[json.loads(x)for x in(root/'resources.jsonl').read_text().splitlines()];assert not any('stopped'in x for x in rows)and rows[-1]['own_gpu_released'];verify(s)
   write(root/'actual_terminal.json',dict(passed=True,actual_Popen_wait_code=0,actual_child_exit_code=0,GPU_released=True,software_probe_only=True,sustained_qualification=False,result=read(root/'GPU_probe_result.json')));print('GRAPH_PROBE_COMPLETE',root,flush=True)
  except BaseException:
   write(root/'failure.json',dict(traceback=traceback.format_exc()))
   if process.poll()is None:guard.stop_job(process)
   raise
if __name__=='__main__':main()
