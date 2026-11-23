"""Run the two frozen official groups sequentially; no failed-GPU retries."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time
T=Path(__file__).resolve().parents[1];P=T.parents[1];O=T/'outputs/native_seven_campaign_v8'

def write(p,v):
 assert p.resolve().is_relative_to(O.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def state(v):
 p=O/('live_status.'+str(os.getpid())+'.tmp');p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(p,O/'live_status.json')
def main():
 from native_governor180_v5.observer import latest
 from native_seven_official_v9.common import read,sha,verify
 from native_seven_official_v9.configuration import GROUPS
 from native_seven_official_v9.adoption import IMPORTED_PLAIN_SESSIONS
 O.mkdir(parents=True,exist_ok=True)
 with(O/'campaign.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  write(O/'launch.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),groups=['plain','gated'],sequential_groups=True,no_GPU_work_in_this_parent=True,frozen_backend='native_seven_official_v9',source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')}))
  inputs={};groups={};began=time.monotonic()
  for group in('plain','gated'):
   first=IMPORTED_PLAIN_SESSIONS if group=='plain'else 0
   spec=read(T/'outputs/native_seven_official_v9'/(group+'_'+str(first).zfill(3))/'study.json');verify(spec);assert spec['group']==group and spec['official_seed']==0 and spec['actual_seed_start']==100000
   while True:
    v=latest();state(dict(phase='CPU_only_wait_for_empty_GPU',next_group=group,available_GPUs=v['available_indices'],account_GPU_count=v['own_gpu_count'],elapsed_seconds=time.monotonic()-began,no_GPU_work_in_parent=True))
    if v['available_indices']and v['own_gpu_count']<3:break
    if time.monotonic()-began>34*3600:raise TimeoutError('CPU-only wait reached bounded34h; no GPU retry')
    time.sleep(15)
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(P/'deployment')+':'+str(T))
   command=[sys.executable,'-B','-u','-m','native_seven_official_v9.supervision','--group',group]
   state(dict(phase='official_group_supervision',group=group,command=command,GPU_actual_state_requires_stage_live_status=True))
   with(O/(group+'_host.log')).open('x')as f:
    p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env);code=p.wait()
   actual=O/(group+'_actual_wait_exit.json');write(actual,dict(exit_code=code,actual_Popen_wait_return=True,pid=p.pid,proc_absent_after_wait=not Path('/proc',str(p.pid)).exists()));assert code==0,('No automatic failed GPU retry',group,code)
   final=T/'outputs/native_seven_official_v9'/('supervision_'+group)/'official100_complete.json';r=read(final);verify(r);assert r['complete']and r['scenes_per_task']==100 and set(r['groups'])==set(GROUPS[group]['tasks'])
   groups.update(r['groups']);inputs[str(final)]=sha(final);inputs[str(actual)]=sha(actual)
  assert len(groups)==7 and all(v['scenes']==100 for v in groups.values())
  value=dict(complete=True,official_seed=0,groups=groups,TTA_episodes=700,paired_control_episodes=700,failed_partial_excluded=True,all_GPU_release_and_native_resource_controls_qualified=True,input_sha256=inputs)
  write(O/'all_seven_official100_complete.json',value);state(dict(phase='complete',groups=groups,all_GPU_released=True));print('ALL_SEVEN_OFFICIAL100_COMPLETE',json.dumps(groups),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  if O.exists()and not(O/'failure.json').exists():write(O/'failure.json',dict(traceback=traceback.format_exc(),automatic_GPU_retry=False))
  raise
