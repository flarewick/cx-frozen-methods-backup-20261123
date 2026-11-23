"""Four CPU-only containers, same reserved <=16-core set as the GPU host."""
import os,sys,time,json,secrets,subprocess,hashlib,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[2];D=R/'deployment';W=R/'multi_event_gate_v1';OUT=W/'outputs/gate_transfer_profile_v208';sys.path.insert(0,str(D))
from evaluate import CANONICAL_PROJECT,add_env
import user_container as container
class Pool:
 def __init__(self,cores,label):
  assert 4<=len(cores)<=16
  self.cores=cores;self.children=[];self.tag='cx_topp218_'+secrets.token_hex(6);self.folder=OUT/(label+'_'+self.tag);self.folder.mkdir(mode=0o700);self.config_file=self.folder/'config.json'
  canonical=lambda path:str(CANONICAL_PROJECT/path.relative_to(R))
  self.config=dict(schema=1,token=secrets.token_hex(24),workers=[dict(CPU_core=cores[i],socket='/tmp/'+self.tag+'/worker'+str(i)+'.sock',ready=canonical(self.folder/('worker'+str(i)+'_ready.json')),metrics=canonical(self.folder/('worker'+str(i)+'_metrics.json')))for i in range(4)])
  self.config_file.write_text(json.dumps(self.config,indent=2)+'\n');os.chmod(self.config_file,0o600);self.canonical_config=canonical(self.config_file)
  try:
   for i in range(4):
    cmd=container.command(['bash',str(CANONICAL_PROJECT/'deployment/task_adapter_python.sh'),canonical(HERE/'worker218.py'),self.canonical_config,str(i)],mount_project=True)
    log=(self.folder/('worker'+str(i)+'.log')).open('w')
    proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();self.children.append(proc)
   began=time.monotonic()
   while True:
    if any(p.poll()is not None for p in self.children):raise RuntimeError('CPU TOPP worker exited before readiness')
    files=[self.folder/('worker'+str(i)+'_ready.json')for i in range(4)]
    if all(f.exists()for f in files):
     rows=[json.loads(f.read_text())for f in files]
     assert all(r['passed']and r['no_GPU_devices']and r['CPU_core']==cores[i]for i,r in enumerate(rows));break
    if time.monotonic()-began>90:raise TimeoutError('CPU TOPP worker preparation deadline')
    time.sleep(.2)
  except BaseException:self.close();raise
 def healthy(self):return all(p.poll()is None for p in self.children)
 def close(self):
  import gpu_guard_bounded_metrics as guard
  for proc in self.children:
   if proc.poll()is None:guard.stop_job(proc)
  tmp=container.WORK/'run-tmp'/self.tag
  if tmp.is_dir()and tmp.stat().st_uid==os.getuid():shutil.rmtree(tmp)
