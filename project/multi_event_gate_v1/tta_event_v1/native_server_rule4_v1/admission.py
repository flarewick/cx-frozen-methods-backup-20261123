from pathlib import Path
import subprocess,sys,os,json

def decision(fleet,selected_uuid):
 if not any(c['uuid']==selected_uuid for c in fleet['cards']):return False,'Selected GPU is absent'
 if fleet['own_gpu_count']>=3:return False,'Account already uses three GPUs'
 for c in fleet['cards']:
  if c['uuid']==selected_uuid:
   if c['pids']:return False,'Selected GPU is occupied'
   continue
  if not c['own_pids']:continue
  memory=c['own_fraction']is not None and c['own_fraction']>=.5
  utilization=c['attributable_utilization']is not None and c['attributable_utilization']>=75
  if not(memory or utilization):return False,'Server rule4: existing own GPU'+str(c['index'])+' below75% utilization and50% own memory'
 return True,'Every prior own GPU satisfies actual server rule4'

def snapshot():
 p=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('worker.py'))],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,check=True,timeout=3)
 return json.loads(p.stdout)

def require(uuid):
 fleet=snapshot();okay,reason=decision(fleet,uuid)
 if not okay:raise RuntimeError(reason)
 return dict(passed=True,reason=reason,fleet=fleet,no_CUDA_or_GPU_work=True)
