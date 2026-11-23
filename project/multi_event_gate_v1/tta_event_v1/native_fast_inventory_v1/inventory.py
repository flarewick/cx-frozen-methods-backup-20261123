"""Same inventory schema, fresh complete NVML read, unchanged three-second bound."""
import json,os,subprocess,sys
from pathlib import Path
T=Path(__file__).resolve().parents[1]
def validate(fleet):
 cards=fleet['cards'];assert len(cards)==7 and fleet['max_gpus']==3
 assert [c['index']for c in cards]==list(range(7))and len({c['uuid']for c in cards})==7
 for c in cards:
  assert isinstance(c['uuid'],str)and c['uuid'].startswith('GPU-')
  for k in('memory_mib','total_mib','sm'):assert type(c[k])is int
  assert 0<=c['memory_mib']<=c['total_mib']and c['total_mib']>4096 and 0<=c['sm']<=100
  for k in('pids','own_pids'):assert isinstance(c[k],list)and all(type(p)is int and p>0 for p in c[k])and len(set(c[k]))==len(c[k])
  assert set(c['own_pids'])<=set(c['pids'])
  assert type(c['available'])is bool and c['available']==(not c['pids']and c['memory_mib']<=64 and c['sm']<=5)
 assert type(fleet['own_gpu_count'])is int and fleet['own_gpu_count']==sum(bool(c['own_pids'])for c in cards)
 assert fleet['available_indices']==[c['index']for c in cards if c['available']]
 return fleet
def inventory():
 command=[sys.executable,'-B',str(Path(__file__).with_name('worker.py'))]
 p=subprocess.run(command,cwd=T,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=3)
 return validate(json.loads(p.stdout))
