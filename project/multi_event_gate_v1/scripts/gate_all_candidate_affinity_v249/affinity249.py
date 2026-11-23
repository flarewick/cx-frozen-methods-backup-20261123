"""Measured NUMA CPU placement only; no model or scene scheduling change."""
import os,time,json
from pathlib import Path
import psutil

def physical(cpu):
 root=Path('/sys/devices/system/cpu',f'cpu{cpu}','topology')
 return tuple(int((root/n).read_text())for n in ['physical_package_id','core_id'])

def select(fleet,allowed,loads):
 used=set()
 for card in fleet['cards']:
  for pid in card['own_pids']:
   try:used.update(physical(c)for c in os.sched_getaffinity(pid))
   except ProcessLookupError:pass
 candidates=[]
 for node in [0,1]:
  groups={}
  for c in sorted(allowed):
   p=physical(c)
   if p not in used and Path(f'/sys/devices/system/node/node{node}/cpu{c}').exists():groups.setdefault(p,[]).append(c)
  # Use physical-core max sibling utilization so a busy hyperthread is counted.
  rows=[]
  for p,cs in groups.items():
   siblings=[c for c in range(len(loads))if physical(c)==p]
   rows.append((max(loads[c]for c in siblings),min(cs)))
  rows.sort();chosen=rows[:16]
  if len(chosen)>=12:candidates.append((-len(chosen),sum(x[0]for x in chosen)/len(chosen),node,chosen))
 if not candidates:raise RuntimeError('No unused 12..16-core NUMA set; no GPU launched')
 _,mean,node,rows=min(candidates);cores=[c for _,c in rows]
 assert 12<=len(cores)<=16 and len({physical(c)for c in cores})==len(cores)
 assert not({physical(c)for c in cores}&used)
 return cores,node,dict(node=node,cores=cores,sibling_max_loads=[v for v,_ in rows],mean_load=mean,account_used_physical=[list(p)for p in sorted(used)])

def choose(fleet):
 loads=psutil.cpu_percent(interval=2,percpu=True)
 cores,node,record=select(fleet,os.sched_getaffinity(0),loads)
 print('AFFINITY249_MEASURED',json.dumps(record,sort_keys=True),flush=True)
 return cores,node
