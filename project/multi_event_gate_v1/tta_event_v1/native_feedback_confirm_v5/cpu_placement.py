"""Read-only GPU-local topology plan; never changes another process affinity."""
from pathlib import Path
import argparse,hashlib,json,os,sys
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1];O=T/'outputs/native_feedback_confirm_v5'
def parse(text):
 out=set()
 for part in text.strip().split(','):
  if not part:raise ValueError('Empty CPU range')
  ends=part.split('-')
  if len(ends)>2 or not all(x.isdigit()for x in ends):raise ValueError('Invalid CPU range')
  a=int(ends[0]);b=int(ends[-1])
  if b<a:raise ValueError('Reversed CPU range')
  out.update(range(a,b+1))
 return out
def cores(cpus):
 result={}
 for cpu in cpus:
  root=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
  result[cpu]=(int((root/'physical_package_id').read_text()),int((root/'core_id').read_text()))
 return result
def select(allowed,local,topology,occupied,count=16):
 candidates=set(allowed)&set(local)
 safe=[c for c in candidates if topology[c]not in occupied]
 if len(safe)<count:raise RuntimeError('Fewer than16 available GPU-local logical CPUs excluding other own GPU cores')
 result=[];seen=set()
 while len(result)<count:
  c=min((x for x in safe if x not in result),key=lambda x:(topology[x]in seen,x))
  result.append(c);seen.add(topology[c])
 return sorted(result)
def plan(gpu):
 sys.path.insert(0,str(P/'deployment'))
 import gpu_doctor
 from gpu_policy import inventory
 info=gpu_doctor.gpu_snapshot(gpu);pci=info['pci'].lower().split(':')
 address='%04x:%s:%s'%(int(pci[0],16),pci[1],pci[2]);device=Path('/sys/bus/pci/devices')/address
 local=parse((device/'local_cpulist').read_text());node=int((device/'numa_node').read_text());assert node>=0
 allowed=set(os.sched_getaffinity(0));topology=cores(allowed|local);occupied=set();others=[]
 for card in inventory()['cards']:
  if card['uuid']==info['uuid']:continue
  for pid in card['own_pids']:
   try:affinity=set(os.sched_getaffinity(pid))
   except ProcessLookupError:continue
   topology.update(cores(affinity));occupied.update(topology[c]for c in affinity)
   others.append(dict(pid=pid,gpu_uuid=card['uuid'],logical_CPUs=sorted(affinity)))
 chosen=select(allowed,local,topology,occupied)
 return dict(version='native_feedback_confirm_v5_GPU_local_CPU_plan',gpu_uuid=info['uuid'],pci=address,numa_node=node,
  selected_logical_CPUs=chosen,logical_CPU_limit=16,selected_physical_cores=[list(topology[c])for c in chosen],
  GPU_local_cpulist=sorted(local),other_own_gpu_processes=others,
  no_other_process_affinity_modified=True,prior_v2_affinity=list(range(96,112)),
  prior_v2_affinity_GPU_local=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
def apply(value):
 chosen=set(value['selected_logical_CPUs']);assert len(chosen)==16 and chosen<=set(value['GPU_local_cpulist'])
 assert chosen<=set(os.sched_getaffinity(0)),'Frozen placement incompatible with inherited CPU affinity'
 os.sched_setaffinity(0,chosen);assert set(os.sched_getaffinity(0))==chosen
 return sorted(chosen)
def pages():
 result={}
 for line in Path('/proc/self/numa_maps').read_text().splitlines():
  fields=line.split()
  if not any(v.startswith('anon=')for v in fields):continue
  for part in fields:
   key,sep,value=part.partition('=')
   if sep and key.startswith('N')and key[1:].isdigit()and value.isdigit():result[key]=result.get(key,0)+int(value)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args();v=plan(a.gpu)
 f=O/'cpu_placement_plan.json';f.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(v,indent=2)+'\n'
 if f.exists():assert f.read_text()==text,'Placement plan already frozen'
 else:f.write_text(text)
 print('GPU_LOCAL_CPU_PLAN',v['selected_logical_CPUs'],v['numa_node'],flush=True)
