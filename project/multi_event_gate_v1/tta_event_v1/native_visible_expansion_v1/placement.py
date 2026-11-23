"""Prefer GPU-local16CPUs; when occupied, use disjoint physical cores on another NUMA node."""
from pathlib import Path
import os,sys,json,hashlib
T=Path(__file__).resolve().parents[1];P=T.parents[1];sys.path.insert(0,str(P/'deployment'))
from native_resource_policy_v4.placement import parse,cores,select
from native_fast_inventory_v1.inventory import inventory
import gpu_doctor

def plan(gpu):
 info=gpu_doctor.gpu_snapshot(gpu);pci=info['pci'].lower().split(':');address='%04x:%s:%s'%(int(pci[0],16),pci[1],pci[2]);device=Path('/sys/bus/pci/devices')/address
 local=parse((device/'local_cpulist').read_text());node=int((device/'numa_node').read_text());assert node>=0
 allowed=set(os.sched_getaffinity(0));topology=cores(allowed|local);occupied=set();others=[]
 for card in inventory()['cards']:
  if card['uuid']==info['uuid']:continue
  for pid in card['own_pids']:
   try:affinity=set(os.sched_getaffinity(pid))
   except ProcessLookupError:continue
   topology.update(cores(affinity));occupied.update(topology[c]for c in affinity)
   others.append(dict(pid=pid,gpu_uuid=card['uuid'],logical_CPUs=sorted(affinity)))
 local_safe={c for c in allowed&local if topology[c]not in occupied}
 if len(local_safe)>=16:eligible=local;chosen_node=node;mode='GPU_local_disjoint_physical_cores'
 else:
  candidates=[]
  for path in Path('/sys/devices/system/node').glob('node[0-9]*'):
   cpus=parse((path/'cpulist').read_text());safe={c for c in allowed&cpus if topology[c]not in occupied}
   if len(safe)>=16:candidates.append((len({topology[c]for c in safe}),int(path.name[4:]),cpus))
  if not candidates:raise RuntimeError('No16logicalCPUs on physical cores disjoint from other own GPU processes')
  _,chosen_node,eligible=max(candidates,key=lambda x:(x[0],-x[1]));mode='remote_NUMA_disjoint_physical_cores_local_occupied'
 chosen=select(allowed,eligible,topology,occupied)
 return dict(version='expanded_developer_disjoint_CPU_plan_v1',gpu_uuid=info['uuid'],pci=address,numa_node=node,CPU_numa_node=chosen_node,placement_mode=mode,selected_logical_CPUs=chosen,logical_CPU_limit=16,selected_physical_cores=[list(topology[c])for c in chosen],GPU_local_cpulist=sorted(local),CPU_eligible_cpulist=sorted(eligible),other_own_gpu_processes=others,no_other_process_affinity_modified=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
def apply(value):
 chosen=set(value['selected_logical_CPUs']);assert len(chosen)==16 and chosen<=set(value['CPU_eligible_cpulist'])and chosen<=set(os.sched_getaffinity(0))
 actual=cores(chosen);assert [list(actual[c])for c in sorted(chosen)]==value['selected_physical_cores']
 local=parse((Path('/sys/bus/pci/devices')/value['pci']/'local_cpulist').read_text());assert set(value['GPU_local_cpulist'])==local
 os.sched_setaffinity(0,chosen);assert set(os.sched_getaffinity(0))==chosen
 return sorted(chosen)
