"""Freeze fresh GPU-local CPU and GPU identity without allocating CUDA memory."""
from pathlib import Path
import argparse,json,sys,subprocess
T=Path(__file__).resolve().parents[1];P=T.parents[1];O=T/'outputs/native_visible_runtime_v1'
sys.path.insert(0,str(P/'deployment'));sys.path.insert(0,str(T))
from native_resource_policy_v4.placement import plan
from gpu_doctor import gpu_snapshot
from native_resource_policy_v4.admission import check_admission

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=5);a=p.parse_args();info=gpu_snapshot(a.gpu);check_admission(info)
 name=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=name','--format=csv,noheader'],text=True).strip();assert name=='NVIDIA GeForce RTX 4090'
 assert info['uuid']=='GPU-0f89dea0-728e-1d23-fd62-d25b9bad3e4f','Reuse requires same original control GPU'
 placement=plan(a.gpu);device=dict(gpu_index=a.gpu,gpu_uuid=info['uuid'],gpu_name=name,pci=info['pci'],source_GPU_uuid='GPU-fa87bc41-1919-efe1-73f9-11f38008763e',
  scope='New twelve developer episodes plus eight whole previous controls; same original control server/GPU; no independent scene or EVAL mixing')
 O.mkdir(parents=True,exist_ok=True)
 for filename,value in(('device_plan.json',device),('cpu_placement_plan.json',placement)):
  with(O/filename).open('x')as f:json.dump(value,f,indent=2);f.write('\n')
 print('FRESH_GPU_AND_CPU_PLAN_READY',json.dumps(device),placement['selected_logical_CPUs'],flush=True)
if __name__=='__main__':main()
