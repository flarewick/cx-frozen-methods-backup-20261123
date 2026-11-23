"""Freeze fresh GPU-local CPU and GPU identity without allocating CUDA memory."""
from pathlib import Path
import argparse,json,sys,subprocess
T=Path(__file__).resolve().parents[1];P=T.parents[1];O=T/'outputs/native_expert_full_cache_v8'
sys.path.insert(0,str(P/'deployment'));sys.path.insert(0,str(T))
from native_feedback_confirm_v5.cpu_placement import plan
from gpu_doctor import gpu_snapshot,check_admission

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=2);a=p.parse_args();info=gpu_snapshot(a.gpu);check_admission(info)
 name=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=name','--format=csv,noheader'],text=True).strip();assert name=='NVIDIA GeForce RTX 4090'
 placement=plan(a.gpu);device=dict(gpu_index=a.gpu,gpu_uuid=info['uuid'],gpu_name=name,pci=info['pci'],source_GPU_uuid='GPU-fa87bc41-1919-efe1-73f9-11f38008763e',
  scope='Whole remaining expert-cache cases on another RTX4090 of same server; source identities/hardware provenance explicit, no episode-branch splicing or EVAL mixing')
 O.mkdir(parents=True,exist_ok=True)
 for filename,value in(('device_plan.json',device),('cpu_placement_plan.json',placement)):
  with(O/filename).open('x')as f:json.dump(value,f,indent=2);f.write('\n')
 print('FRESH_GPU_AND_CPU_PLAN_READY',json.dumps(device),placement['selected_logical_CPUs'],flush=True)
if __name__=='__main__':main()
