"""Actual two CPU subprocesses on four saved native q0 requests; no GPU."""
from pathlib import Path
import ast,json,subprocess,sys,time,shutil
import numpy as np,torch
from .common import C,T,O,read,write,sha

def wait_file(path,process,seconds):
 start=time.monotonic()
 while not path.exists():
  if process.poll()is not None:raise RuntimeError('CPU coordinator exited before '+str(path))
  if time.monotonic()-start>seconds:raise TimeoutError(str(path))
  time.sleep(.05)
 return read(path)
def main():
 old=T/'outputs/native_cross_task100_v3/batches/002/runs/f06bbbf3b5f44bb498e919d679727527';root=O/'CPU_semantic_parallel_fixture';root.mkdir();write(root/'protocol.json',read(old/'protocol.json'))
 # Identical client means identical q0 content binding and no new model features.
 def client(file):return ast.dump(next(n for n in ast.parse(file.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='SemanticClient'))
 assert client(C/'semantic.py')==client(T/'native_cross_task100_v3/semantic.py')
 jobs=sorted((old/'semantic/requests').glob('*.json'));assert len(jobs)==8
 log=(root/'coordinator.log').open('x');p=subprocess.Popen([sys.executable,'-u','-B','-m','native_cross_task100_v14.semantic','--batch',str(root)],stdout=log,stderr=subprocess.STDOUT)
 try:
  ready=wait_file(root/'semantic/ready.json',p,240);assert ready['independent_CPU_workers']==4 and ready['threads_per_worker']==2 and not ready['CUDA_initialized'];start=time.monotonic()
  for job in jobs:write(root/'semantic/requests'/job.name,read(job))
  rows=[];serial=0.
  for job in jobs:
   actual=wait_file(root/'semantic/answers'/job.name,p,180);original=read(old/'semantic/answers'/job.name)
   for key,value in original.items():
    if key in('cold_CPU_inference_seconds','mask_file','mask_sha256'):continue
    assert actual[key]==value,(job.name,key)
   if original['available']:
    with np.load(actual['mask_file'],allow_pickle=False)as z:a=z['masks'].copy()
    with np.load(original['mask_file'],allow_pickle=False)as z:b=z['masks'].copy()
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
   serial+=original['cold_CPU_inference_seconds'];rows.append(dict(key=job.stem,available=actual['available'],metadata_exact=True,masks_exact=actual['available'],original_seconds=original['cold_CPU_inference_seconds'],parallel_worker_seconds=actual['cold_CPU_inference_seconds']))
  wall=time.monotonic()-start;old_busy=[]
  for worker in range(2):
   assigned=list((old/'semantic/workers'/str(worker)/'assigned').glob('*.json'));old_busy.append(sum(read(old/'semantic/answers'/p.name)['cold_CPU_inference_seconds']for p in assigned))
  assert wall<max(old_busy),(wall,old_busy)
  write(root/'semantic/stop.json',dict(CPU_fixture_complete=True));code=p.wait(timeout=90);assert code==0
  result=read(root/'semantic/result.json');assert result['complete']and result['requests']==8 and result['actual_worker_exit_codes']==[0,0,0,0]and not result['CUDA_initialized'];assert wall<serial,(wall,serial)
  assert not torch.cuda.is_initialized()
  write(O/'CPU_semantic_parallel_checks.json',dict(passed=True,actual_CPU_subprocess_exit_code=code,actual_four_worker_exit_codes=result['actual_worker_exit_codes'],eight_saved_request_metadata_and_masks_exact=True,parallel_request_wall_seconds=wall,original_two_worker_busy_lower_bound_seconds=max(old_busy),original_two_worker_busy_seconds=old_busy,original_serial_inference_sum_seconds=serial,rows=rows,original_client_AST_exact=True,CUDA_initialized=False,new_scientific_scenes=0,training_steps=0,source_sha256={str(f):sha(f)for f in C.glob('*.py')},input_sha256={str(f):sha(f)for f in jobs}))
  print('FOUR_CPU_SEMANTIC_EXACT_AND_FASTER_PASS',wall,serial,flush=True)
 finally:
  if p.poll()is None:p.terminate();p.wait(timeout=15)
  log.close()
if __name__=='__main__':main()
