"""CPU one-thread preload/fork proof; GPU child; wait proof before CPU cleanup."""
import argparse,json,os,time,traceback
from pathlib import Path
from sampler_gpu_spec import run,verify,write,read,sha,C,T,O
def main():
 p=argparse.ArgumentParser();p.add_argument('--barrier',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--nonce',required=True);a=p.parse_args()
 selected=os.environ.get('CUDA_VISIBLE_DEVICES','');os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 torch.set_num_threads(1)
 from cx_tta_online.native_entry_storage import NativeEntryStorage
 from cx_tta.evaluate_b2 import resolve_tokenizer
 from adapter_cache_cpu import load_cpu_policy,state_digest
 from child_exit import wait_and_record
 import sampler_gpu_work
 spec=run();verify(spec);a.root.mkdir(parents=True,exist_ok=True);runtime=a.root/'minimal_policy_runtime';runtime.mkdir()
 storage=NativeEntryStorage();storage.configure_runtime(a.root,runtime,resolve_tokenizer())
 try:
  policy,info=load_cpu_policy(verify_state=True)
  proof=next(json.loads(line.split(' ',1)[1]) for line in (T.parents[1]/'deployment/reports/target_adapter_cache_cpu_arrow_nvml_20261024.log').read_text().splitlines() if line.startswith('ADAPTER_CACHE_CPU_PROOF '))
  assert proof['passed'] and proof['helper_sha256']==sha(T.parents[1]/'deployment/adapter_cache_cpu.py')
  converted=state_digest(policy._model);assert converted==proof['converted_state_sha256'];jobs=[]
  for j in spec['jobs']:
   with np.load(j['history'],allow_pickle=False) as z:data={k:z[k].copy() for k in ('states','cam0','cam1','cam2')}
   data['query_ticks']=np.asarray(read(j['clock'])['query_observation_ticks'],np.int64);assert len(data['states'])==len(data['query_ticks'])==j['queries'];jobs.append(data)
  assert not torch.cuda.is_initialized() and len(list(Path('/proc/self/task').iterdir()))==1
  proof_child=os.fork()
  if proof_child==0:
   assert not torch.cuda.is_initialized() and not torch.cuda._is_in_bad_fork() and int(torch.arange(10).sum())==45
   os._exit(0)
  _,status=os.waitpid(proof_child,0);assert os.waitstatus_to_exitcode(status)==0
  write(a.barrier/'ready.json',dict(protocol_identity=spec['identity'],source_sha256=spec['source_sha256'],one_thread=True,CUDA_uninitialized=True,CPU_fork_child_passed=True,converted_state_sha256=converted,cached_queries=149))
  start=time.monotonic()
  while not (a.barrier/'permit.json').exists():
   if time.monotonic()-start>240:raise TimeoutError('No fresh GPU admission permit')
   time.sleep(.05)
  permit=read(a.barrier/'permit.json');assert permit['gpu_uuid']==a.gpu_uuid and permit['attempt_nonce']==a.nonce
  child=os.fork()
  if child==0:
   try:
    os.environ['CUDA_VISIBLE_DEVICES']=selected;assert not torch.cuda.is_initialized() and not torch.cuda._is_in_bad_fork()
    sampler_gpu_work.execute(policy,spec,a.root,a.barrier,a.gpu_uuid,jobs)
   except BaseException:
    traceback.print_exc();__import__('sys').stdout.flush();__import__('sys').stderr.flush();os._exit(1)
   os._exit(0)
  code=wait_and_record(child,a.barrier/'normal_child_exit.json',attempt_nonce=a.nonce,gpu_uuid=a.gpu_uuid,protocol_identity=spec['identity'],cpu_only_fixture=False)
 finally:storage.cleanup()
 raise SystemExit(code)
if __name__=='__main__':main()
