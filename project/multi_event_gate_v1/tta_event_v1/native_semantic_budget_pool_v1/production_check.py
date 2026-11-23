"""Actual production dispatcher/worker arrival replay; no GPU or fitting data."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped,verify,identity
from .audit import audit
from .transform import transform,restore

def main():
 source_stage=T/'outputs/native_onpolicy_collection_v29/fit_007';source=mapped(read(source_stage/'execution_state.json')['root']);original=read(source/'protocol.json');verify(original)
 assert read(source_stage/'actual_host_wait_exit.json')['exit_code']==1
 candidate=T/'native_onpolicy_collection_v30'
 for name,kind in(('semantic.py','dispatcher'),('semantic_worker.py','worker')):
  prior=(T/'native_onpolicy_collection_v29'/name).read_text().replace('native_onpolicy_collection_v29','native_onpolicy_collection_v30')
  assert restore((candidate/name).read_text(),kind)==prior and transform(prior,kind)==(candidate/name).read_text()
 root=T/'outputs/native_onpolicy_collection_v30/fit_007/CPU_actual_production_semantic_fixture_v1';assert not root.exists();root.mkdir(parents=True)
 cpus=set(read(source/'cpu_placement_plan.json')['selected_logical_CPUs']);assert len(cpus)==16;os.sched_setaffinity(0,cpus)
 spec=dict(original);spec.pop('identity');spec['source_sha256']=dict(original['source_sha256']);spec['input_sha256']=dict(original['input_sha256']);spec['scope']='CPU_only_semantic_diagnostic_no_native_or_fit_qualification'
 files=[candidate/'common.py',candidate/'semantic.py',candidate/'semantic_worker.py',*list(Path(__file__).parent.glob('*'))]
 for f in files:
  if f.is_file()and f.suffix in('.py','.md'):spec['source_sha256'][str(f)]=sha(f)
 spec['identity']=identity(spec);write(root/'protocol.json',spec);write(root/'CPU_fixture_ONLY.json',dict(no_GPU=True,no_native_or_fit_qualification=True,failed_masks_numeric_reference_only=True))
 requests=sorted((source/'semantic/requests').glob('*.json'),key=lambda p:(p.stat().st_mtime_ns,p.name))[:6];assert len(requests)==6
 rows=[read(p)for p in requests];arrival=[(p.stat().st_mtime_ns-requests[0].stat().st_mtime_ns)/1e9 for p in requests];answers=[root/'semantic/answers'/p.name for p in requests]
 inputs={str(p):sha(p)for p in requests};inputs[str(source/'protocol.json')]=sha(source/'protocol.json');inputs[str(source_stage/'actual_host_wait_exit.json')]=sha(source_stage/'actual_host_wait_exit.json')
 command=[sys.executable,'-B','-u','-m','native_onpolicy_collection_v30.semantic','--batch',str(root)];process=None
 try:
  with(root/'CPU_dispatcher.log').open('x')as log:
   process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CX_ONPOLICY_STAGE':'fit_007','CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'})
   start=time.monotonic()
   while not(root/'semantic/ready.json').exists():
    assert process.poll()is None and time.monotonic()-start<240;time.sleep(.02)
   sent=0;start=time.monotonic();first=None
   while not all(p.exists()for p in answers):
    elapsed=time.monotonic()-start;assert process.poll()is None and elapsed<420
    while sent<6 and arrival[sent]<=elapsed:
     write(root/'semantic/requests'/requests[sent].name,rows[sent]);sent+=1
    if first is None and any(p.exists()for p in answers):first=elapsed
    time.sleep(.01)
   total=time.monotonic()-start
   write(root/'semantic/stop.json',dict(CPU_diagnostic_complete=True));code=process.wait(timeout=90)
  actual=dict(exit_code=code,actual_Popen_wait_return=True,pid=process.pid,proc_absent_after_wait=not Path('/proc',str(process.pid)).exists());write(root/'CPU_dispatcher_actual_exit.json',actual);assert code==0 and actual['proc_absent_after_wait']
  bounded=audit(root);checks=[]
  for request,row,answer in zip(requests,rows,answers,strict=True):
   value=read(answer);ref=read(source/'semantic/answers'/request.name)
   assert value['available']==ref['available']and value['reason']==ref['reason']and value['models_state_sha256']==ref['models_state_sha256']
   assert not value['CUDA_initialized']and value['CPU_RNG_preserved']and value['actual_CPU_threads']in(2,4,8)
   for field in('roles','actual_object_count','actual_object_slots'):
    assert value['metadata'].get('raw',{}).get(field)==ref['metadata'].get('raw',{}).get(field)
   if value['available']:
    with np.load(mapped(value['mask_file']),allow_pickle=False)as a,np.load(mapped(ref['mask_file']),allow_pickle=False)as b:assert np.array_equal(a['masks'],b['masks'])
   inputs[str(answer)]=sha(answer);inputs[str(mapped(row['frame_file']))]=sha(mapped(row['frame_file']));checks.append(dict(content_key=value['content_key'],threads=value['actual_CPU_threads'],availability_reason_roles_masks_exact=True))
  old=T/'outputs/native_semantic_arrival_pool_v1/CPU_actual_arrival_pool_latency_v1.json';evidence=read(old);assert evidence['passed'];baseline=[r for r in evidence['runs']if r['mode']=='baseline'];assert first<min(r['first_answer_seconds']for r in baseline)and total<min(r['all_six_seconds']for r in baseline)
  inputs[str(old)]=sha(old);inputs[str(root/'CPU_semantic_thread_budget_audit.json')]=sha(root/'CPU_semantic_thread_budget_audit.json')
  O=T/'outputs/native_semantic_budget_pool_v1';O.mkdir(exist_ok=True)
  write(O/'CPU_production_arrival_checks_v1.json',dict(passed=True,first_answer_seconds=first,all_six_seconds=total,actual_arrival_seconds=arrival,actual_production_dispatcher_and_workers=True,actual_parent_wait0=True,actual_worker_waits0=True,maximum_active_threads=bounded['maximum_active_threads'],same_original_NN_models_float32_and_final_masks=True,checks=checks,no_GPU_optimizer_or_native_qualification=True,source_sha256={str(f):sha(f)for f in files if f.is_file()and f.suffix in('.py','.md')},input_sha256=inputs))
  print('ACTUAL_PRODUCTION_ADAPTIVE_CPU_POOL_PASS',first,total,bounded['maximum_active_threads'],flush=True)
 finally:
  if process is not None and process.poll()is None:
   process.terminate()
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':main()
