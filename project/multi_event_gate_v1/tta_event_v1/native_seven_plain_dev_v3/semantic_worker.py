"""Independent CPU earliest-RGB owner; no GPU, task, seed or outcome features."""
from pathlib import Path
import argparse,time,random,traceback
from .semantic import content_key
from .common import read,write,sha,identity,verify

def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--worker',type=int,choices=(0,1,2,3),required=True);a=p.parse_args();root=a.batch/'semantic';work=root/'workers'/str(a.worker);spec=read(a.batch/'protocol.json');verify(spec)
 import numpy as np,torch
 from native_pot_mic_semantics_v1.inference import infer_or_missing,rng_digest
 from native_pot_mic_semantics_v1 import semantic
 from adapter_cache_cpu import state_digest
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
 py=random.getstate();ns=np.random.get_state();before=torch.random.get_rng_state().clone();rng=rng_digest()
 try:
  with torch.random.fork_rng(devices=[]):detector=semantic.FrozenInstances('cpu')
 finally:random.setstate(py);np.random.set_state(ns)
 assert not torch.cuda.is_initialized()and torch.equal(before,torch.random.get_rng_state())and rng_digest()==rng
 models={'DINO':detector.detector,'SAM':detector.sam};digests={k:state_digest(v)for k,v in models.items()}
 from .common import T
 assert digests=={k:read(T/'outputs/native_arrived_mask_v3/cpu_models.json')['model_state_sha256'][k]for k in models}
 torch.set_num_threads(2);write(work/'ready.json',dict(passed=True,models_state_sha256=digests,CUDA_initialized=False,source_identity=spec['identity']))
 done=set();began=time.monotonic()
 while True:
  pending=sorted((work/'assigned').glob('*.json'))
  for req in pending:
   if req.name in done:continue
   row=read(req);assert set(row)=={'frame_file','frame_sha256','instruction','content_key'}
   assert sha(row['frame_file'])==row['frame_sha256']
   with np.load(row['frame_file'],allow_pickle=False)as z:images=tuple(z['cam'+str(i)].copy()for i in range(3));command=z['command_state'].copy()
   assert content_key(images,command,row['instruction'])==row['content_key']
   allocation=work/'controls'/req.name;lease=read(allocation);threads=lease['threads']
   assert lease['content_key']==row['content_key']and lease['worker']==a.worker and threads in(2,4,8)and lease['maximum_active_CPU_threads']==8 and lease['active_after']<=8
   torch.set_num_threads(threads);assert torch.get_num_threads()==threads
   before=rng_digest();cpu=torch.random.get_rng_state().clone();start=time.monotonic()
   value,metadata,inferred=infer_or_missing(detector,images[0],row['instruction'])
   assert before==rng_digest()and torch.equal(cpu,torch.random.get_rng_state())and not torch.cuda.is_initialized()
   result=dict(available=bool(value['available']),reason=value['reason'],models_state_sha256=digests,metadata=metadata,content_key=row['content_key'],frame_file=row['frame_file'],frame_sha256=row['frame_sha256'],query=0,tick=0,actual_CPU_inference=inferred,actual_CPU_threads=threads,CPU_thread_allocation_file=str(allocation),CPU_thread_allocation_sha256=sha(allocation),cold_CPU_inference_seconds=time.monotonic()-start,CPU_RNG_preserved=True,CUDA_initialized=False)
   if value['available']:
    mask=root/'answers'/(req.stem+'.npz');mask.parent.mkdir(exist_ok=True)
    with mask.open('xb')as f:np.savez_compressed(f,masks=value['masks'])
    result.update(mask_file=str(mask),mask_sha256=sha(mask))
   write(root/'answers'/req.name,result);done.add(req.name);print('BEST100_CPU_SEMANTIC',req.stem,result['available'],flush=True)
  if(work/'stop.json').exists():break
  if time.monotonic()-began>12000:raise TimeoutError('CPU semantic owner deadline')
  time.sleep(.05)
 assert all(state_digest(v)==digests[k]for k,v in models.items());verify(spec)
 write(work/'result.json',dict(complete=True,requests=len(done),models_unchanged=True,CUDA_initialized=False,all_RNG_preserved=True,source_identity=spec['identity']))

if __name__=='__main__':
 try:main()
 except BaseException:
  import sys
  root=Path(sys.argv[sys.argv.index('--batch')+1])/'semantic/workers'/sys.argv[sys.argv.index('--worker')+1]
  write(root/'failure.json',dict(traceback=traceback.format_exc()));raise
