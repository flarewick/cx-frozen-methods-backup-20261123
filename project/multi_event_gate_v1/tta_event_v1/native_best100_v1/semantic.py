"""Independent CPU earliest-RGB owner; no GPU, task, seed or outcome features."""
from pathlib import Path
import argparse,time,random,traceback
from .common import read,write,sha,identity,verify

def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();root=a.batch/'semantic';spec=read(a.batch/'protocol.json');verify(spec)
 import numpy as np,torch
 from native_expert_full_seed_v6.batch import infer_or_missing,rng_digest
 from seed import semantic
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
 torch.set_num_threads(2);write(root/'ready.json',dict(passed=True,models_state_sha256=digests,CUDA_initialized=False,source_identity=spec['identity']))
 done=set();began=time.monotonic()
 while True:
  pending=sorted((root/'requests').glob('*.json'))
  for req in pending:
   if req.name in done:continue
   row=read(req);assert set(row)=={'frame_file','frame_sha256','instruction','content_key'}
   assert sha(row['frame_file'])==row['frame_sha256']
   with np.load(row['frame_file'],allow_pickle=False)as z:images=tuple(z['cam'+str(i)].copy()for i in range(3));command=z['command_state'].copy()
   assert content_key(images,command,row['instruction'])==row['content_key']
   before=rng_digest();cpu=torch.random.get_rng_state().clone();start=time.monotonic()
   value,metadata,inferred=infer_or_missing(detector,images[0],row['instruction'])
   assert before==rng_digest()and torch.equal(cpu,torch.random.get_rng_state())and not torch.cuda.is_initialized()
   result=dict(available=bool(value['available']),reason=value['reason'],models_state_sha256=digests,metadata=metadata,content_key=row['content_key'],frame_file=row['frame_file'],frame_sha256=row['frame_sha256'],query=0,tick=0,actual_CPU_inference=inferred,cold_CPU_inference_seconds=time.monotonic()-start,CPU_RNG_preserved=True,CUDA_initialized=False)
   if value['available']:
    mask=root/'answers'/(req.stem+'.npz');mask.parent.mkdir(exist_ok=True)
    with mask.open('xb')as f:np.savez_compressed(f,masks=value['masks'])
    result.update(mask_file=str(mask),mask_sha256=sha(mask))
   write(root/'answers'/req.name,result);done.add(req.name);print('BEST100_CPU_SEMANTIC',req.stem,result['available'],flush=True)
  if(root/'stop.json').exists():break
  if time.monotonic()-began>12000:raise TimeoutError('CPU semantic owner deadline')
  time.sleep(.05)
 assert all(state_digest(v)==digests[k]for k,v in models.items());verify(spec)
 write(root/'result.json',dict(complete=True,requests=len(done),models_unchanged=True,CUDA_initialized=False,all_RNG_preserved=True,source_identity=spec['identity']))

def content_key(images,command,instruction):
 from native_visible_expand_semantics_v2.store import key
 return identity(key(images,command,instruction))

class SemanticClient:
 def __init__(self,root):
  import threading
  self.root=Path(root);self.cache={};self.lock=threading.Lock();self.result={'available':0}
 def at_initial(self,observed,instruction,*,query,tick):
  assert query==tick==0
  import numpy as np
  key=content_key(observed['rgb'],observed['command_state'],instruction);started=time.monotonic()
  with self.lock:
   req=self.root/'requests'/(key+'.json');frame=self.root/'requests'/(key+'.npz')
   if not req.exists():
    frame.parent.mkdir(parents=True,exist_ok=True)
    with frame.open('xb')as f:np.savez_compressed(f,**{'cam'+str(i):im for i,im in enumerate(observed['rgb'])},command_state=observed['command_state'])
    write(req,dict(frame_file=str(frame),frame_sha256=sha(frame),instruction=instruction,content_key=key))
  answer=self.root/'answers'/(key+'.json')
  while not answer.exists():
   if(self.root/'failure.json').exists():raise RuntimeError('CPU semantic owner failed')
   if time.monotonic()-started>240:raise TimeoutError('No actual CPU q0 semantic response')
   time.sleep(.05)
  row=read(answer);assert row['content_key']==key and not row['CUDA_initialized']and row['CPU_RNG_preserved'];masks=None
  if row['available']:
   assert sha(row['mask_file'])==row['mask_sha256']
   with np.load(row['mask_file'],allow_pickle=False)as z:masks=z['masks'].copy()
   assert masks.shape==(2,240,320)and masks.dtype==bool
  import hashlib
  expected=dict(rgb=tuple(im.copy()for im in observed['rgb']),command_state=observed['command_state'].copy(),instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),available=row['available'])
  report=dict(available=row['available'],reason=row['reason'],models_state_sha256=row['models_state_sha256'],CPU_result_sha256=sha(answer),CPU_result_file=str(answer),cold_CPU_inference_seconds=row['cold_CPU_inference_seconds'],actual_lookup_seconds=time.monotonic()-started,query=0,tick=0,all_three_RGB_command_language_exact=True,old_expert_masks_NOT_imported=True,case_task_seed_not_cache_keys=True)
  return expected,masks,report
if __name__=='__main__':
 try:main()
 except BaseException:
  import sys
  root=Path(sys.argv[sys.argv.index('--batch')+1])/'semantic';write(root/'failure.json',dict(traceback=traceback.format_exc()));raise
