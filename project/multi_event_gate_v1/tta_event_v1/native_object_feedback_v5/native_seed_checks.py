"""Actual immutable CPU models on development q0, no policy or outcome fitting."""
from pathlib import Path
import json,random,numpy as np,torch
from .semantic import FrozenInstances
from .inference import infer_or_missing
from .language import new_prompt
from native_object_transfer_pilot_v2.common import T,read,sha,mapped
from adapter_cache_cpu import state_digest

def main():
 torch.set_num_threads(2);root=T/'outputs/native_object_transfer_pilot_v2/development/batches/000/runs/f2bb5f409e694dc1b64773507f3ac04a';O=T/'outputs/native_object_feedback_v5/CPU_development_seeds';O.mkdir(parents=True,exist_ok=False)
 with torch.random.fork_rng(devices=[]):model=FrozenInstances('cpu')
 before={k:state_digest(v)for k,v in {'DINO':model.detector,'SAM':model.sam}.items()};records=[];inputs={}
 for accepted in sorted((root/'accepted').glob('*.json')):
  item=read(accepted);meta=read(root/'initial_prefetch'/accepted.name);old=read(root/'semantic/answers'/(meta['content_key']+'.json'));frame=mapped(old['frame_file']);inputs[str(frame)]=sha(frame);inputs[str(accepted)]=sha(accepted)
  with np.load(frame,allow_pickle=False)as z:rgb=z['cam0'].copy()
  roles,phrases=new_prompt(item['instruction']);assert all(p in item['instruction'].lower()for p in phrases)
  value,receipt,inferred=infer_or_missing(model,rgb,item['instruction']);raw={k:v for k,v in value.items()if k!='masks'};record=dict(case=accepted.stem,instruction=item['instruction'],grounding_phrases=phrases,previous_available=old['available'],available=value['available'],reason=value['reason'],metadata=raw,CPU_receipt=receipt)
  if value['available']:
   file=O/(accepted.stem+'.npz')
   with file.open('xb')as f:np.savez_compressed(f,masks=value['masks'])
   record['mask_file']=str(file);record['mask_sha256']=sha(file)
  (O/(accepted.stem+'.json')).write_text(json.dumps(record,indent=2)+'\n');records.append(record);print('ACTUAL_Q0_V5_CPU',accepted.stem,old['available'],value['available'],phrases,flush=True)
 assert before=={k:state_digest(v)for k,v in {'DINO':model.detector,'SAM':model.sam}.items()}and not torch.cuda.is_initialized()
 result=dict(passed=True,records=records,actual_CPU_models_unchanged=True,CUDA_initialized=False,no_native_rollouts_or_efficacy_claim=True,unavailable_denominator_retained=True,input_sha256=inputs)
 (O/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
