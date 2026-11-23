"""Actual frozen CPU grounding on all four retained roller TRAIN scenes."""
from pathlib import Path
import types,numpy as np,torch
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped
from native_object_feedback_v4.semantic import FrozenInstances as Original
from adapter_cache_cpu import state_digest
from .language import prompt,VERSION
def main():
 torch.set_num_threads(2);o=T/'outputs/native_object_association_v1/descriptor_probe';o.mkdir(parents=True,exist_ok=False)
 r=mapped(read(T/'outputs/native_onpolicy_coverage_v7/pilot_fit/execution_state.json')['root']);records=[];inputs={}
 with torch.random.fork_rng(devices=[]):model=Original('cpu')
 fn=Original._infer_new;method=types.FunctionType(fn.__code__,dict(fn.__globals__,prompt=prompt,VERSION=VERSION),fn.__name__,fn.__defaults__,fn.__closure__)
 before={k:state_digest(v)for k,v in {'DINO':model.detector,'SAM':model.sam}.items()}
 for n in range(6,10):
  case='grab_roller:712000'+str(n);item=read(r/'accepted'/(case+'.json'));c=r/'cases'/case/'M_event_full';result=read(c/'result.json');source=next(q['query']for q in result['queries']if q['event']['decision']['selected_event'])
  for q in (0,source):
   f=c/('observation%03d.npz'%q);inputs[str(f)]=sha(f)
   with np.load(f,allow_pickle=False)as z:rgb=z['cam0'].copy()
   value=method(model,rgb,item['instruction']);raw={k:v for k,v in value.items()if k!='masks'};record=dict(case_id=case,query=q,instruction=item['instruction'],metadata=raw)
   if value['available']:
    mask=o/(case+'_q'+str(q)+'.npz')
    with mask.open('xb')as h:np.savez_compressed(h,masks=value['masks'])
    record.update(mask_file=str(mask),mask_sha256=sha(mask))
   records.append(record);write(o/(case+'_q'+str(q)+'.json'),record);print('ACTUAL_TRAIN_DESCRIPTOR',case,q,prompt(item['instruction'])[1],value['available'],value['reason'],flush=True)
 assert before=={k:state_digest(v)for k,v in {'DINO':model.detector,'SAM':model.sam}.items()}and not torch.cuda.is_initialized()
 write(o/'result.json',dict(passed=True,records=records,actual_models_unchanged=True,CUDA_initialized=False,input_sha256=inputs,original_grounding_math_source_sha256=sha(Path(__import__(Original.__module__,fromlist=['x']).__file__)),offline_TRAIN_diagnostic_only=True,optimizer_steps=0,TTA_benefit_proven=False))
if __name__=='__main__':main()
