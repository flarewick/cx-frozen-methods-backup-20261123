"""All twenty retained TRAIN initials; no scene filtering or outcome inputs."""
from pathlib import Path
import numpy as np,torch
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped
from adapter_cache_cpu import state_digest
from .semantic import FrozenInstances
from .inference import infer_or_missing
from .language import prompt,VERSION
def main():
 torch.set_num_threads(2);o=T/'outputs/native_object_association_v1/candidate_checks';o.mkdir(parents=True,exist_ok=False)
 r=mapped(read(T/'outputs/native_onpolicy_coverage_v7/pilot_fit/execution_state.json')['root']);records=[];inputs={}
 with torch.random.fork_rng(devices=[]):model=FrozenInstances('cpu')
 before={k:state_digest(v)for k,v in {'DINO':model.detector,'SAM':model.sam}.items()}
 for f in sorted((r/'accepted').glob('*.json')):
  item=read(f);c=r/'cases'/item['case_id']/'M_event_full';receipt=read(c/'semantic_initial_receipt.json');answer=mapped(receipt['CPU_result_file']);old=read(answer);frame=c/'observation000.npz';inputs[str(frame)]=sha(frame);inputs[str(answer)]=sha(answer);inputs[str(f)]=sha(f)
  with np.load(frame,allow_pickle=False)as z:rgb=z['cam0'].copy()
  value,meta,inferred=infer_or_missing(model,rgb,item['instruction']);changed=value.get('grounding_decision_path')=='exact_previously_omitted_nominal_modifiers'
  if not changed:
   assert value['available']==old['available']and value['reason']==old['reason']
   if value['available']:
    with np.load(mapped(old['mask_file']),allow_pickle=False)as z:assert np.array_equal(value['masks'],z['masks'])
  raw={k:v for k,v in value.items()if k!='masks'};row=dict(case_id=item['case_id'],instruction=item['instruction'],old_available=old['available'],available=value['available'],changed=changed,metadata=raw)
  if value['available']:
   mask=o/(item['case_id']+'.npz')
   with mask.open('xb')as h:np.savez_compressed(h,masks=value['masks'])
   row.update(mask_file=str(mask),mask_sha256=sha(mask))
  write(o/(item['case_id']+'.json'),row);records.append(row);print('CPU_ACTUAL_CANDIDATE_Q0',item['case_id'],old['available'],value['available'],changed,flush=True)
 assert len(records)==20 and sum(x['available']for x in records if x['case_id'].startswith('grab_roller:'))==4
 assert before=={k:state_digest(v)for k,v in {'DINO':model.detector,'SAM':model.sam}.items()}and not torch.cuda.is_initialized()
 write(o/'result.json',dict(passed=True,records=records,input_sha256=inputs,original_weights_thresholds_dtype_unchanged=True,unchanged_branches_bit_exact=True,CUDA_initialized=False,all20_fixed_TRAIN_scenes_retained=True,optimizer_steps=0,no_source_query_redetection=True,no_native_feedback_or_TTA_benefit_claim=True,source_sha256={str(p):sha(p)for p in Path(__file__).resolve().parent.glob('*.py')}))
if __name__=='__main__':main()
