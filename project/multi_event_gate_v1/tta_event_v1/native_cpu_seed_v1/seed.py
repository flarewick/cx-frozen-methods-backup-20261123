"""Device-only CPU port of the frozen original earliest-frame pair adjudication."""
from pathlib import Path
import importlib.util
import numpy as np,torch
T=Path(__file__).resolve().parent.parent
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
semantic=module('CPU_seed_exact_frozen_semantic',T/'language_instances_v2/semantic.py')
refine=module('CPU_seed_exact_observed_refinement',T/'language_instances_v2/refinement.py').refine
adjudicate=module('CPU_seed_exact_joint_adjudication',T/'joint_instances_v3/adjudicate.py').select

def infer_pair_cpu(detector,image,instruction):
 from PIL import Image
 if detector.device!='cpu' or torch.cuda.is_initialized():raise ValueError('CPU-only semantic owner')
 cpu=torch.random.get_rng_state().clone()
 raw=detector.infer([image],[instruction])[0]
 result=refine(image,instruction,raw);roles,phrases=semantic.prompt(instruction);colors=(roles['target'],roles['reference']) if roles['directed'] else roles.get('instance_colors');joint=None
 if colors is not None and len(raw['candidates'])==2 and all(raw['candidates']) and sum(map(len,raw['candidates']))>2:
  candidates=[dict(c,phrase_index=p,proposal_index=i) for p,group in enumerate(raw['candidates']) for i,c in enumerate(group)]
  ins=detector.sam_processor(images=[Image.fromarray(image)],input_boxes=[[c['box'] for c in candidates]],return_tensors='pt').to('cpu')
  with torch.inference_mode():out=detector.sam(**ins,multimask_output=False)
  masks=detector.sam_processor.image_processor.post_process_masks(out.pred_masks.cpu(),ins['original_sizes'].cpu(),ins['reshaped_input_sizes'].cpu())[0][:,0].numpy().astype(bool);quality=out.iou_scores.cpu().numpy()[0,:,0];joint=adjudicate(image,instruction,candidates,masks,quality);result=dict(result,available=joint['available'],reason=joint['reason'])
  if joint['available']:result['masks']=masks[joint['selected_indices']]
  del ins,out
 assert torch.equal(cpu,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 metadata=dict(raw={k:v for k,v in raw.items()if k!='masks'},decision={k:v for k,v in result.items()if k!='masks'},
               joint=joint,head_batch=1,CPU_RNG_preserved=True,CUDA_initialized=False,device='cpu')
 return result,metadata
