"""Keep the original trained path; only add explicit heterogeneous grounding."""
import torch
from native_cpu_seed_v1.seed import infer_pair_cpu
from native_expert_full_seed_v6.batch import rng_digest
from . import semantic

def infer_or_missing(detector,image,instruction):
 old_roles,old_phrases=semantic.original.prompt(instruction)
 if old_phrases:
  value,metadata=infer_pair_cpu(detector,image,instruction)
  return value,metadata,True
 roles,phrases=semantic.prompt(instruction)
 if not phrases:
  return dict(available=False,reason='unsupported_language_retained'),dict(roles=roles,phrases=phrases,device='cpu',actual_model_forward=False,frozen_parser_unchanged=True,language_rewritten=False,unknown_never_imputed=True),False
 assert roles['kind']=='explicit_heterogeneous_placement_pair'and detector.device=='cpu'
 cpu=torch.random.get_rng_state().clone();before=rng_digest();raw=detector.infer([image],[instruction])[0]
 assert torch.equal(cpu,torch.random.get_rng_state())and before==rng_digest()and not torch.cuda.is_initialized()
 metadata=dict(raw={k:v for k,v in raw.items()if k!='masks'},decision={k:v for k,v in raw.items()if k!='masks'},joint=None,head_batch=1,CPU_RNG_preserved=True,CUDA_initialized=False,device='cpu',interface_version='native_transfer_semantics_v1',different_categories_same_shape_color_criterion_inapplicable=True)
 return raw,metadata,True
