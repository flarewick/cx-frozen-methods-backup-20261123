"""Same immutable CPU inference/RNG contract as the original feedback interface."""
import copy
from native_transfer_semantics_v2.inference import rng_digest
from . import semantic
def infer_or_missing(model,rgb,instruction):
 from native_transfer_semantics_v2.semantic import prompt as old_prompt
 from native_transfer_semantics_v2.inference import infer_or_missing as previous
 if old_prompt(instruction)[1]:return previous(model,rgb,instruction)
 roles,phrases=semantic.prompt(instruction)
 if not phrases:return dict(available=False,reason='unsupported_language_retained'),dict(roles=roles,phrases=phrases,device='cpu',actual_model_forward=False,interface_version=semantic.VERSION,unknown_never_imputed=True),False
 before=rng_digest();value=model.infer([rgb],[instruction])[0];assert before==rng_digest()
 raw={k:copy.deepcopy(v)for k,v in value.items()if k!='masks'}
 return value,dict(raw=raw,decision=raw,joint=None,head_batch=1,CPU_RNG_preserved=True,CUDA_initialized=False,device='cpu',interface_version=semantic.VERSION,actual_object_count=value.get('actual_object_count',roles['actual_object_count']),language_rewritten=False),True
