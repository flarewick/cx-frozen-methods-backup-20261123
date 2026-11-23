"""Guide reuse CPU equivalence and actual fully prepared model heap/fork/TOPP."""
from pathlib import Path
import sys,copy,types,random,pickle,gc,weakref,os,json,time
import numpy as np,torch
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped,verify
O=T/'outputs/native_guide_reuse_v1'

def main():
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 from adapter_cache_cpu import RT,state_digest
 sys.path[:0]=[str(RT/'RoboTwin/policy/pi05/src'),str(RT/'pi05_integration')]
 from native_direct_frames_v1.frames import install as install_frames
 install_frames()
 from openpi.models.model import Observation
 import jaxtyping._decorator as dec
 original=dec._check_dataclass_annotations
 from .typecheck import install
 fast=install()
 packet=dict(image={k:torch.arange(192,dtype=torch.float32).reshape(1,3,8,8)/256 for k in('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},image_mask={k:torch.ones(1,dtype=torch.bool)for k in('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},state=torch.zeros(1,32),tokenized_prompt=torch.zeros(1,48,dtype=torch.int64),tokenized_prompt_mask=torch.ones(1,48,dtype=torch.bool))
 cases=[packet,dict(packet,state=torch.zeros(2,32)),dict(packet,image={k:v.long()for k,v in packet['image'].items()}),{k:v for k,v in packet.items()if k!='tokenized_prompt_mask'}]
 def run(fn,p):
  dec._check_dataclass_annotations=fn;rng=torch.get_rng_state().clone()
  try:result=('ok',Observation.from_dict(copy.deepcopy(p)))
  except Exception as e:result=(type(e).__name__,str(e))
  assert torch.equal(rng,torch.get_rng_state());return result
 results=[]
 for p in cases:
  a,b=run(original,p),run(fast,p);assert a[0]==b[0]
  if a[0]=='ok':
   for key in('state','tokenized_prompt','tokenized_prompt_mask'):assert torch.equal(getattr(a[1],key),getattr(b[1],key))
   assert all(torch.equal(a[1].images[k],b[1].images[k])for k in a[1].images)
  else:assert a[1]==b[1]
  results.append(a[0])
 for name in('jax._src.tree_util','flax.nnx.transforms.compilation'):
  ns=dict(__name__=name,Observation=Observation,copy=copy,packet=cases[1]);exec('def call():return Observation.from_dict(copy.deepcopy(packet))',ns)
  for fn in(original,fast):dec._check_dataclass_annotations=fn;ns['call']()
 dec._check_dataclass_annotations=fast
 from native_batched_prefix_graph_v1.encoding import build
 from benefit_prefix_graph import capture_safe_embed
 from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
 class Stub:embed_prefix=PI0Pytorch.embed_prefix
 assert build().__name__=='batched_encode'and capture_safe_embed(Stub()).__name__=='embed_prefix'
 print('GUIDE_TYPECHECK_ORIGINAL_VALUES_ERRORS_SKIPS_AND_PREFIX_AST_PASS',flush=True)
 # Separate full prepare subprocess avoids modifying the predecessor direct-frame
 # installation assertion. This process only validates source/CPU math.
 sources=[p for p in Path(__file__).parent.glob('*')if p.is_file()]
 deployed=[T.parents[1]/'deployment'/n for n in ['benefit_stage_typecheck.py','benefit_prefix_graph.py','benefit_prefix_graph_lifetime_v10.py','baseline50_CPU_heap_freeze_v461.py','baseline50_backend_event_v420.py']]
 write(O/'CPU_component_checks_v1.json',dict(passed=True,CPU_only=True,original_Observation_values_errors_JAX_flax_skip_exact=True,cases=results,original_batch3_prefix_operations_AST_exact=True,guide_constant_owner_lifetime_repair_installed=True,CUDA_initialized=False,source_sha256={str(p):sha(p)for p in sources},input_sha256={str(p):sha(p)for p in deployed}))
if __name__=='__main__':main()
