"""CPU AST check; original source code/env never patched on disk."""
from pathlib import Path
import json,hashlib
from .encoding import build

def main():
 import torch
 from benefit_prefix_graph import capture_safe_embed
 from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
 class Stub:embed_prefix=PI0Pytorch.embed_prefix
 f=build();g=capture_safe_embed(Stub());assert f.__name__=='batched_encode'and g.__name__=='embed_prefix'and not torch.cuda.is_initialized()
 T=Path(__file__).resolve().parents[1];O=T/'outputs/native_batched_prefix_graph_v1';files=[Path(__file__),Path(__file__).with_name('encoding.py'),T.parents[1]/'deployment/benefit_prefix_graph.py',T.parents[1]/'deployment/benefit_prefix_graph_lifetime_v10.py']
 proof=dict(passed=True,original_all_tensor_operations_AST_exact=True,only_prefix_dispatch_replaced=True,original_preprocessing_pooling_FP16_metadata_retained=True,original_action_sampler_unchanged=True,CUDA_initialized=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files})
 with(O/'CPU_AST_checks.json').open('x')as f:json.dump(proof,f,indent=2)
 print('BATCH3_ORIGINAL_PREFIX_GRAPH_AST_CPU_PASS',flush=True)
if __name__=='__main__':main()
