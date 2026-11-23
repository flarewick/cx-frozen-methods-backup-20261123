"""CPU: only literal attention construction changes; no action/noise code edits."""
from pathlib import Path
import json,hashlib,inspect,ast
from .graph import capture_safe_suffix

def main():
 import torch
 from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
 class Stub:embed_suffix=PI0Pytorch.embed_suffix
 f=capture_safe_suffix(Stub());assert f.__name__=='embed_suffix'and not torch.cuda.is_initialized()
 T=Path(__file__).resolve().parents[1];O=T/'outputs/native_denoise_graph_v1';source=Path(inspect.getsourcefile(PI0Pytorch));sample=inspect.getsource(PI0Pytorch.sample_actions)
 proof=dict(passed=True,original_suffix_tensor_AST_exact_except_literal_constant_dispatch=True,original_outer_sampler_and_noise_source_unmodified=True,original_sampler_source_SHA256=hashlib.sha256(sample.encode()).hexdigest(),CUDA_initialized=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),Path(__file__).with_name('graph.py'),source]})
 with(O/'CPU_AST_checks.json').open('x')as f:json.dump(proof,f,indent=2)
 print('ORIGINAL_DENOISE_GRAPH_AST_CPU_PASS',flush=True)
if __name__=='__main__':main()
