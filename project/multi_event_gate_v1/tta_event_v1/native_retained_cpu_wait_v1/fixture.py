"""Actual isolated CPU tensors remain unchanged while their process waits."""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
import torch

root=Path(sys.argv[1]);torch.set_num_threads(1)
torch.manual_seed(17);np.random.seed(17)
x=torch.randn(32,32);a=np.random.randn(64)
def fingerprint():
 return dict(tensor=hashlib.sha256(x.numpy().tobytes()).hexdigest(),numpy=hashlib.sha256(a.tobytes()).hexdigest(),
  torch_RNG=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
  numpy_RNG=hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest(),CUDA_initialized=torch.cuda.is_initialized())
before=fingerprint();assert not before['CUDA_initialized']
spec=json.loads((root/'protocol.json').read_text())
ready=dict(passed=True,cuda_initialized=False,os_threads=len(list(Path('/proc/self/task').iterdir())),source_sha256=spec['source_sha256'],CPU_only_fixture=True)
assert ready['os_threads']==1
(root/'CPU_before.json').write_text(json.dumps(before))
(root/'ready.tmp').write_text(json.dumps(ready));(root/'ready.tmp').rename(root/'source_bound_CPU_preload_proof.json')
began=time.monotonic()
while not(root/'finish_CPU_wait.json').exists():
 assert time.monotonic()-began<45
 time.sleep(.02)
after=fingerprint();assert after==before
(root/'CPU_after.json').write_text(json.dumps(after))
