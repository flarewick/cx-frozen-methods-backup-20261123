"""Actual CPU Module.to values/aliases/RNG, restoration and refusal checks; no CUDA claim."""
from pathlib import Path
import ast,copy,hashlib,json,os

def run():
 import torch
 from .transfer import install_one
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 torch.manual_seed(17)
 class M(torch.nn.Module):
  def __init__(self):
   super().__init__();self.a=torch.nn.Linear(19,13);self.b=torch.nn.Linear(13,7)
   self.register_buffer('counter',torch.arange(9,dtype=torch.int64));self.tied=self.a.weight
 m=M().eval().requires_grad_(False);other=copy.deepcopy(m);x=torch.randn(11,19)
 before=torch.random.get_rng_state().clone();pointers={k:v.data_ptr()for k,v in m.state_dict().items()}
 rows=[];fences=[];original=m.to
 install_one(m,target='cpu',synchronize=lambda:fences.append('actual_completion'),emit=rows.append)
 assert m.to('cpu')is m
 assert len(rows)==len(fences)==1 and 'to' not in m.__dict__ and m.to==original
 other.to('cpu')
 assert m.tied is m.a.weight and m.counter.dtype==torch.int64
 assert pointers=={k:v.data_ptr()for k,v in m.state_dict().items()}
 assert all(torch.equal(v,other.state_dict()[k])for k,v in m.state_dict().items())
 assert torch.equal(m.b(m.a(x)),other.b(other.a(x)))and torch.equal(before,torch.random.get_rng_state())
 assert not m.training and not any(p.requires_grad for p in m.parameters())
 called=[];old=m.to
 def fail(*a,**kw):called.append(kw);raise RuntimeError('actual original transfer failure')
 m.to=fail;install_one(m,target='cpu',synchronize=lambda:fences.append('wrong'),emit=rows.append)
 try:m.to('cpu')
 except RuntimeError:pass
 else:raise AssertionError('Original failure swallowed')
 assert m.to is fail and len(rows)==1 and len(fences)==1 and called==[dict(non_blocking=True)]
 m.to=old;restore=install_one(m,target='cpu',synchronize=lambda:None,emit=rows.append)
 try:m.to('cpu',dtype=torch.float64)
 except AssertionError:pass
 else:raise AssertionError('Unexpected precision change accepted')
 restore();assert m.to==old
 assert not torch.cuda.is_initialized()
 return dict(passed=True,CPU_only=True,actual_Module_to=True,values_aliases_buffers_RNG_exact=True,
             original_failure_propagates=True,original_method_restored=True,
             unexpected_precision_change_refused=True,GPU_transfer_performance_NOT_tested=True)
if __name__=='__main__':
 from native_policy_execution_v1.common import T,write,sha
 C=Path(__file__).resolve().parent;v=run();v['source_sha256']={str(p):sha(p)for p in C.glob('*.py')}
 write(T/'outputs/native_oneway_model_transfer_v1/CPU_contract_checks_v1.json',v)
 print('CPU_ORIGINAL_MODULE_TRANSFER_CONTRACTS_PASS',flush=True)
