"""Check the original precision only at serialized GPU-owner boundaries."""
import threading
def install(base,torch):
 owner=threading.get_ident();original=base.infer
 def check():
  if threading.get_ident()!=owner:raise RuntimeError('Precision audit escaped actual GPU owner')
  assert torch.get_float32_matmul_precision()=='high','Native precision differs from byte-parity probe'
  return True
 def infer(*args,**kwargs):
  check();return original(*args,**kwargs)
 base.infer=infer
 return check
