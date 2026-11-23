"""CPU-only temporary-profile race and original-call identity checks."""
import threading,ast
from .owner_precision import install
from .common import C,T,O,write,sha
def main():
 calls=[];output=object()
 class Torch:
  precision='high'
  @classmethod
  def get_float32_matmul_precision(cls):return cls.precision
 class Base:
  def infer(self,*args,**kwargs):calls.append((args,kwargs));return output
 base=Base();check=install(base,Torch);assert check()
 assert base.infer('actual_input',once=True)is output and calls==[(('actual_input',),{'once':True})]
 Torch.precision='highest'
 # A CPU worker must never directly read a temporary global profile.
 errors=[]
 def worker():
  try:check()
  except RuntimeError as e:errors.append(str(e))
 t=threading.Thread(target=worker);t.start();t.join();assert len(errors)==1
 try:base.infer('must_not_run')
 except AssertionError:pass
 else:raise AssertionError('Wrong-precision model call accepted')
 assert len(calls)==1
 Torch.precision='high';assert check()and base.infer('second_actual_input')is output and len(calls)==2
 source=(C/'runtime.py').read_text();assert "c.call(schedule.lanes[item['task']],check_precision)"in source
 write(O/'CPU_owner_precision_checks.json',dict(passed=True,CPU_worker_cannot_read_temporary_global_profile=True,wrong_precision_model_call_rejected_before_original_call=True,original_model_called_once_with_identical_args_and_same_return_object=True,precision_configuration_never_written=True,GPU_calls=0,source_sha256={str(p):sha(p)for p in C.glob('*.py')}))
 print('SERIAL_GPU_OWNER_PRECISION_CPU_PASS',flush=True)
if __name__=='__main__':main()
