"""One required model transfer: original Module.to, queued copies, one completion fence."""
import time

def install_one(module, *, target, synchronize, emit):
 original=module.to
 previous=module.__dict__.get('to');had_instance='to' in module.__dict__
 def restore():
  if had_instance:module.to=previous
  else:module.__dict__.pop('to',None)
 def transfer(*args,**kwargs):
  # Runtime has exactly these two original calls; reject any precision/layout change.
  assert args==(target,) and not kwargs, 'Unexpected original model transfer'
  started=time.monotonic()
  try:
   value=original(*args,non_blocking=True)
   assert value is module
   synchronize()
  finally:restore()
  emit(dict(original_Module_to_once=True,target=target,non_blocking=True,
            original_dtype_layout_and_parameter_values=True,
            one_completion_fence_before_return=True,original_method_restored=True,
            seconds=time.monotonic()-started,no_policy_query_or_extra_compute=True))
  return value
 module.to=transfer
 return restore

def install(prepared, root):
 import torch
 from native_policy_execution_v1.common import write
 assert not torch.cuda.is_initialized()
 for name,model in [('VLA',prepared['base']._model),('SAM2',prepared['mask_model'])]:
  assert all(v.device.type=='cpu' for v in model.state_dict().values())
  install_one(model,target='cuda',synchronize=torch.cuda.synchronize,
              emit=lambda row,name=name:write(root/('model_transfer_'+name+'.json'),row))
