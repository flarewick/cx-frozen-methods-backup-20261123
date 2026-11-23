"""Observe original owner calls without replacing queued functions or touching tensors."""
import threading,time,types
class OwnerProfile:
 def __init__(self,service):
  assert threading.get_ident()==service.owner
  self.service=service;self.owner=service.owner;self.original=service.execute
  self.stack=[];self.rows={};self.closed=False
  def observed(unused,lane,fn,args,kwargs):
   assert not self.closed and threading.get_ident()==self.owner
   key=(str(lane.name),getattr(fn,'__module__',''),getattr(fn,'__qualname__',type(fn).__name__),getattr(fn,'_cx_stage',''))
   frame=[time.perf_counter_ns(),0];self.stack.append(frame);failed=True
   try:
    result=self.original(lane,fn,args,kwargs);failed=False;return result
   finally:
    elapsed=time.perf_counter_ns()-frame[0];assert self.stack.pop()is frame
    if self.stack:self.stack[-1][1]+=elapsed
    exclusive=elapsed-frame[1];assert exclusive>=0
    row=self.rows.setdefault(key,dict(calls=0,exceptions=0,inclusive_host_ns=0,exclusive_host_ns=0,max_host_ns=0))
    row['calls']+=1;row['exceptions']+=int(failed);row['inclusive_host_ns']+=elapsed;row['exclusive_host_ns']+=exclusive;row['max_host_ns']=max(row['max_host_ns'],elapsed)
  self.wrapper=types.MethodType(observed,service);service.execute=self.wrapper
 def snapshot(self):
  assert threading.get_ident()==self.owner
  return dict(passed=False,bookkeeping_only=True,closed=self.closed,open_call_depth=len(self.stack),owner_thread=self.owner,host_wall_time_only_NOT_GPU_kernel_time=True,no_synchronize_tensor_read_or_extra_requests=True,
   rows=[dict(lane=k[0],module=k[1],function=k[2],stage=k[3],**v)for k,v in sorted(self.rows.items())])
 def close(self):
  assert not self.closed and not self.stack and threading.get_ident()==self.owner
  assert self.service.execute is self.wrapper
  self.service.execute=self.original;self.closed=True
  value=self.snapshot();value.update(passed=True,bookkeeping_only=False);return value
