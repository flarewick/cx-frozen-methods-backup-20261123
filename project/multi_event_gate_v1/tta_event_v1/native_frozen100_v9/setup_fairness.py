"""Original setup joins ordinary requests' priority1 FIFO; no work changes."""
import json

def install(c,output):
 original=c.service.call
 def call(lane,fn,*args,**kwargs):
  if getattr(fn,'_cx_stage',None)=='setup':
   before=getattr(fn,'_cx_priority',1)
   assert before==5,'Original setup priority changed unexpectedly'
   fn._cx_priority=1
   row=dict(stage='setup',original_priority=5,effective_priority=1,ordinary_FIFO_sequence_unchanged=True,original_function_and_args_preserved=True,no_extra_calls_or_GPU_compute=True)
   output.parent.mkdir(parents=True,exist_ok=True)
   with output.open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  return original(lane,fn,*args,**kwargs)
 c.service.call=call
 return call
