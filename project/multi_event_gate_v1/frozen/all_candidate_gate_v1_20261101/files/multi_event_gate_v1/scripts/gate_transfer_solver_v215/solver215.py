"""Cache only original solver availability; full original TOPP algorithm retained."""
import sys,os,threading,functools
INSTALLED=False
class AvailabilityCache:
 def __init__(self,original):self.original=original;self.lock=threading.RLock();self.key=None;self.value=None;self.evaluations=0;self.hits=0
 def environment(self):
  paths=tuple(sys.path);signatures=[]
  for path in paths:
   try:
    st=os.stat(path or os.getcwd());signature=(st.st_dev,st.st_ino,st.st_mode,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
   except OSError as e:signature=(type(e).__name__,e.errno)
   signatures.append(signature)
  return(paths,os.getcwd(),tuple(signatures),tuple(id(f)for f in sys.meta_path),tuple((name,id(sys.modules.get(name)))for name in ('ecos','qpoases','cvxpy')))
 def __call__(self,output_msg=True):
  with self.lock:
   key=self.environment()
   if key!=self.key:
    value=self.original(output_msg=False);self.evaluations+=1
    assert isinstance(value,tuple)and all(isinstance(row,tuple)and len(row)==2 for row in value)
    self.value=value;self.key=self.environment()
   else:self.hits+=1
   result=tuple(list(self.value))
  if output_msg:print(result)
  return result

def install():
 global INSTALLED
 if INSTALLED:return
 import toppra.solverwrapper as exports
 import importlib
 module=importlib.import_module('toppra.solverwrapper.solverwrapper')
 assert exports.available_solvers is module.available_solvers
 original=module.available_solvers;cache=AvailabilityCache(original)
 @functools.wraps(original)
 def available(output_msg=True):return cache(output_msg)
 module.available_solvers=exports.available_solvers=available
 module._resource_cache215=cache;INSTALLED=True
