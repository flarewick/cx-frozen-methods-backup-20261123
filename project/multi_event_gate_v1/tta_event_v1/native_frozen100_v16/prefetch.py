"""Submit exact q0 RGB/own commands/language early; original CPU inference unchanged."""
import time
from .semantic import SemanticClient as Original,content_key
from .common import write,sha
class PrefetchClient(Original):
 def __init__(self,root):
  super().__init__(root);self.prefetched=set()
 def register_verified(self,row):
  from .common import read
  assert set(row)=={'frame_file','frame_sha256','instruction','content_key'}
  req=self.root/'requests'/(row['content_key']+'.json')
  assert read(req)==row and (self.root/'answers'/(row['content_key']+'.json')).exists()
  with self.lock:self.prefetched.add(row['content_key'])
 def submit(self,observed,instruction):
  import numpy as np
  key=content_key(observed['rgb'],observed['command_state'],instruction)
  with self.lock:
   req=self.root/'requests'/(key+'.json');frame=self.root/'requests'/(key+'.npz')
   if not req.exists():
    frame.parent.mkdir(parents=True,exist_ok=True)
    with frame.open('xb')as f:np.savez_compressed(f,**{'cam'+str(i):im for i,im in enumerate(observed['rgb'])},command_state=observed['command_state'])
    write(req,dict(frame_file=str(frame),frame_sha256=sha(frame),instruction=instruction,content_key=key))
   self.prefetched.add(key)
  return key
 def at_initial(self,observed,instruction,*,query,tick):
  key=content_key(observed['rgb'],observed['command_state'],instruction)
  with self.lock:assert key in self.prefetched,'Prefetch does not match actual policy q0 bytes'
  return super().at_initial(observed,instruction,query=query,tick=tick)
