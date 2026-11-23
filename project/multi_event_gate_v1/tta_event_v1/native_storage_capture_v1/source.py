"""Pure CPU fingerprints of already-computed source outputs; no extra forward."""
import hashlib
import numpy as np
from native_policy_execution_v1.common import write
def tracked(original,folder):
 class RecordedSource(original):
  def finish(self):
   value=super().finish()
   rows=[]if self.owner.memory is None else self.owner.memory.rows
   assert len(rows)==self.owner.actual_tracking_calls
   values=[]
   for r in rows:
    row={k:r[k]for k in ['global_tick','elapsed_episode_s','kind','RGB_sha256','raw_logits_sha256']}
    for name in ['masks','presence']:
     a=np.asarray(r[name]);row[name]=dict(dtype=a.dtype.str,shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
    values.append(row)
   write(folder/('tracking_equivalence_source%03d.json'%self.source_query),dict(source_query=self.source_query,
    known_seed=self.owner.seed is not None,actual_tracking_calls=self.owner.actual_tracking_calls,rows=values,no_extra_model_forward=True))
   return value
 return RecordedSource
