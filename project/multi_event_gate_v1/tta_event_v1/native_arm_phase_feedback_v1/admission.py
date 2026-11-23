"""At most3 real nonoverlapping feedback sources; frozen action gate unchanged."""
import numpy as np
from stage_feedback_v3.completion import KINDS
from .completion import ARM
from .readouts import KIND

def choose(pool,query,selected,active,*,seed_available,initial,commands):
 if not seed_available:return None,'initial_semantics_unavailable'
 original=None
 if active and selected is not None:
  kinds=selected['kinds']
  if len(kinds)==1:original=kinds[0]
  elif len(kinds)==2 and set(kinds)<=set(KINDS)and len({KINDS[k][0]for k in kinds})==2:original=tuple(kinds)
 kind=original
 if query==0 and kind is None:
  p=np.r_[np.asarray(initial)[None,ARM],np.asarray(commands)[:,ARM]]
  if np.linalg.norm(np.diff(p,axis=0),axis=1).sum()>1e-6:kind=KIND
 if kind is None:return None,'no_new_requested_source'
 if len(pool.queue.events)>=3:return None,'original_max3_feedback_source_budget'
 if any(query in(source,source+1)for source in pool.live):return None,'prior_original_window_still_live'
 return kind,'original_selected_event'if original is not None else'initial_actual_own_ARM_commands'
