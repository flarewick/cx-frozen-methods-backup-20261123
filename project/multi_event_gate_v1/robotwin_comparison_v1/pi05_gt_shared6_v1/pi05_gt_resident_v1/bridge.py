"""Qualified response sessions return directions; actor execution is a separate witness."""
from pathlib import Path
import numpy as np
import torch
from .session import FeedbackSession
from .loader import qualified
from native_shared_response_v1.session import state_sha
from native_direction_training_v1.data import T,read,sha
from native_direction_training_v2.train import model_for
from native_response_binding_v1.bind import inputs,tensor,ARM

from .loader import load_model

class DirectionBridge:
 def __init__(self,model,qualification,*,mode,queue=None):
  if mode not in ('shadow','TTA'):raise ValueError('Explicit trained shadow or TTA mode required')
  self.qualification_sha256=qualified(model,qualification);self.mode=mode
  self.session=FeedbackSession(model,queue=queue,shadow_only=mode=='shadow',qualification_manifest=qualification)
 def step(self,*,current,rows_by_source,issuing_queries,predictor):
  reference=tensor(np.asarray(current['raw_delta'])[:,ARM][None]);h=tensor(current['hidden'][None]);logits=tensor(current['logits'][:,ARM][None])
  lease=self.session.at_query(query=current['query'],tick=current['tick'],query_proof_sha256=current['query_proof_sha256'])
  if lease is None:
   with torch.no_grad():output=reference if self.mode=='shadow'else self.session._direction(h,logits,reference)
   report=dict(reason='no_new_packet',accepted=False)
  else:
   values=inputs(lease,rows=rows_by_source[lease.source_query],issuing_queries=issuing_queries,current=current,predictor=predictor)
   output,report=self.session.consume(lease,**values)
   report=dict(report);report['session_would_apply_direction']=report.pop('applied_to_actor')
  value=output.detach().cpu().numpy()[0].copy()
  report.update(mode=self.mode,query=current['query'],qualification_sha256=self.qualification_sha256,returned_direction_changed=not torch.equal(output,reference),returned_max_shift=float((output-reference).abs().max()),scope='ARM direction composition; actor execution must be witnessed separately')
  return value,report
 def finish(self):return self.session.finish()
