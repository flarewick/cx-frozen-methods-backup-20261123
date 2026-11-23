"""Qualified response sessions return directions; actor execution is a separate witness."""
from pathlib import Path
import numpy as np
import torch
from native_shared_response_v1.session import FeedbackSession,qualified,state_sha
from native_direction_training_v1.data import T,read,sha
from native_direction_training_v2.train import model_for
from native_response_binding_v1.bind import inputs,tensor,ARM

def load_model(file):
 file=Path(file).resolve()
 if not file.is_relative_to((T/'outputs').resolve()):raise ValueError('Isolated trained qualification required')
 q=read(file)
 assert q['frozen']and q['actual_training_host_exit0']and not q['awaiting_actual_host_exit0']and q['actual_updates']==1000
 assert sha(q['training_host_exit_file'])==q['training_host_exit_sha256']and read(q['training_host_exit_file'])['exit_code']==0
 assert sha(q['training_checkpoint'])==q['training_checkpoint_sha256']and sha(q['training_result_file'])==q['training_result_sha256']
 result=read(q['training_result_file']);assert result['complete']and result['actual_updates']==1000
 model,_,_=model_for(read(T/'outputs/native_feedback_confirm_v5/direction_training_authorization.json'))
 payload=torch.load(q['training_checkpoint'],map_location='cpu',weights_only=True)
 assert payload['step']==1000 and payload['protocol_sha256']==q['training_protocol_sha256']
 model.response.load_state_dict(payload['response'],strict=True)
 for name,value in model.named_buffers():assert torch.equal(value,payload['buffers'][name])
 assert state_sha(model.response.state_dict())==q['response_state_sha256']
 qualified(model,file)
 return model.eval(),q

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
