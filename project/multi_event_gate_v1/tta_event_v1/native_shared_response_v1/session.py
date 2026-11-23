"""Episode-local packet lease, fast-state rollback, trust and shadow application.

Random interface weights are usable ONLY in shadow mode. Actual actor application
requires an immutable independently qualified, direction-trained manifest.
"""
from pathlib import Path
import copy,hashlib,json
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_sha(state):
 h=hashlib.sha256()
 for name,value in sorted(state.items()):
  h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def qualified(model,path):
 if path is None:raise ValueError('Actual actor updates require a trained qualification manifest')
 path=Path(path);T=Path(__file__).resolve().parent.parent
 if not path.resolve().is_relative_to((T/'outputs').resolve()):raise ValueError('Qualification must belong to isolated TTA outputs')
 v=json.loads(path.read_text());assert v['frozen']and v['direction_trained']and v['independent_feedback_gate_passed']
 for key in('independent_feedback_report','training_checkpoint','training_protocol'):
  p=Path(v[key]);assert sha(p)==v[key+'_sha256']
 report=json.loads(Path(v['independent_feedback_report']).read_text())
 assert report['complete']and report['independent_feedback_gate_passed']
 assert v['response_state_sha256']==state_sha(model.response.state_dict())
 for p,h in v['source_sha256'].items():assert sha(p)==h
 for p in(Path(__file__),Path(__file__).parent/'model.py'):
  assert str(p)in v['source_sha256']and v['source_sha256'][str(p)]==sha(p)
 return sha(path)

class FeedbackSession:
 """Queue-compatible owner. Each released Feedback object is consumed once."""
 def __init__(self,model,*,queue=None,shadow_only=True,qualification_manifest=None):
  from native_phase_feedback_v1.feedback_queue import FeedbackQueue
  self.model=copy.deepcopy(model).eval()
  self.model.response.requires_grad_(False)
  self.shadow_only=bool(shadow_only)
  self.qualification_sha256=None if self.shadow_only else qualified(self.model,qualification_manifest)
  self.queue=FeedbackQueue()if queue is None else queue
  self._clear()
 def _clear(self):
  self.fast=tuple(v.detach()for v in self.model.zero_fast());self.anchors=[]
  self._lease=None;self.attempts=0;self.accepted=0;self.closed=False;self.records=[]
 def __getattr__(self,name):return getattr(object.__getattribute__(self,'queue'),name)
 def reset(self):
  self.queue.reset();self._clear()
 def at_query(self,*,query,tick,query_proof_sha256):
  if self.closed:raise ValueError('Closed episode/branch')
  if self._lease is not None:raise ValueError('A released feedback lease has not been consumed')
  self._lease=self.queue.at_query(query=query,tick=tick,query_proof_sha256=query_proof_sha256)
  return self._lease
 def _direction(self,hidden,logits,reference):
  ok,_=self.model.trust(hidden,logits,reference,self.fast,self.anchors)
  return self.model.direction(hidden,logits,reference,self.fast)if ok else reference
 def consume(self,packet,*,row_ids,hidden,context,command,kind,kernel_prediction,target,
             next_hidden,next_logits,next_reference):
  if self.closed or packet is None or packet is not self._lease:raise ValueError('Stale, copied or cross-session feedback lease')
  if self.attempts>=3:raise ValueError('Frozen maximum three source packets')
  if tuple(row_ids)!=packet.rows or not row_ids:raise ValueError('Exact nonempty arrived packet rows required')
  if not(len(row_ids)==len(hidden)==len(context)==len(command)==len(kind)==len(target)):raise ValueError('Feedback/query binding count differs')
  self._lease=None;self.attempts+=1
  before=tuple(v.detach().clone()for v in self.fast)
  digest=lambda fast:state_sha(dict(gain=fast[0],offset=fast[1]))
  report=dict(query=packet.query,source_query=packet.source_query,observation_tick=packet.observation_tick,
    rows=len(row_ids),fast_before=digest(before),shadow_only=self.shadow_only,applied_to_actor=False,
    accepted=False,qualification_sha256=self.qualification_sha256)
  try:
   with torch.enable_grad():
    proposed,inner=self.model.two_steps(hidden,context,command,kind,kernel_prediction,target,fast=before,online=True)
   with torch.no_grad():
    oldloss=float(((self.model(hidden,context,command,kind,kernel_prediction,before)-target)/.003125).square().mean())
    newloss=float(((self.model(hidden,context,command,kind,kernel_prediction,proposed)-target)/.003125).square().mean())
   ok,trust=self.model.trust(next_hidden,next_logits,next_reference,proposed,self.anchors)
   report.update(inner=inner,trust=trust,response_loss_before=oldloss,response_loss_after=newloss)
   changed=digest(proposed)!=digest(before)
   if ok and changed and newloss<oldloss:
    self.fast=tuple(v.detach().clone()for v in proposed);self.accepted+=1
    self.anchors=[tuple(v.detach().clone()for v in(next_hidden,next_logits,next_reference)),*self.anchors][:2]
    report.update(accepted=True,reason='response_and_trust_passed_not_task_benefit')
   else:
    self.fast=before;report['reason']='trust_noop_or_response_rejected_rollback'
  except (ValueError,RuntimeError)as exc:
   self.fast=before;report.update(reason='invalid_feedback_or_gradient_rollback',error_type=type(exc).__name__,error=str(exc))
  output=next_reference if self.shadow_only else self._direction(next_hidden,next_logits,next_reference)
  report['applied_to_actor']=bool(not self.shadow_only and not torch.equal(output,next_reference))
  report.update(fast_after=digest(self.fast),attempts=self.attempts,accepted_updates=self.accepted)
  self.records.append(report)
  return output,report
 def finish(self):
  result=self.queue.finish();records=copy.deepcopy(self.records)
  self._clear();self.closed=True
  return dict(queue=result,updates=records,fast_reset_on_close=True,shadow_only=self.shadow_only)
