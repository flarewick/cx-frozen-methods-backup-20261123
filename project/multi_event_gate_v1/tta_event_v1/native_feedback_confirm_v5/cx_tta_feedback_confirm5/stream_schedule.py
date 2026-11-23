"""Fixed four lanes; first two bridges gate acceptance and remaining admission."""
from .task_affinity import make_queues,enqueue,assigned_task
class StreamSchedule:
 def __init__(self,scenes,initial,bridge_count=2):
  self.scenes=scenes;self.initial=initial;self.bridge_count=max(0,bridge_count-initial)
  self.queues=make_queues();self.tail_admitted=False
  self.first_end=min(initial+4,len(scenes))if self.bridge_count else len(scenes)
  for i in range(initial,self.first_end):enqueue(self.queues,scenes,i)
  if not self.bridge_count or self.first_end==len(scenes):self._finish()
 def _finish(self):
  for q in self.queues.values():
   for _ in range(2):q.put(None)
  self.tail_admitted=True
 def admit_after_bridges(self,proof,nonce,protocol):
  if self.tail_admitted:return False
  if(proof.get('nonce')!=nonce or proof.get('protocol_identity')!=protocol
      or proof.get('bridges')!=self.bridge_count):raise ValueError('CPU compatibility gate identity mismatch')
  for i in range(self.first_end,len(self.scenes)):enqueue(self.queues,self.scenes,i)
  self._finish();return True
