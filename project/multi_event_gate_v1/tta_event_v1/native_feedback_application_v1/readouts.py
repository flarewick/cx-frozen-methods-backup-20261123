from native_best100_endpoint_v1.readouts import CurrentReadouts as Original
from .completion import CompletionQueue
class CurrentReadouts(Original):
 def __init__(self,recorder,reader,*,kind=None,initial=None,base=None,on_arrival=None):
  if kind is None or isinstance(kind,str):return super().__init__(recorder,reader,kind=kind,initial=initial,base=base,on_arrival=on_arrival)
  super().__init__(recorder,reader,kind=None,initial=initial,base=base,on_arrival=on_arrival)
  self.completion=CompletionQueue();self.completion.register('event','origin',kind,initial['command_state'][[6,13]],base,decision_tick=0,frame_id=self.frame(initial))
