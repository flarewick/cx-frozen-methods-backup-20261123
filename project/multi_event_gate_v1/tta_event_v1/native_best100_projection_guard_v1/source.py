"""Original source/queue ownership, with only local projection abstention."""
from native_live_response_source_v1.source import SourceStream as Original
from .local import LocalFeedbackStream

class SourceStream(Original):
 def __init__(self, *args, **kwargs):
  super().__init__(*args, **kwargs)
  if self.local is not None:
   local = LocalFeedbackStream.__new__(LocalFeedbackStream)
   local.__dict__.update(self.local.__dict__)
   self.local = local
