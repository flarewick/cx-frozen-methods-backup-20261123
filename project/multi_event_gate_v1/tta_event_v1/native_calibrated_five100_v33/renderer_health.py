"""Readonly fatal OIDN witness; never reuse images after a denoiser error."""
class LogMonitor:
 def __init__(self,path):self.path=path;self.position=0;self.suffix=b'';self.failed=False
 def check(self):
  with self.path.open('rb')as f:f.seek(self.position);value=f.read();self.position=f.tell()
  joined=self.suffix+value;self.suffix=joined[-32:]
  self.failed=self.failed or b'OIDN Error:'in joined
  return 'Native renderer reported OIDN error; batch invalid'if self.failed else None
