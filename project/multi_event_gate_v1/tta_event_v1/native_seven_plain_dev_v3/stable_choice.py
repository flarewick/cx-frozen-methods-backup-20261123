"""Require a genuinely quiet empty-card interval before CPU preload; no lock or GPU reservation."""
class StableEmpty:
 def __init__(self,seconds=20):assert seconds>=2;self.seconds=seconds;self.first={}
 def update(self,fleet,now):
  eligible={r['uuid']for r in fleet['cards']if fleet['own_gpu_count']<3 and r['available']and not r['pids']and r['memory_mib']<=64 and r['sm']<=5}
  self.first={u:t for u,t in self.first.items()if u in eligible}
  for uuid in eligible:self.first.setdefault(uuid,now)
  return {uuid for uuid,started in self.first.items()if now-started>=self.seconds}
