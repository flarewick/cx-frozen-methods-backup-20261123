"""Release completed CPU allocations; live tensors remain owned and unchanged."""
from pathlib import Path
import ctypes,gc

def rss_bytes():
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.startswith('VmRSS:'):return int(line.split()[1])*1024
 raise RuntimeError('Actual process RSS unavailable')
def reclaim(completed_refs):
 if not completed_refs:raise ValueError('Actual completed ownership evidence required')
 gc.collect()
 if any(ref()is not None for ref in completed_refs):raise ValueError('A declared completed owner or tensor remains live')
 before=rss_bytes();libc=ctypes.CDLL(None,use_errno=True)
 trim=libc.malloc_trim;trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int
 result=trim(0);after=rss_bytes()
 return dict(completed_references=len(completed_refs),all_completed_refs_dead=True,RSS_before_bytes=before,RSS_after_bytes=after,malloc_trim_result=result,observed_RSS_reduction_bytes=before-after,active_allocations_not_freed=True,GPU_exit_qualification=False)
