"""Publish complete immutable JSON using same-directory hard-link creation."""
from pathlib import Path
import os,json,tempfile

def write(p,x):
 from pi05_gt_resident_v1.common import O
 p=Path(p).resolve();assert p.is_relative_to(O.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 # Hidden .pending files are never scanned as final requests/answers. os.link preserves x-mode exclusivity.
 fd,name=tempfile.mkstemp(prefix='.'+p.name+'.',suffix='.pending',dir=p.parent)
 try:
  with os.fdopen(fd,'w')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
  os.link(name,p)
 finally:
  Path(name).unlink(missing_ok=True)

def install():
 import pi05_gt_resident_v1.common as resident
 import pi05_gt_paired_v1.common as paired
 import native_cross_task100_v19.common as cross
 import native_policy_execution_v1.common as policy
 resident.write=write;paired.write=write;cross.write=write;policy.write=write
