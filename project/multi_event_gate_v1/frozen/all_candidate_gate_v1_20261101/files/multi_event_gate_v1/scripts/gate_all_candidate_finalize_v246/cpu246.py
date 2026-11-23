"""Exact original finalizer after live host verified the completed GPU has exited."""
import sys,os
from pathlib import Path
import torch
assert not torch.cuda.is_initialized()and not list(Path('/dev').glob('nvidia*'))
intent=Path(sys.argv[1]);os.environ['CX_BENEFIT_RECYCLE_FILE']=str(intent.parent/'memory_recycle.json')
import benefit_growth_runtime as runtime
runtime.install()
import benefit_growth_deferred_finalize as deferred
deferred.finalize(intent)
