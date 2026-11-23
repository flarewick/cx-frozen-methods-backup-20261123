"""Coalesce only Torch graph-entry full GC during one genuine planner constructor."""
from contextlib import contextmanager
import gc
from pathlib import Path
import sys,threading,time
class ConstructorGC:
    def __init__(self,owner,graph_file,note=lambda *a,**k:None):
        self.owner=owner;self.graph_file=str(Path(graph_file).resolve());self.original=gc.collect;self.depth=0;self.note=note;self.skipped=0;self.seconds=0.
    def collect(self,*args,**kwargs):
        caller=sys._getframe(1)
        if self.depth and threading.get_ident()==self.owner and caller.f_code.co_name=='__enter__'and str(Path(caller.f_code.co_filename).resolve())==self.graph_file:
            self.skipped+=1;return 0
        began=time.monotonic()
        try:return self.original(*args,**kwargs)
        finally:self.seconds+=time.monotonic()-began
    @contextmanager
    def constructor(self):
        if threading.get_ident()!=self.owner or self.depth:raise RuntimeError('Unexpected planner GC owner/nesting')
        before=self.seconds;skipped=self.skipped;self.collect();self.depth=1;began=time.monotonic()
        try:yield
        finally:
            self.depth=0;self.collect();self.note('planner_gc_schedule',constructor_seconds=round(time.monotonic()-began,3),full_gc_seconds=round(self.seconds-before,3),graph_full_gc_coalesced=self.skipped-skipped)
    def install(self):
        if gc.collect!=self.original:raise RuntimeError('Python GC was already patched')
        gc.collect=self.collect
    def restore(self):
        if self.depth:raise RuntimeError('Planner constructor still active')
        gc.collect=self.original
