"""Overlap ordinary planner initialization with rendering; captures stay exclusive."""
from contextlib import contextmanager
import functools
import hashlib
import inspect
import json
from pathlib import Path
import threading
D=Path(__file__).resolve().parent

@contextmanager
def ordinary_constructor(gate):
    tid=threading.get_ident()
    with gate.condition:
        if gate.writer!=tid or gate.depth<1:raise RuntimeError('Planner initialization must originate in exclusive scene setup')
        depth=gate.depth;gate.writer=None;gate.depth=0;gate.condition.notify_all()
    try:yield
    finally:
        # The outer setup retains its exact original writer depth even on error.
        with gate.write():
            with gate.condition:gate.depth+=depth

def protect_graphs(cuda,gate,owner):
    original_context=cuda.graph;graph_type=cuda.CUDAGraph
    original_begin=graph_type.capture_begin;original_end=graph_type.capture_end;active={}
    def exclusive():
        if threading.get_ident()!=owner:raise RuntimeError('CUDA capture escaped planner owner')
        context=gate.write();context.__enter__()
        try:cuda.synchronize()
        except BaseException:context.__exit__(None,None,None);raise
        return context
    @functools.wraps(original_begin)
    def begin(graph,*args,**kwargs):
        if id(graph)in active:raise RuntimeError('Nested CUDA graph capture')
        context=exclusive()
        try:result=original_begin(graph,*args,**kwargs)
        except BaseException:context.__exit__(None,None,None);raise
        active[id(graph)]=context;return result
    @functools.wraps(original_end)
    def end(graph,*args,**kwargs):
        context=active.pop(id(graph),None)
        if context is None:raise RuntimeError('Unprotected CUDA graph capture end')
        try:return original_end(graph,*args,**kwargs)
        finally:context.__exit__(None,None,None)
    class ExclusiveGraph(original_context):
        def __enter__(self):
            self._cx_capture_context=exclusive()
            try:return super().__enter__()
            except BaseException:self._cx_capture_context.__exit__(None,None,None);raise
        def __exit__(self,*args):
            try:return super().__exit__(*args)
            finally:self._cx_capture_context.__exit__(None,None,None)
    graph_type.capture_begin=begin;graph_type.capture_end=end;cuda.graph=ExclusiveGraph
    def restore():
        if active:raise RuntimeError('CUDA graph capture still active')
        graph_type.capture_begin=original_begin;graph_type.capture_end=original_end;cuda.graph=original_context
    return restore

def install(cls,c):
    import torch
    proof=json.loads((D/'reports/target_collection_capture_gate_cpu_20261024.json').read_text())
    if not proof['passed']or proof['helper_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise RuntimeError('Capture isolation CPU proof stale')
    if hashlib.sha256(Path(inspect.getsourcefile(cls)).read_bytes()).hexdigest()!=proof['original_planner_sha256']:raise RuntimeError('Original planner source changed')
    if hashlib.sha256(inspect.getsource(torch.cuda.graph).encode()).hexdigest()!=proof['torch_graph_context_sha256']:raise RuntimeError('Original CUDA graph context changed')
    from collection_gc_schedule import ConstructorGC
    gc_proof=json.loads((D/'reports/target_collection_gc_schedule_cpu_20261024.json').read_text())
    if not gc_proof['passed']or gc_proof['helper_sha256']!=hashlib.sha256((D/'collection_gc_schedule.py').read_bytes()).hexdigest():raise RuntimeError('Constructor GC CPU proof stale')
    collector=ConstructorGC(c.service.owner,inspect.getsourcefile(torch.cuda.graph),c.note);collector.install()
    restore=protect_graphs(torch.cuda,c.scene_gate,c.service.owner)
    original=cls.__init__
    @functools.wraps(original)
    def create(self,*args,**kwargs):
        with collector.constructor():
            with ordinary_constructor(c.scene_gate):return original(self,*args,**kwargs)
    cls.__init__=create
    c.note('capture_isolation_installed',global_capture_settings_unchanged=True)
    return restore
