"""Defer optional PCD dependency; original RGB/pickle/camera code stays unchanged."""
import ast
import importlib
import importlib.util
import sys
import threading
import types

class DeferredOpen3D(types.ModuleType):
    def __init__(self,spec):
        super().__init__('open3d');self.__spec__=spec;self.__package__='open3d';self.__loader__=spec.loader
        self._cx_lock=threading.RLock();self._cx_real=None
    def __getattr__(self,name):
        if name.startswith('_cx_'):raise AttributeError(name)
        with self._cx_lock:
            if self._cx_real is None:
                if sys.modules.get('open3d')is self:del sys.modules['open3d']
                try:self._cx_real=importlib.import_module('open3d')
                except BaseException:sys.modules['open3d']=self;raise
            return getattr(self._cx_real,name)

def install(runtime):
    # Bind the exact original call sites: camera never accesses o3d; only the
    # original explicit save_pcd helper may need it. Unexpected calls still load
    # the genuine package instead of faking any geometry/point-cloud operation.
    for path,allowed in ((runtime/'envs/camera/camera.py',set()),(runtime/'envs/utils/save_file.py',{'save_pcd'})):
        tree=ast.parse(path.read_text())
        def visit(node,owner=None):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):owner=node.name
            if isinstance(node,ast.Name)and node.id=='o3d'and owner not in allowed:raise RuntimeError('Open3D runtime use changed: '+str(path))
            for child in ast.iter_child_nodes(node):visit(child,owner)
        visit(tree)
    if 'open3d'in sys.modules:raise RuntimeError('Open3D already loaded before deferred optional import')
    spec=importlib.util.find_spec('open3d')
    if spec is None:raise ImportError('Original Open3D dependency missing')
    module=DeferredOpen3D(spec);sys.modules['open3d']=module
    return module
