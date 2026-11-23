"""Preload ordinary original MotionGen dependencies, never its CUDA tensor defaults."""
import ast
import hashlib
from pathlib import Path
import time

def preload(runtime_source):
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU dependencies require no CUDA context')
    path=Path(runtime_source)/'envs/curobo/src/curobo/wrap/reacher/motion_gen.py'
    tree=ast.parse(path.read_text());namespace=dict(__name__='cx_cpu_imports',__package__='curobo.wrap.reacher');imports=[]
    for node in tree.body:
        if not isinstance(node,(ast.Import,ast.ImportFrom)):continue
        began=time.monotonic()
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        if torch.cuda.is_initialized():raise RuntimeError('Dependency unexpectedly initialized CUDA: '+ast.unparse(node))
        imports.append(dict(statement=ast.unparse(node),seconds=round(time.monotonic()-began,4)))
    # Common scene math/geometry imports are also ordinary CPU modules.
    import gymnasium,transforms3d,trimesh,toppra,imageio
    if torch.cuda.is_initialized():raise RuntimeError('Scene dependency initialized CUDA')
    return dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),imports=imports,cuda_initialized=False)
