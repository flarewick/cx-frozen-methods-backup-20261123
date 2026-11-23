"""Explicit lane RNG isolation; native Torch sampler runs under one shared lock.

NumPy/Python calls keep their original algorithms via independent RandomState/
Random instances. Unregistered compiler/library threads use the original APIs.
Only setup/planning/probe/predict may touch Torch RNG, inside torch_scope().
"""
from contextlib import contextmanager
import copy
import functools
import random
import threading


class Lane:
    def __init__(self,name,seed=0):
        import numpy as np
        self.name=name
        self.numpy=np.random.RandomState(seed)
        self.python=random.Random(seed)
        self.torch_cpu=None;self.torch_cuda=None


class RngRouter:
    def __init__(self,cuda=False):
        self.local=threading.local();self.lock=threading.RLock();self.cuda=cuda;self.originals=[]

    def current(self):return getattr(self.local,'lane',None)

    def install(self):
        import numpy as np
        if self.originals:raise RuntimeError('RNG routing already installed')
        for module,attr,cls in ((np.random,'numpy',np.random.RandomState),(random,'python',random.Random)):
            # Forward only actual RNG methods, retaining utility/type APIs.
            for name in dir(cls):
                if name.startswith('_') or not callable(getattr(cls,name,None)) or not callable(getattr(module,name,None)):continue
                original=getattr(module,name)
                def make(original=original,name=name,attr=attr):
                    @functools.wraps(original)
                    def dispatch(*args,**kwargs):
                        lane=self.current()
                        return (getattr(getattr(lane,attr),name) if lane else original)(*args,**kwargs)
                    return dispatch
                self.originals.append((module,name,original));setattr(module,name,make())

    def uninstall(self):
        for module,name,original in reversed(self.originals):setattr(module,name,original)
        self.originals.clear()

    @contextmanager
    def bind(self,lane):
        before=self.current()
        if before is not None and before is not lane:raise RuntimeError('Cross-lane nested binding')
        self.local.lane=lane
        try:yield
        finally:self.local.lane=before

    @contextmanager
    def torch_scope(self,lane):
        import torch
        if self.current() is not lane:raise RuntimeError('Torch call outside its RNG lane')
        depth=getattr(self.local,'torch_depth',0)
        if depth:
            self.local.torch_depth=depth+1
            try:yield
            finally:self.local.torch_depth=depth
            return
        with self.lock:
            outer_cpu=torch.get_rng_state().clone()
            outer_cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if self.cuda else []
            if lane.torch_cpu is None:
                lane.torch_cpu=outer_cpu.clone();lane.torch_cuda=[s.clone() for s in outer_cuda]
            torch.set_rng_state(lane.torch_cpu)
            if self.cuda:torch.cuda.set_rng_state_all(lane.torch_cuda)
            self.local.torch_depth=1
            try:yield
            finally:
                try:
                    if self.cuda:torch.cuda.synchronize()
                    lane.torch_cpu=torch.get_rng_state().clone()
                    lane.torch_cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if self.cuda else []
                finally:
                    torch.set_rng_state(outer_cpu)
                    if self.cuda:torch.cuda.set_rng_state_all(outer_cuda)
                    self.local.torch_depth=0

    def capture(self,lane):
        import torch
        if not getattr(self.local,'torch_depth',0):raise RuntimeError('Capture requires Torch scope')
        return dict(numpy=copy.deepcopy(lane.numpy.get_state()),python=lane.python.getstate(),
                    torch_cpu=torch.get_rng_state().clone(),
                    torch_cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if self.cuda else [])

    def restore(self,lane,state):
        import torch
        if not getattr(self.local,'torch_depth',0):raise RuntimeError('Restore requires Torch scope')
        lane.numpy.set_state(copy.deepcopy(state['numpy']));lane.python.setstate(state['python'])
        torch.set_rng_state(state['torch_cpu'])
        if self.cuda:torch.cuda.set_rng_state_all(state['torch_cuda'])
