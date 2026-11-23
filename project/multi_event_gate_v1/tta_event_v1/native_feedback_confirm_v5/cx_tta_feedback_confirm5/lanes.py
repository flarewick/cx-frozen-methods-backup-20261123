"""Four CPU simulation lanes, one frozen encoder owner; no extra GPU process."""
import ast
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import textwrap
import threading
import weakref
from concurrent_rng import RngRouter,Lane
from model_service import ModelService
from scene_gate import SceneGate


def gripper_only(original):
    source=ast.parse(textwrap.dedent(inspect.getsource(original.plan_grippers)))
    if any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self' for n in ast.walk(source)):
        raise ValueError('Original gripper interpolation requires planner state')
    class RecordedQposOnly(original):
        def __init__(self,*args,**kwargs):pass
        def __getattribute__(self,name):
            if name in ('__class__','__dict__','plan_grippers'):return object.__getattribute__(self,name)
            raise RuntimeError('Recorded-qpos replay must not request mutable expert planner state: '+name)
        def __setattr__(self,name,value):raise RuntimeError('Recorded-qpos planner is stateless')
    return RecordedQposOnly


class Coordinator:
    def __init__(self,cuda=False,native=None,native_step=None):
        self.rng=RngRouter(cuda);self.scene_gate=SceneGate();self.cancel=threading.Event()
        self.native=native;self.native_step=native_step;self.service=ModelService(self);self.task_lifetimes=[];self.scene_lifetimes=[]
    def note(self,stage,**fields):print('LANE_EVENT',json.dumps(dict(stage=stage,**fields)),flush=True)
    def check_cancel(self):
        if self.cancel.is_set():raise InterruptedError('Peer failed or supervisor stopped')
    def call(self,lane,fn,*a,**k):return self.service.call(lane,fn,*a,**k)
    def routed(self,fn):
        def call(*a,**k):
            lane=self.rng.current()
            if lane is None:raise RuntimeError('Torch/RNG call outside registered lane')
            return self.call(lane,fn,*a,**k)
        return call
    def prepare_task(self,task,lane):
        self.task_lifetimes.append(weakref.ref(task))
        setup=task.setup_demo;close=task.close_env
        def setup_wrapped(*a,**k):
            def execute():
                with self.scene_gate.write():
                    value=setup(*a,**k)
                    self.scene_lifetimes.append(weakref.ref(task.scene))
                    from collection_native_step import install as install_step
                    from .render_upload import install
                    install_step(task,self.scene_gate,self.native_step)
                    install(task,self.scene_gate,self.native)
                    from .prefix_fast import install as install_prefix
                    install_prefix(task,self.scene_gate)
                    return value
            execute._cx_priority=5;execute._cx_stage='setup'
            return self.call(lane,execute)
        def close_wrapped(*a,**k):
            with self.scene_gate.write():return close(*a,**k)
        task.setup_demo=setup_wrapped;task.close_env=close_wrapped
        for name in ('get_obs','_update_render'):
            original=getattr(task,name)
            def wrapped(*a,_original=original,**k):
                self.check_cancel()
                with self.scene_gate.read():return _original(*a,**k)
            setattr(task,name,wrapped)
        return task
