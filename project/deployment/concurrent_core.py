"""Single-process lane controls around unchanged original episode closures."""
from contextlib import contextmanager
import copy
import functools
import threading
import time
import json
from scene_gate import SceneGate
from concurrent_rng import Lane,RngRouter


class StopRequested(RuntimeError):pass


class PolicyGate:
    def __init__(self,policy,coordinator,lane):
        object.__setattr__(self,'_wrapped',policy);object.__setattr__(self,'_coordinator',coordinator);object.__setattr__(self,'_lane',lane)
    def __getattr__(self,name):return getattr(self._wrapped,name)
    def __setattr__(self,name,value):
        if name=='predict':return object.__setattr__(self,name,value)
        setattr(self._wrapped,name,value)
    def __delattr__(self,name):
        if name=='predict':return object.__delattr__(self,name)
        delattr(self._wrapped,name)
    def predict(self,*args,**kwargs):
        c=self._coordinator;c.check_cancel()
        def invoke():
            result=self._wrapped.predict(*args,**kwargs)
            c.verify_weights()
            return result
        invoke._cx_priority=0;invoke._cx_stage="predict"
        return c.call(self._lane,invoke)


class Coordinator:
    def __init__(self,cuda=False):
        self.rng=RngRouter(cuda);self.scene_gate=SceneGate()
        self.cancel=threading.Event();self.base=None;self.weight_versions=None
        self.screens={};self.base_factory=None;self.service=None;self.wrapper_factory=None;self.native=None
        self.began=time.monotonic()

    def note(self,stage,**extra):
        print("CONCURRENT_TIMING",json.dumps(dict(stage=stage,seconds=round(time.monotonic()-self.began,3),**extra)),flush=True)

    def call(self,lane,fn,*args,**kwargs):
        if self.service:return self.service.call(lane,fn,*args,**kwargs)
        with self.rng.torch_scope(lane):return fn(*args,**kwargs)

    def check_cancel(self):
        if self.cancel.is_set():raise StopRequested('Peer failed or supervisor stopped')

    def verify_weights(self):
        if self.weight_versions is not None:
            from shared_weights import versions
            if versions(self.base._model)!=self.weight_versions:
                raise RuntimeError('Shared frozen parameters changed during evaluation')

    def get_base(self,lane):
        self.check_cancel()
        def invoke():
            if self.base is None:
                self.note("base_create_begin")
                self.base=self.base_factory()
                self.note("base_create_end")
                self.base._model.eval().requires_grad_(False)
                from shared_weights import versions
                self.weight_versions=versions(self.base._model)
            self.verify_weights();return self.base
        invoke._cx_priority=2;invoke._cx_stage="create_base"
        return self.call(lane,invoke)

    def screen(self,lane,original,seed,index):
        self.check_cancel()
        key=(seed,index)
        def invoke():
            if key not in self.screens:
                phase=getattr(lane,'phase',None);lane.phase='expert_screen'
                try:row=original(seed,index)
                finally:lane.phase=phase
                self.screens[key]=(copy.deepcopy(row),self.rng.capture(lane),'live_original_screen')
            row,state,origin=self.screens[key]
            if state is not None:self.rng.restore(lane,state)
            result=copy.deepcopy(row)
            result['deployment_screen_origin']=origin
            return result
        invoke._cx_priority=5;invoke._cx_stage="screen"
        return self.call(lane,invoke)

    def prepare_task(self,task,lane):
        def wrap_method(name,torch_scope=False,exclusive=False):
            original=getattr(task,name)
            @functools.wraps(original)
            def invoke(*a,**k):
                self.check_cancel()
                if torch_scope:
                    def execute():
                        with self.scene_gate.write():
                            result=original(*a,**k)
                            if self.native is not None:
                                from native_physics import install_task
                                install_task(task,self.scene_gate,self.native)
                                from native_render import install_task as install_render
                                install_render(task,self.scene_gate,self.native,self.note)
                            return result
                    execute._cx_priority=4;execute._cx_stage="setup_demo"
                    return self.call(lane,execute)
                with (self.scene_gate.write() if exclusive else self.scene_gate.read()):return original(*a,**k)
            setattr(task,name,invoke)
        wrap_method('setup_demo',True)
        for name in ('get_obs','_update_render'):wrap_method(name)
        wrap_method('close_env',exclusive=True)
        return task


def closure_set(function,name,value):
    if name not in function.__code__.co_freevars:raise ValueError('Original closure no longer has '+name)
    function.__closure__[function.__code__.co_freevars.index(name)].cell_contents=value


def configure_callbacks(callbacks,coordinator,lane):
    rollout=callbacks['rollout'];coordinator.prepare_task(callbacks['task'],lane)
    def base():return coordinator.get_base(lane)
    closure_set(rollout,'create_base_policy',base)
    if coordinator.wrapper_factory is None:
        from frozen_factory import FrozenFactory
        coordinator.wrapper_factory=FrozenFactory(callbacks['make_policy'])
    original_make=coordinator.wrapper_factory
    def make(*a,**k):
        def construct():return PolicyGate(original_make(*a,**k),coordinator,lane)
        construct._cx_priority=3;construct._cx_stage="make_policy"
        return coordinator.call(lane,construct)
    closure_set(rollout,'make_policy',make)
    original_probe=callbacks['steady_probe']
    def probe(*a,**k):
        coordinator.note("probe_begin",lane=lane.name)
        lane.phase="discarded_probe"
        try:return original_probe(*a,**k)
        finally:
            lane.phase="rollout"
            coordinator.note("probe_end",lane=lane.name)
    closure_set(rollout,'steady_probe',probe)
    screen=callbacks['screen']
    callbacks['screen']=lambda seed,index:coordinator.screen(lane,screen,seed,index)
    return callbacks
