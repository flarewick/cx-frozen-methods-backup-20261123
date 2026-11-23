"""One owner OS thread executes every original sampler and Torch RNG phase."""
from concurrent.futures import Future,TimeoutError
import queue
import threading
import time
import itertools


class ModelService:
    def __init__(self,coordinator):
        self.coordinator=coordinator;self.owner=threading.get_ident();self.requests=queue.PriorityQueue();self.sequence=itertools.count()
        self.executed=0

    def execute(self,lane,fn,args,kwargs):
        if threading.get_ident()!=self.owner:raise RuntimeError('Model service escaped its owner thread')
        c=self.coordinator;c.check_cancel()
        with c.rng.bind(lane),c.rng.torch_scope(lane):
            self.executed+=1
            began=time.monotonic()
            try:return fn(*args,**kwargs)
            finally:
                elapsed=time.monotonic()-began
                if elapsed>1:c.note("owner_call",lane=lane.name,operation=getattr(fn,"_cx_stage",fn.__name__),duration=round(elapsed,3))

    def call(self,lane,fn,*args,**kwargs):
        if threading.get_ident()==self.owner:return self.execute(lane,fn,args,kwargs)
        self.coordinator.check_cancel();future=Future()
        self.requests.put((getattr(fn,"_cx_priority",1),next(self.sequence),lane,fn,args,kwargs,future))
        while True:
            try:return future.result(timeout=.1)
            except TimeoutError:
                if future.done():raise
                self.coordinator.check_cancel()

    def process_one(self,timeout=.05):
        if threading.get_ident()!=self.owner:raise RuntimeError('Only service owner may process requests')
        try:_,_,lane,fn,args,kwargs,future=self.requests.get(timeout=timeout)
        except queue.Empty:return False
        if not future.set_running_or_notify_cancel():return True
        try:future.set_result(self.execute(lane,fn,args,kwargs))
        except BaseException as exc:future.set_exception(exc)
        return True
