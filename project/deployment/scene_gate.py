"""Concurrent rendering of distinct scenes; exclusive setup/global cache clear."""
from contextlib import contextmanager
import threading


class SceneGate:
    def __init__(self):
        self.condition=threading.Condition(threading.RLock());self.readers={};self.writer=None;self.depth=0;self.pending=0

    @contextmanager
    def read(self):
        tid=threading.get_ident()
        with self.condition:
            nested=tid in self.readers or self.writer==tid
            while not nested and (self.writer is not None or self.pending):self.condition.wait()
            self.readers[tid]=self.readers.get(tid,0)+1
        try:yield
        finally:
            with self.condition:
                self.readers[tid]-=1
                if not self.readers[tid]:del self.readers[tid]
                self.condition.notify_all()

    @contextmanager
    def write(self):
        tid=threading.get_ident()
        with self.condition:
            self.pending+=1
            try:
                while self.writer not in (None,tid) or any(t!=tid for t in self.readers):self.condition.wait()
                self.writer=tid;self.depth+=1
            finally:self.pending-=1
        try:yield
        finally:
            with self.condition:
                self.depth-=1
                if not self.depth:self.writer=None
                self.condition.notify_all()
