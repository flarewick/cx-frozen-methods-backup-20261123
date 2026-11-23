"""Overlap original native rendering with already queued safe model requests."""
import ast,contextlib,inspect,queue,textwrap,threading,time,heapq
from concurrent.futures import ThreadPoolExecutor

@contextlib.contextmanager
def suspended_rng(c,torch):
    router=c.rng;lane=router.current();depth=getattr(router.local,'torch_depth',0)
    assert lane is not None and depth>0
    cpu=torch.random.get_rng_state().clone()
    cuda=[s.clone()for s in torch.cuda.get_rng_state_all()]if router.cuda else []
    router.local.lane=None;router.local.torch_depth=0
    try:yield
    finally:
        try:
            assert router.current()is None and getattr(router.local,'torch_depth',0)==0
            assert torch.equal(cpu,torch.random.get_rng_state())
            if router.cuda:assert all(torch.equal(a,b)for a,b in zip(cuda,torch.cuda.get_rng_state_all(),strict=True))
        finally:router.local.lane=lane;router.local.torch_depth=depth

def install_queue_filter(service,allowed,pumping,note):
    """Preserve promotions; bypass ordinary unsafe work only during render wait."""
    method=service.process_one;glob=method.__func__.__globals__;original=glob['_fair18_take']
    text=textwrap.dedent(inspect.getsource(original))
    anchor='   setups=[(i,x)for i,x in enumerate(q.queue)if getattr(x[3],\'_cx_stage\',None)==\'setup\']\n'
    # inspect strips the enclosing indent, while retaining original body spacing.
    if anchor not in text:
        anchor=anchor[1:]
    assert text.count(anchor)==1
    indent=anchor[:len(anchor)-len(anchor.lstrip())]
    guard=indent+"if not _render_queue_allowed(q.queue,burst[0],planner_burst[0]):raise queue.Empty\n"
    old_take='else:item=q._get()'
    new_take='else:item=_render_take_ordinary(q)'
    assert text.count(old_take)==1
    candidate=text.replace(anchor,guard+anchor).replace(old_take,new_take)
    assert ast.dump(ast.parse(candidate.replace(guard,'').replace(new_take,old_take)))==ast.dump(ast.parse(text))
    closure=inspect.getclosurevars(original).nonlocals
    selected=[None]
    def check(items,burst,planner_burst):
        selected[0]=None
        if not pumping():return True
        setups=[x for x in items if getattr(x[3],'_cx_stage',None)=='setup']
        planners=[x for x in items if getattr(x[3],'_cx_stage','').startswith('original_expert_')]
        if burst>=6 and setups:
            result=allowed(min(setups,key=lambda x:x[1]))
            if not result:note('setup_promotion_wait')
            return result
        if planner_burst>=16 and planners:
            result=allowed(min(planners,key=lambda x:x[1]))
            if not result:note('expert_promotion_wait')
            return result
        safe=[item for item in items if allowed(item)]
        if not safe:
            note('no_safe_request_ready');return False
        selected[0]=min(safe,key=lambda item:(item[0],item[1]))
        if selected[0]is not min(items):note('ordinary_request_bypassed')
        return allowed(selected[0])
    def ordinary(q):
        if not pumping():return q._get()
        item=selected[0];assert item is not None
        index=next(i for i,value in enumerate(q.queue)if value is item)
        q.queue.pop(index);heapq.heapify(q.queue);selected[0]=None
        return item
    namespace=dict(original.__globals__,**closure,_render_queue_allowed=check,_render_take_ordinary=ordinary)
    exec(compile(candidate,'<original-fair-take-with-render-deferral>','exec'),namespace)
    glob['_fair18_take']=namespace[original.__name__]
    return lambda:glob.__setitem__('_fair18_take',original)

class NativeOverlap:
    def __init__(self,c,torch,native,graph_ready,*,control_codes=(),CPU_fixture_codes=None,on_progress=None):
        self.c,self.torch,self.native,self.graph_ready=c,torch,native,graph_ready
        self.owner=threading.get_ident();self.local=threading.local();self.pumping=False
        self.pending=None;self.on_progress=on_progress;self.last_progress=time.monotonic()
        if CPU_fixture_codes is None:
            from native_policy_execution_v1.actor import ActorQuery
            self.safe_codes={code for code in ActorQuery.query.__code__.co_consts if inspect.iscode(code)and code.co_name in('encode','sample')}
            assert len(self.safe_codes)==2
            seeds={code for code in ActorQuery.__init__.__code__.co_consts if inspect.iscode(code)and code.co_name=='seed'}
            assert len(seeds)==1;self.safe_codes.update(seeds)
            for code in control_codes:
                assert inspect.iscode(code)
                if code.co_name=='check':assert code.co_qualname=='install.<locals>.check'and code.co_filename.endswith('/owner_precision.py')
                elif code.co_name=='diagnostic_owner':assert code.co_qualname=='execute.<locals>.run_case.<locals>.diagnostic_owner'and code.co_filename.endswith('/runtime.py')
                elif code.co_name=='operation':assert code.co_qualname=='execute.<locals>.run_case.<locals>.tracker.<locals>.operation'and code.co_filename.endswith('/runtime.py')
                else:raise AssertionError('Non-original owner control requested')
        else:
            assert not c.rng.cuda and not torch.cuda.is_initialized()
            self.safe_codes=set(CPU_fixture_codes)
        self.safe_codes.update(control_codes)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='original-native-render')
        self.wait_counts={};self.worker_metrics=[]
        def note(reason):self.wait_counts[reason]=self.wait_counts.get(reason,0)+1
        self.unfilter=install_queue_filter(c.service,self.allowed,lambda:self.pumping,note);self.calls=0;self.pumped=0;self.closed=False
        self.worker_threads=set();self.requests=[];self.last_selected=None
    def __getattr__(self,name):return getattr(self.native,name)
    def allowed(self,item):
        if not self.pumping:return True
        lane,fn=item[2],item[3]
        if getattr(fn,'__code__',None)not in self.safe_codes:return False
        # Only original policy closures; no planning, setup, cleanup or callbacks.
        permitted=fn.__name__=='encode'and lane.name.endswith('_encoder')or fn.__name__=='sample'and lane.name.endswith('_sampler')and self.graph_ready()
        permitted = (
            permitted
            or (fn.__name__ == 'seed' and lane.name.endswith(('_encoder', '_sampler')))
            or fn.__name__ in ('check', 'diagnostic_owner', 'operation')
        )
        if permitted:self.last_selected=dict(method=fn.__name__,lane=lane.name,owner_thread=threading.get_ident(),graph_initialized=self.graph_ready())
        return permitted
    @contextlib.contextmanager
    def active(self,enabled):
        old=getattr(self.local,'enabled',False);self.local.enabled=enabled
        try:yield
        finally:self.local.enabled=old
    def run(self,fn,args):
        if threading.get_ident()!=self.owner or not getattr(self.local,'enabled',False)or self.pumping:return fn(*args)
        gate=self.c.scene_gate
        with gate.condition:
            if gate.writer is not None or self.owner not in gate.readers:return fn(*args)
        ready=threading.Event()
        def render():
            tid=threading.get_ident()
            started=time.monotonic()
            self.worker_threads.add(tid)
            with gate.condition:
                assert gate.writer is None and self.owner in gate.readers
                # This read continues the already admitted original owner read.
                gate.readers[tid]=gate.readers.get(tid,0)+1;ready.set()
            try:return fn(*args)
            finally:
                self.worker_metrics.append(dict(method=fn.__name__,worker_seconds=time.monotonic()-started))
                with gate.condition:
                    gate.readers[tid]-=1
                    if not gate.readers[tid]:del gate.readers[tid]
                    gate.condition.notify_all()
        future=self.pool.submit(render)
        try:
            while not ready.wait(.01):
                if future.done():return future.result()
                self.c.check_cancel()
        except BaseException:
            future.result();raise
        self.calls+=1
        self.pending=future
        try:
            with suspended_rng(self.c,self.torch):
                self.pumping=True
                try:
                    while not future.done():
                        self.c.check_cancel()
                        self.last_selected=None
                        if self.c.service.process_one(timeout=0):
                            assert self.last_selected is not None
                            self.requests.append(self.last_selected);self.pumped+=1
                        else:time.sleep(.001)
                finally:self.pumping=False
            return future.result()
        finally:
            # Never leave a renderer alive after releasing its scene read.
            if not future.done():future.result()
            self.pending=None
            if self.on_progress is not None and time.monotonic()-self.last_progress>=5:
                self.on_progress(self.snapshot());self.last_progress=time.monotonic()
    def render_step(self,*args):return self.run(self.native.render_step,args)
    def take_pictures(self,*args):return self.run(self.native.take_pictures,args)
    def close(self):
        assert threading.get_ident()==self.owner and not self.closed
        self.pool.shutdown(wait=True);self.unfilter();self.closed=True
    def await_renderer(self):
        assert threading.get_ident()==self.owner
        if self.pending is not None:self.pending.result()
    def snapshot(self):
        return dict(passed=False,bookkeeping_only=True,actual_batch_completion_NOT_assumed=True,original_native_calls=self.calls,original_owner_requests=self.pumped,owner_thread=self.owner,worker_threads=sorted(self.worker_threads),requests=list(self.requests),worker_pool_closed=self.closed,no_extra_capture_or_inference=True,wait_counts=dict(self.wait_counts),worker_metrics=list(self.worker_metrics))
    def report(self):
        assert self.closed
        value=self.snapshot();value['passed']=True;return value
