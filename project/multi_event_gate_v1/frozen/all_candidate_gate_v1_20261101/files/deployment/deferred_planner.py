"""Defer unused CuRobo construction from discarded probes until expert demand.

The original joint-action TOPP and gripper interpolation remain unchanged.
Live expert screening uses one serialized original left/right planner pair. Their constructors replay
in their original order under the RNG state captured in the discarded probe,
then the current scene RNG is restored. Policy qpos dynamics and histories never use this mutable expert planner pair.
"""
import ast
import copy
import inspect
import hashlib
import json
from pathlib import Path
import textwrap
import types


def make_class(original,coordinator):
    # plan_grippers is the only supported cheap method. Fail closed if the
    # installed original starts reading planner state or calling another method.
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.plan_grippers)))
    if any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self' for n in ast.walk(tree)):
        raise ValueError('Original gripper interpolation now depends on initialized CuRobo state')
    groups={};shared={'signature':None,'pair':None}
    def signature(value):
        if isinstance(value,dict):return {str(k):signature(v)for k,v in sorted(value.items())}
        if isinstance(value,(tuple,list)):return [signature(v)for v in value]
        if hasattr(value,'p')and hasattr(value,'q'):return dict(p=value.p.tolist(),q=value.q.tolist())
        if isinstance(value,(str,Path)):
            path=Path(value)
            if path.is_file():return dict(file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            return str(value)
        return value

    class Group:
        def __init__(self,lane):
            if getattr(lane,'phase',None)!='discarded_probe':raise RuntimeError('Deferred planners must originate in the discarded original probe')
            self.lane=lane;self.nodes=[];self.saved=coordinator.rng.capture(lane)
        def ensure(self):
            def create():
                if getattr(coordinator.rng.current(),'phase',None)!='expert_screen':raise RuntimeError('CuRobo materialization is restricted to original expert screening')
                if all(object.__getattribute__(node,'_cx_real')is not None for node in self.nodes):return
                if len(self.nodes)!=2:raise RuntimeError('Expected original left/right planner pair')
                key=json.dumps([signature([object.__getattribute__(node,'_cx_args'),object.__getattribute__(node,'_cx_kwargs')])for node in self.nodes],sort_keys=True)
                if shared['pair']is not None:
                    if shared['signature']!=key:raise RuntimeError('Expert planner configuration/base pose differs; cannot share mutable planning state')
                    for node,real in zip(self.nodes,shared['pair']):object.__setattr__(node,'_cx_real',real)
                    return
                current=coordinator.rng.capture(self.lane)
                coordinator.note('deferred_planners_materialize_begin',lane=self.lane.name)
                try:
                    coordinator.rng.restore(self.lane,self.saved)
                    for node in self.nodes:
                        args=object.__getattribute__(node,'_cx_args');kwargs=object.__getattribute__(node,'_cx_kwargs')
                        real=original(*args,**kwargs);object.__setattr__(node,'_cx_real',real)
                finally:coordinator.rng.restore(self.lane,current)
                shared['signature']=key;shared['pair']=[object.__getattribute__(node,'_cx_real')for node in self.nodes]
                coordinator.note('deferred_planners_materialize_end',lane=self.lane.name)
            return coordinator.call(self.lane,create)

    class Deferred(original):
        def __init__(self,*args,**kwargs):
            lane=coordinator.rng.current()
            if lane is None:raise RuntimeError('CuRobo constructor outside an evaluation lane')
            group=groups.get(lane.name)
            if group is None:group=groups[lane.name]=Group(lane)
            if len(group.nodes)>=2:raise RuntimeError('Native planner pair unexpectedly recreated')
            object.__setattr__(self,'_cx_group',group);object.__setattr__(self,'_cx_real',None)
            object.__setattr__(self,'_cx_args',copy.deepcopy(args));object.__setattr__(self,'_cx_kwargs',copy.deepcopy(kwargs))
            group.nodes.append(self)
        def __getattribute__(self,name):
            if name.startswith('_cx_') or name in ('__class__','__dict__','plan_grippers'):
                return object.__getattribute__(self,name)
            lane=coordinator.rng.current()
            if lane is None or getattr(lane,'phase',None)!='expert_screen':raise RuntimeError('Mutable CuRobo state requested outside expert screening')
            if coordinator.service is not None:
                import threading
                if threading.get_ident()!=coordinator.service.owner:raise RuntimeError('Expert planner escaped the serialized model owner')
            group=object.__getattribute__(self,'_cx_group');group.ensure()
            return getattr(object.__getattribute__(self,'_cx_real'),name)
        def __setattr__(self,name,value):
            if name.startswith('_cx_'):return object.__setattr__(self,name,value)
            object.__getattribute__(self,'_cx_group').ensure()
            return setattr(object.__getattribute__(self,'_cx_real'),name,value)
    Deferred.__name__='DeferredOriginalCuRoboPlanner'
    return Deferred,groups


def install(coordinator):
    import envs.robot.robot as robot
    here=Path(__file__).resolve().parent
    proof=json.loads((here/'reports/target_deferred_planner_cpu_20261023.json').read_text())
    original_source=Path(inspect.getsourcefile(robot.CuroboPlanner))
    if not proof.get('passed') or proof['helper_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or proof['original_gripper_source_sha256']!=hashlib.sha256(original_source.read_bytes()).hexdigest():
        raise RuntimeError('Deferred planner CPU proof does not bind this original runtime')
    deferred,groups=make_class(robot.CuroboPlanner,coordinator)
    robot.CuroboPlanner=deferred
    coordinator.note('deferred_original_planners_installed')
    return groups
