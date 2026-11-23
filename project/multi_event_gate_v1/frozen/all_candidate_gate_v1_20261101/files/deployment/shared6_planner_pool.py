"""One original planner pair for owner-serialized expert screening only.

Policy qpos control keeps independent original gripper interpolation and cannot
read mutable planner state. The first pair is constructed eagerly, in the real
first screen, exactly where the original Robot calls its constructors.
"""
import ast,copy,hashlib,inspect,json,textwrap,threading
from pathlib import Path
D=Path(__file__).resolve().parent

def signature(value):
    if isinstance(value,dict):return {str(k):signature(v)for k,v in sorted(value.items())}
    if isinstance(value,(list,tuple)):return [signature(v)for v in value]
    if hasattr(value,'p')and hasattr(value,'q'):return dict(p=value.p.tolist(),q=value.q.tolist())
    if hasattr(value,'tolist'):return value.tolist()
    if isinstance(value,(str,Path)):
        p=Path(value)
        return dict(file_sha256=hashlib.sha256(p.read_bytes()).hexdigest())if p.is_file()else str(value)
    return value

def make_class(original,c):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.plan_grippers)))
    if any(isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='self'for n in ast.walk(tree)):
        raise RuntimeError('Original gripper interpolation depends on initialized planner state')
    groups={};shared=[];keys=[]
    def expert():
        lane=c.rng.current()
        if lane is None or getattr(lane,'phase',None)!='expert_screen':raise RuntimeError('Mutable planner access outside original expert screening')
        if c.service is not None and threading.get_ident()!=c.service.owner:raise RuntimeError('Mutable planner escaped serialized owner')
        return lane
    class Pooled(original):
        def __init__(self,*args,**kwargs):
            lane=c.rng.current()
            if lane is None or(c.service is not None and threading.get_ident()!=c.service.owner):raise RuntimeError('Planner proxy construction escaped scene owner')
            group=groups.setdefault(lane.name,[]);index=len(group)
            if index>=2:raise RuntimeError('Original planner pair unexpectedly recreated')
            key=json.dumps(signature([args,kwargs]),sort_keys=True)
            if index==len(shared):
                expert()
                if index and len(group)!=1:raise RuntimeError('Original left/right constructor ordering changed')
                # No RNG save/restore: the original eager constructor consumes
                # exactly its original RNG here, inside the first live screen.
                real=original(*args,**kwargs);shared.append(real);keys.append(key)
                c.note('original_expert_planner_created',side=index,pairs=1)
            else:
                if keys[index]!=key:raise RuntimeError('Expert planner configuration/base pose differs; cannot pool')
                real=shared[index]
                c.note('original_expert_planner_reused',lane=lane.name,side=index)
            object.__setattr__(self,'_cx_real',real);group.append(self)
        def __getattribute__(self,name):
            if name.startswith('_cx_')or name in('__class__','__dict__','plan_grippers'):return object.__getattribute__(self,name)
            expert();return getattr(object.__getattribute__(self,'_cx_real'),name)
        def __setattr__(self,name,value):
            if name.startswith('_cx_'):return object.__setattr__(self,name,value)
            expert();setattr(object.__getattribute__(self,'_cx_real'),name,value)
    Pooled.__name__='OwnerOnlyOriginalExpertPlanner'
    return Pooled,groups

def install(c):
    import envs.robot.robot as robot
    proof=json.loads((D/'reports/target_shared6_planner_pool_cpu_20261024.json').read_text())
    if not proof.get('passed')or proof['helper_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()or proof['original_planner_sha256']!=hashlib.sha256(Path(inspect.getsourcefile(robot.CuroboPlanner)).read_bytes()).hexdigest():raise RuntimeError('Expert planner pool CPU proof missing/stale')
    pooled,groups=make_class(robot.CuroboPlanner,c);robot.CuroboPlanner=pooled
    c.note('owner_only_expert_planner_pool_installed',real_pairs=1);return groups
