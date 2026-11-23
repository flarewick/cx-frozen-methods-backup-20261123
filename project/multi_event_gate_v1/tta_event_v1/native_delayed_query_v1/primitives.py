"""Bind original pure function code objects without executing CPU script setup.

Old data scripts change CUDA_VISIBLE_DEVICES and CPU affinity at import. Those
top-level scripts are not executed here. Original function bodies are compiled
unchanged from full source, then bound to explicitly supplied pure dependencies.
"""
from pathlib import Path
from types import FunctionType, CodeType, SimpleNamespace
import hashlib
import importlib.util
import re
import builtins
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
T=Path(__file__).resolve().parent.parent
SOURCE_SHA256={}

def functions(path,names,env):
    path=Path(path)
    env['__builtins__']=builtins.__dict__
    text=path.read_text();SOURCE_SHA256[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    code=compile(text,str(path),'exec')
    out={}
    for name in names:
        hits=[v for v in code.co_consts if isinstance(v,CodeType)and v.co_name==name]
        if len(hits)!=1:raise ValueError('Ambiguous original pure function')
        out[name]=FunctionType(hits[0],env,name)
    env.update(out)
    return SimpleNamespace(**out)

def pure_projection():
    path=T/'local_tcp_v1/projection.py'
    spec=importlib.util.spec_from_file_location('delayed_pure_hardware_projection',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    SOURCE_SHA256[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return value

projection=pure_projection()
local=functions(T/'stage_feedback_v2/association.py',('rgb_check','language_roles','color_mask'),
                dict(np=np,cv2=cv2,re=re,COLORS=('red','green','blue','yellow')))
semantic=functions(T/'language_instances_v2/semantic.py',('prompt',),dict(re=re,local=local))
semantic.local=local
surface=functions(T/'visible_surface_v1/surface.py',('prototype','refine'),dict(np=np,semantic=semantic))
flow=functions(T/'visible_surface_v1/flow.py',('patches','corners','track'),
               dict(np=np,cv2=cv2,MAX_POINTS=24))
flow.MAX_POINTS=24
features=SimpleNamespace(tool_positions=projection.tool_positions)
native=functions(T/'native_physical_v1/data.py',('velocity_features',),dict(np=np,features=features))
joints=functions(T/'visible_surface_v1/commands.py',('full_joint_features',),
                 dict(np=np,Rotation=Rotation,batch_fk=projection.batch_fk,ARM=np.array([0,1,2,3,4,5,7,8,9,10,11,12])))
ARM=np.array([0,1,2,3,4,5,7,8,9,10,11,12]);GRIP=np.array([6,13])
old=functions(T/'visible_surface_v1/data.py',('tcp','action','context','supported'),
              dict(np=np,common=SimpleNamespace(features=features),flow=flow,native=native,joints=joints,ARM=ARM,GRIP=GRIP))
old.surface,old.flow,old.ARM,old.GRIP=surface,flow,ARM,GRIP

def configuration():
    """Published hardware/configuration only; no task object or measured joints."""
    import json,yaml
    from cx_tta.robot_kinematics import RobotFK
    plan=json.loads((T.parents[1]/'multi_event_gate_v1/outputs/stacking_adapter/shared_20261014_031147_697383/training_plan.json').read_text())
    cfg=json.loads(Path(plan['robot_config']).read_text())
    cameras=yaml.safe_load(Path('/data/cj/robo_pi/250010033/qfw/robotwin/RoboTwin/task_config/_camera_config.yml').read_text())
    return cfg,projection.FixedHead(cfg,cameras),[RobotFK(cfg,side)for side in('left','right')]
