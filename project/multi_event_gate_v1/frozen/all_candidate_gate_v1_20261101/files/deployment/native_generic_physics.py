"""GIL-free original control with the original task success callback at EVERY tick."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import types
import numpy as np

HERE=Path(__file__).resolve().parent

class JointShadow:
    def __init__(self,joint):
        self.position=np.asarray(joint.get_drive_target(),np.float32).copy()
        self.velocity=np.asarray(joint.get_drive_velocity_target(),np.float32).copy()
        if self.position.shape!=(1,):raise ValueError('Only original one-DOF gripper joints supported')
    @property
    def drive_target(self):return self.position.copy()
    def get_drive_target(self):return self.position.copy()
    def set_drive_target(self,value):self.position=np.asarray(value,np.float32).reshape(1).copy()
    def set_drive_velocity_target(self,value):self.velocity=np.asarray(value,np.float32).reshape(1).copy()


def gripper_paths(robot,commands,side):
    shadow=types.SimpleNamespace(left_entity=None,right_entity=None,_entity_qf=lambda entity:None)
    for name in ('left_gripper_scale','right_gripper_scale','left_gripper_val','right_gripper_val'):setattr(shadow,name,copy.deepcopy(getattr(robot,name)))
    for name in ('left_gripper','right_gripper'):
        setattr(shadow,name,[(JointShadow(joint),factor,bias) for joint,factor,bias in getattr(robot,name)])
    shadow.get_normal_real_gripper_val=types.MethodType(robot.get_normal_real_gripper_val.__func__,shadow)
    method=types.MethodType(robot.set_gripper.__func__,shadow)
    joints=getattr(shadow,side+'_gripper');rows=[];intent=[]
    for command in commands:
        method(command,side)
        rows.append([v for joint,_,_ in joints for v in (joint.position[0],joint.velocity[0])])
        intent.append(getattr(shadow,side+'_gripper_val'))
    return np.asarray(rows,np.float32),np.asarray(intent,np.float64)


def native_gripper_paths(native,robot,commands,side):
    if np.__version__!='1.26.4':raise RuntimeError('Native gripper rounding requires verified NumPy1.26.4')
    joints=getattr(robot,side+'_gripper');scale=getattr(robot,side+'_gripper_scale')
    return native.grippers(np.asarray(commands,np.float64),np.array([j.get_drive_target()[0]for j,_,_ in joints],np.float32),np.array([m for _,m,_ in joints],np.float64),np.array([b for _,_,b in joints],np.float64),float(scale[0]),float(scale[1]))


def load_native():
    proof=json.loads((HERE/'reports/target_native_generic_physics_cpu_20261024.json').read_text())
    build=json.loads((HERE/'.work/native_generic_control/build.json').read_text());binary=Path(build['binary'])
    if not proof.get('passed') or proof['helper_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or proof['binary_sha256']!=hashlib.sha256(binary.read_bytes()).hexdigest() or build['source_sha256']!=hashlib.sha256((HERE/'native_generic_control.cpp').read_bytes()).hexdigest():raise RuntimeError('Native CPU physics proof missing/stale')
    # Initialize SAPIEN's bundled dependent libraries before our SDK-linked extension.
    import sapien
    sys.path.insert(0,str(binary.parent));import cx_native_generic_control
    return cx_native_generic_control


def install_task(task,scene_gate,native):
    if getattr(task,'_cx_native_physics_installed',False):return
    original=task.take_action
    proof=json.loads((HERE/'reports/target_native_generic_physics_cpu_20261024.json').read_text())
    if hashlib.sha256(Path(inspect.getsourcefile(original.__func__)).read_bytes()).hexdigest()!=proof['original_base_source_sha256']:raise RuntimeError('Original base control source changed')
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.__func__)))
    fn=tree.body[0];loops=[n for n in ast.walk(fn)if isinstance(n,ast.While)]
    if len(loops)!=1 or ast.unparse(loops[0].test)!='now_left_id < left_n_step or now_right_id < right_n_step':raise ValueError('Frozen original control loop changed')
    replacement=ast.parse('if self._cx_control(left_result if topp_left_flag else None, right_result if topp_right_flag else None, left_n_step, right_n_step, topp_left_flag, topp_right_flag, left_gripper, right_gripper):\n    return').body[0]
    class Replace(ast.NodeTransformer):
        def visit_While(self,node):return ast.copy_location(replacement,node)
    tree=Replace().visit(tree);ast.fix_missing_locations(tree);ns={}
    exec(compile(tree,original.__func__.__code__.co_filename,'exec'),original.__func__.__globals__,ns)
    optimized=types.MethodType(ns['take_action'],task)
    def control(left,right,ln,rn,lf,rf,lg,rg):
        robot=task.robot
        lgr,li=native_gripper_paths(native,robot,lg,'left');rgr,ri=native_gripper_paths(native,robot,rg,'right')
        empty=np.empty((0,0),np.float64)
        def check_success(lvalue,rvalue):
            robot.left_gripper_val=np.float64(lvalue);robot.right_gripper_val=np.float64(rvalue)
            task._update_render()
            return bool(task.check_success())
        with scene_gate.read():
            result=native.control(robot.left_entity,robot.right_entity,robot.left_arm_joints,robot.right_arm_joints,
                [v[0] for v in robot.left_gripper],[v[0]for v in robot.right_gripper],task.scene.physx_system,check_success,
                left['position']if lf else empty,left['velocity']if lf else empty,right['position']if rf else empty,right['velocity']if rf else empty,
                lgr,rgr,li,ri,ln,rn,lf,rf,float(robot.left_gripper_val),float(robot.right_gripper_val))
        success,i,j,ticks,lvalue,rvalue=result
        robot.left_gripper_val=np.float64(lvalue);robot.right_gripper_val=np.float64(rvalue)
        if success:task.eval_success=True;task.get_obs()
        return success
    task._cx_control=control
    def take_action(action,action_type='qpos'):
        if task.crazy_random_light or action_type!='qpos':return original(action,action_type)
        if task.render_freq or task.eval_video_path is not None:raise ValueError('Native physics requires original headless no-video protocol')
        return optimized(action,action_type)
    task.take_action=take_action;task._cx_native_physics_installed=True
