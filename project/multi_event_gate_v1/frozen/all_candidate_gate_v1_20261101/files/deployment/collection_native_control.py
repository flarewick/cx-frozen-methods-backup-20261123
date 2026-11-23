"""Original dense expert control in GIL-free chunks ending at EVERY capture tick."""
import ast,copy,hashlib,inspect,json,sys,textwrap,types
from pathlib import Path
import numpy as np
from native_physics import JointShadow
D=Path(__file__).resolve().parent

def load():
 import sapien
 build=json.loads((D/'.work/collection_control/build.json').read_text());binary=Path(build['binary']);proof=json.loads((D/'reports/target_collection_control_cpu_20261024.json').read_text())
 if not proof['passed']or proof['helper_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()or proof['binary_sha256']!=hashlib.sha256(binary.read_bytes()).hexdigest()or build['source_sha256']!=hashlib.sha256((D/'collection_control.cpp').read_bytes()).hexdigest():raise RuntimeError('Expert dense control CPU proof stale')
 sys.path.insert(0,str(binary.parent));import cx_collection_control
 return cx_collection_control

def grippers(robot,control,side):
 joints=getattr(robot,side+'_gripper')
 if control is None:return np.empty((0,len(joints)*2),np.float32),np.empty(0,np.float64)
 shadow=types.SimpleNamespace(left_entity=None,right_entity=None,_entity_qf=lambda _:None)
 for name in ('left_gripper_scale','right_gripper_scale','left_gripper_val','right_gripper_val'):setattr(shadow,name,copy.deepcopy(getattr(robot,name)))
 for name in ('left_gripper','right_gripper'):setattr(shadow,name,[(JointShadow(j),m,b)for j,m,b in getattr(robot,name)])
 shadow.get_normal_real_gripper_val=types.MethodType(robot.get_normal_real_gripper_val.__func__,shadow);setter=types.MethodType(robot.set_gripper.__func__,shadow);rows=[];intent=[]
 for command in control['result'][:control['num_step']]:
  setter(command,side,control['per_step']);rows.append([v for j,_,_ in getattr(shadow,side+'_gripper')for v in (j.position[0],j.velocity[0])]);intent.append(getattr(shadow,side+'_gripper_val'))
 if len(rows)!=control['num_step']:raise ValueError('Incomplete original gripper sequence')
 return np.asarray(rows,np.float32).reshape(len(rows),len(joints)*2),np.asarray(intent,np.float64)

def install(task,gate,native,verify=True):
 if getattr(task,"_cx_native_dense_control",False):return
 original=task.take_dense_action
 if verify:
  proof=json.loads((D/'reports/target_collection_control_cpu_20261024.json').read_text());source=Path(inspect.getsourcefile(original))
  if hashlib.sha256(source.read_bytes()).hexdigest()!=proof['instrumented_base_sha256']:raise RuntimeError('Original dense-control runtime changed')
  if hashlib.sha256(Path(inspect.getsourcefile(task.robot.set_arm_joints)).read_bytes()).hexdigest()!=proof['original_robot_sha256']:raise RuntimeError('Original robot controls changed')
 def control(seq,save_freq=-1):
  r=task.robot
  if task.render_freq or len(r.left_arm_joints)!=6 or len(r.right_arm_joints)!=6 or not r.left_gripper or not r.right_gripper:return original(seq,save_freq)
  all_arms=r.left_arm_joints+r.right_arm_joints;all_grips=[j for side in ('left','right')for j,_,_ in getattr(r,side+'_gripper')]
  if len(set(map(id,all_arms+all_grips)))!=len(all_arms+all_grips):return original(seq,save_freq)
  freq=task.save_freq if save_freq==-1 else save_freq
  if freq is not None and (type(freq)is not int or freq<1):return original(seq,save_freq)
  if freq is not None:task._take_picture()
  def arm(name):
   v=seq[name]
   return (np.empty((0,6)),np.empty((0,6)))if v is None else (np.ascontiguousarray(v['position'],np.float64),np.ascontiguousarray(v['velocity'],np.float64))
  lp,lv=arm('left_arm');rp,rv=arm('right_arm');gl,li=grippers(r,seq['left_gripper'],'left');gr,ri=grippers(r,seq['right_gripper'],'right');length=max(len(lp),len(rp),len(gl),len(gr));begin=0
  while begin<length:
   end=length if freq is None else min(length,((begin+freq-1)//freq)*freq+1)
   dt=float(task.scene.get_timestep());tick=task._cx_tick
   with gate.read():dense,lvalue,rvalue=native.control(r.left_entity,r.right_entity,r.left_arm_joints,r.right_arm_joints,[j for j,_,_ in r.left_gripper],[j for j,_,_ in r.right_gripper],task.scene.physx_system,lp,lv,rp,rv,gl,gr,li,ri,begin,end,float(r.left_gripper_val),float(r.right_gripper_val),bool(task.save_data))
   r.left_gripper_val=np.float64(lvalue);r.right_gripper_val=np.float64(rvalue);task._cx_tick=tick+end-begin
   for index,command in enumerate(dense):task._cx_dense.append((tick+index,dt,command.copy()))
   if freq is not None and (end-1)%freq==0:task._update_render();task._take_picture()
   begin=end
  if freq is not None:task._take_picture()
  return True
 task.take_dense_action=control;task._cx_native_dense_control=True
