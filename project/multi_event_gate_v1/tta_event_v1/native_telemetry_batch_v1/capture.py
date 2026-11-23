"""Read-only SDK getter batching; no clock/controller or simulation side effects."""
from pathlib import Path
import importlib,hashlib,json,sys
import numpy as np
T=Path(__file__).resolve().parents[1]
def load():
 p=T/'outputs/native_telemetry_batch_v1/build/build.json';m=json.loads(p.read_text());binary=Path(m['binary'])
 assert hashlib.sha256(binary.read_bytes()).hexdigest()==m['binary_sha256']and hashlib.sha256((Path(__file__).parent/'trace.cpp').read_bytes()).hexdigest()==m['source_sha256']
 sys.path.insert(0,str(binary.parent));return importlib.import_module('cx_tta_command_telemetry')
def capture(robot,native):
 if len(robot.left_arm_joints)!=6 or len(robot.right_arm_joints)!=6:raise ValueError('Expected six joints per arm')
 arms=tuple(robot.left_arm_joints)+tuple(robot.right_arm_joints);fingers=[];counts=[]
 for group in(robot.left_gripper,robot.right_gripper):
  if any(item is None or item[0]is None for item in group):raise ValueError('Missing gripper drive')
  fingers.extend(item[0]for item in group);counts.append(len(group))
 command=np.array([robot.get_left_gripper_val(),robot.get_right_gripper_val()],np.float64)
 return native.capture(arms,fingers,counts,command)
