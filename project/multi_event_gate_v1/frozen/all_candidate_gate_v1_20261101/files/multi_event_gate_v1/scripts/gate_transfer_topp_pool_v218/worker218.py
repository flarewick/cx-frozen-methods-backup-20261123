"""No-NVIDIA-device CPU service; executes the unchanged mplib.Planner.TOPP."""
import sys,os,time,json,socket,hashlib,signal
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'gate_transfer_solver_v215'))
from protocol218 import receive,send
config=Path(sys.argv[1]);index=int(sys.argv[2]);cfg=json.loads(config.read_text());worker=cfg['workers'][index]
assert not list(Path('/dev').glob('nvidia*'))and not os.environ.get('CUDA_VISIBLE_DEVICES')
os.sched_setaffinity(0,[worker['CPU_core']])
import mplib.planner,toppra
from solver215 import install;install();toppra.setup_logging('CRITICAL')
original=mplib.planner.Planner.TOPP
source={str(file):hashlib.sha256(Path(file).read_bytes()).hexdigest()for file in [HERE/'worker218.py',HERE/'protocol218.py',Path(mplib.planner.__file__)]}
address=Path(worker['socket']);address.parent.mkdir(mode=0o700,parents=True,exist_ok=True);server=socket.socket(socket.AF_UNIX);server.bind(str(address));os.chmod(address,0o600);server.listen(32);server.settimeout(2)
Path(worker['ready']).write_text(json.dumps(dict(passed=True,no_GPU_devices=True,CPU_core=worker['CPU_core'],pid_in_container=os.getpid(),source_sha256=source))+'\n')
started=time.monotonic();requests=successes=0
try:
 while time.monotonic()-started<21600:
  try:client,_=server.accept()
  except socket.timeout:continue
  with client:
   client.settimeout(10)
   try:
    request=receive(client)
    if request['token']!=cfg['token']:raise PermissionError('CPU TOPP authentication failed')
    robot=SimpleNamespace(joint_vel_limits=request['velocity_limits'],joint_acc_limits=request['acceleration_limits'])
    requests+=1
    try:result=original(robot,request['path'],request['step'],request['verbose']);response=dict(ok=True,result=result);successes+=1
    except Exception as error:response=dict(ok=False,original_error=error)
    send(client,response)
    if requests%128==0:Path(worker['metrics']).write_text(json.dumps(dict(requests=requests,successes=successes,seconds=time.monotonic()-started))+'\n')
   except (EOFError,ConnectionError,socket.timeout):pass
finally:
 server.close();address.unlink(missing_ok=True)
