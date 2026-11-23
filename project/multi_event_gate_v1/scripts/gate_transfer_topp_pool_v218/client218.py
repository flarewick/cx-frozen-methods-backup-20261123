"""Move only original pure-CPU TOPP call; fail closed on transport errors."""
import os,json,socket,threading,itertools,functools
from pathlib import Path
from protocol218 import receive,send
INSTALLED=False;ORIGINAL=None;CLIENT=None
class TransportFailure(BaseException):pass
class Client:
 def __init__(self,path):
  self.path=Path(path);self.config=json.loads(self.path.read_text());self.sequence=itertools.count();self.workers=self.config['workers']
  assert len(self.workers)==4
 def call(self,robot,path,step,verbose):
  row=self.workers[next(self.sequence)%len(self.workers)]
  try:
   with socket.socket(socket.AF_UNIX)as sock:
    sock.settimeout(10);sock.connect(row['socket']);send(sock,dict(token=self.config['token'],path=path,velocity_limits=robot.joint_vel_limits,acceleration_limits=robot.joint_acc_limits,step=step,verbose=verbose));result=receive(sock)
  except Exception as error:raise TransportFailure('CPU TOPP IPC failed; cancel entire evaluation')from error
  if not result['ok']:raise result['original_error']
  return result['result']
def install():
 global INSTALLED,ORIGINAL,CLIENT
 if INSTALLED:return
 import mplib.planner
 ORIGINAL=mplib.planner.Planner.TOPP
 @functools.wraps(ORIGINAL)
 def topp(self,path,step=.1,verbose=False):
  global CLIENT
  if CLIENT is None:
   config=os.environ.get('CX_TOPP218_CONFIG')
   if not config:raise TransportFailure('CPU TOPP configuration required; refuse local silent fallback')
   CLIENT=Client(config)
  return CLIENT.call(self,path,step,verbose)
 mplib.planner.Planner.TOPP=topp;INSTALLED=True
