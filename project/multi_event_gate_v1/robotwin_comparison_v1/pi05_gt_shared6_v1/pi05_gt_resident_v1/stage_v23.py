"""CPU preload, short-lived explicit GPU child; all launch admission is host-owned."""
from pathlib import Path
import os,sys,time,traceback,json
from multiprocessing.connection import Listener
from .bootstrap import install
install()
from .common import read,write,verify,identity
root=Path(sys.argv[1]);spec=read(root/'protocol.json');verify(spec);launch=read(root/'launch.json')
import torch
selected=os.environ.get('CUDA_VISIBLE_DEVICES','');os.environ['CUDA_VISIBLE_DEVICES']=''
from .preflight import prepare
prepared=prepare(spec,root)
address='\0rwc_pi05_gt_'+launch['nonce'];listener=Listener(address,family='AF_UNIX',authkey=bytes.fromhex(launch['key']));write(root/'CPU_ready.json',dict(passed=True,CUDA_initialized=False));connection=listener.accept();permit=connection.recv();assert permit['GO']and permit['uuid']==selected and time.time()-permit['wall_time']<5;verify(spec)
pid=os.fork()
if pid==0:
 try:
  connection.close();listener.close();os.environ['CUDA_VISIBLE_DEVICES']=selected
  from .runtime_v23_verified_bootstrap import execute
  execute(spec,root,dict(gpu_uuid=selected),prepared,root,exit_success=lambda:os._exit(0))
 except BaseException:
  write(root/'GPU_failure.json',dict(error=traceback.format_exc()));traceback.print_exc();os._exit(1)
connection.send(dict(child_created=True));connection.close();listener.close();_,status=os.waitpid(pid,0);code=os.waitstatus_to_exitcode(status);write(root/'GPU_actual_exit.json',dict(actual_returncode=code));prepared['storage'].cleanup();raise SystemExit(code)
