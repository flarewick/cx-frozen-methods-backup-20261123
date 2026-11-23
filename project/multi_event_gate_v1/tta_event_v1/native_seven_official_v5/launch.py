import subprocess,sys
from pathlib import Path
from .common import O,T,write,read,verify
def main():
 assert not list((O/'batches/000/runs').glob('*'))
 process=subprocess.Popen([sys.executable,'-B','-u','-m','native_seven_official_v5.host']);code=process.wait()
 write(O/'actual_host_wait_exit.json',dict(exit_code=code,actual_Popen_wait_return=True,pid=process.pid,proc_absent_after_wait=not Path('/proc',str(process.pid)).exists()))
 raise SystemExit(code)
if __name__=='__main__':main()
