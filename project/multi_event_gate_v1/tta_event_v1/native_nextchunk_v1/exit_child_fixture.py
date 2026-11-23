"""CPU-only PID-namespace fixture, no CUDA or utilization qualification."""
import argparse,os,time
from pathlib import Path
from child_exit import identity,atomic,wait_and_record
def wait(path,seconds=20):
    start=time.monotonic()
    while not path.exists():
        if time.monotonic()-start>seconds:raise TimeoutError('CPU fixture handshake deadline')
        time.sleep(.02)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--nonce',required=True);a=p.parse_args();root=a.root;root.mkdir(parents=True,exist_ok=True)
    child=os.fork()
    if child==0:
        try:
            atomic(root/'child_ready.json',dict(child=identity(os.getpid()),parent=identity(os.getppid()),CPU_ONLY_FIXTURE=True))
            wait(root/'release_child.json')
            os._exit(0)
        except BaseException:os._exit(1)
    code=wait_and_record(child,root/'normal_child_exit.json',attempt_nonce=a.nonce,
         gpu_uuid='GPU-CPU-fixture',protocol_identity='a'*64,cpu_only_fixture=True)
    atomic(root/'parent_cleanup_waiting.json',dict(parent=identity(os.getpid()),child_code=code,CPU_ONLY_FIXTURE=True))
    wait(root/'finish_cpu_cleanup.json')
    raise SystemExit(code)
if __name__=='__main__':main()
