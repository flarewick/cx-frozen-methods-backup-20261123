"""Normal child wait proof across PID namespaces, bound to one fresh attempt."""
import json
import os
from pathlib import Path
import time

def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=p.stat()
        tail=(p/'stat').read_text().rsplit(')',1)[1].split()
        status=(p/'status').read_text()
        nspid=[int(x) for x in next(line for line in status.splitlines() if line.startswith('NSpid:')).split()[1:]]
        namespace=os.readlink(p/'ns/pid')
        return dict(pid=int(pid),uid=stat.st_uid,ppid=int(tail[1]),
             started_ticks=int(tail[19]),state=tail[0],nspid=nspid,pid_namespace=namespace)
    except (OSError,ValueError,IndexError,StopIteration):
        return None

def absent(pid):
    try:os.stat('/proc/'+str(pid));return False
    except FileNotFoundError:return True
    except OSError:return False

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('One child-exit record per attempt; do not overwrite')
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush()
        os.replace(temporary,path)
        # Publish the actual reaped-child proof before potentially slow fsync.
        # The CPU parent still waits for durability before validation/cleanup.
        os.fsync(f.fileno())

def wait_and_record(child,path,*,attempt_nonce,gpu_uuid,protocol_identity,cpu_only_fixture=False):
    """The actual CPU parent must call this before CPU-only scratch cleanup."""
    if type(child)is not int or child<=0 or child==os.getpid():
        raise ValueError('A direct child PID required')
    info=identity(child);parent=identity(os.getpid())
    if info is None or parent is None or info['ppid']!=os.getpid():
        raise ValueError('Child identity unavailable before real waitpid')
    waited,status=os.waitpid(child,0)
    code=os.waitstatus_to_exitcode(status)
    gone=absent(child)
    record=dict(version=1,attempt_nonce=attempt_nonce,gpu_uuid=gpu_uuid,
         protocol_identity=protocol_identity,cpu_only_fixture=cpu_only_fixture,
         parent_inner_pid=os.getpid(),parent_started_ticks=parent['started_ticks'],
         child_inner_pid=child,child_started_ticks=info['started_ticks'],
         pid_namespace=info['pid_namespace'],waited_child_inner_pid=waited,
         wait_status=status,exit_code=code,normal_exit=os.WIFEXITED(status) and code==0,
         proc_absent_after_wait=gone,waited_monotonic_ns=time.monotonic_ns())
    atomic(path,record)
    if waited!=child or not gone:
        raise ValueError('Child is not proven reaped/absent')
    return code
