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
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

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

def verified_identity(host_pid,*,root_pid,table,verified_at_ns=None):
    """Called ONLY for an already verified live GPU PID in the guarded job tree."""
    info=identity(host_pid)
    if info is None or host_pid not in table or info['uid']!=os.getuid() or info['state']=='Z':
        return None
    if info['started_ticks']!=table[host_pid]['started_ticks']:
        return None
    ancestors={host_pid};pid=host_pid
    while pid in table and pid!=root_pid:
        pid=table[pid]['ppid']
        if pid in ancestors:return None
        ancestors.add(pid)
    if root_pid not in ancestors:return None
    parent=identity(info['ppid'])
    if parent is None or parent['uid']!=os.getuid():
        return None
    return dict(host_pid=host_pid,host_parent_pid=info['ppid'],
         child_started_ticks=info['started_ticks'],child_inner_pid=info['nspid'][-1],
         parent_inner_pid=parent['nspid'][-1],parent_started_ticks=parent['started_ticks'],
         pid_namespace=info['pid_namespace'],
         verified_at_ns=time.monotonic_ns() if verified_at_ns is None else verified_at_ns)

def retired(path,expected,verified,*,now_ns=None,pid_absent=absent,read_identity=identity):
    """No exemption for live, unknown, reused, failed or mismatched child IDs."""
    rejected=lambda reason:(set(),dict(accepted=False,reason=reason))
    path=Path(path)
    if not path.exists():return rejected('normal_child_exit_record_not_yet_present')
    try:
        if path.is_symlink() or path.stat().st_uid!=os.getuid():
            return rejected('untrusted_record_file')
        r=json.loads(path.read_text())
    except (OSError,ValueError,TypeError):
        return rejected('unreadable_exit_record')
    for key in ('attempt_nonce','gpu_uuid','protocol_identity','cpu_only_fixture'):
        if r.get(key)!=expected.get(key):
            return rejected('attempt_identity_mismatch_'+key)
    if r.get('version')!=1 or r.get('normal_exit')is not True or r.get('proc_absent_after_wait')is not True:
        return rejected('no_normal_reaped_child_proof')
    if (type(r.get('wait_status'))is not int or r['wait_status']!=0
            or type(r.get('exit_code'))is not int or r['exit_code']!=0
            or r.get('waited_child_inner_pid')!=r.get('child_inner_pid')):
        return rejected('waitpid_did_not_return_normal_child_zero')
    now=time.monotonic_ns() if now_ns is None else now_ns
    if (type(r.get('waited_monotonic_ns'))is not int
            or not expected['attempt_started_ns']<=r['waited_monotonic_ns']<=now):
        return rejected('exit_time_outside_this_attempt')
    matches=[]
    for pid,v in verified.items():
        fields=('child_inner_pid','child_started_ticks','parent_inner_pid','parent_started_ticks','pid_namespace')
        if any(r.get(k)!=v.get(k) for k in fields):continue
        if r['waited_monotonic_ns']<v['verified_at_ns']:continue
        if not pid_absent(pid):continue
        parent=read_identity(v['host_parent_pid'])
        if (parent is None or parent['uid']!=os.getuid()
                or parent['started_ticks']!=v['parent_started_ticks']
                or parent['nspid'][-1]!=v['parent_inner_pid']
                or parent['pid_namespace']!=v['pid_namespace']):
            continue
        matches.append(pid)
    if len(matches)!=1:return rejected('no_unique_previously_verified_absent_child_identity')
    return set(matches),dict(accepted=True,retired_host_pid=matches[0],record=str(path),
         exit_code=0,normal_child_reaped=True,proc_absent_rechecked=True,
         parent_alive_same_identity=True,waited_monotonic_ns=r['waited_monotonic_ns'])
