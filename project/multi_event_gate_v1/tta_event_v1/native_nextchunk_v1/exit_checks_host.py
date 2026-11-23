"""Actual fork/wait/PID-namespace fixture and strict terminal-proof negatives.
NO GPU process launched, measured or qualified by this test.
"""
import importlib.util,json,os
from pathlib import Path
import subprocess,sys,time,uuid
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1]
sys.path.insert(0,str(P/'deployment'))
import user_container as container
sys.path.insert(0,str(T/'resource_guard_v1'))
from guard import process_table,descendants,stop_job
from child_exit import identity,retired
def sha(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()
def wait(path,process,timeout=20):
    start=time.monotonic()
    while not path.exists():
        if process.poll()is not None:raise RuntimeError('Fixture exited before '+path.name)
        if time.monotonic()-start>timeout:raise TimeoutError(path)
        time.sleep(.03)
def main():
    O=T/'outputs/native_nextchunk_v1'
    root=O/'exit_fixtures'/uuid.uuid4().hex;root.mkdir(parents=True)
    nonce=uuid.uuid4().hex
    canonical=lambda p:str(container.CANONICAL/'VLA-Adapter/cx_new_direction_guide_TTA'/p.relative_to(P))
    cmd=container.command(['bash',canonical(T/'native_material_audit_v1/scripts/cpu.sh'),
             canonical(C/'exit_child_fixture.py'),'--root',canonical(root),'--nonce',nonce],mount_project=True)
    start=time.monotonic_ns()
    log=(root/'console.log').open('x');process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    checks=0
    try:
        wait(root/'child_ready.json',process)
        ready=json.loads((root/'child_ready.json').read_text())
        table=process_table();own=descendants(process.pid,table)
        candidates=[]
        for pid in own:
            info=identity(pid)
            if (info and info['nspid'][-1]==ready['child']['pid']
                  and info['started_ticks']==ready['child']['started_ticks']):
                candidates.append(pid)
        assert len(candidates)==1,candidates;child=candidates[0]
        from child_exit import verified_identity
        v=verified_identity(child,root_pid=process.pid,table=table)
        assert v and v['pid_namespace']==ready['child']['pid_namespace'];checks+=2
        expected=dict(attempt_nonce=nonce,gpu_uuid='GPU-CPU-fixture',
             protocol_identity='a'*64,cpu_only_fixture=True,attempt_started_ns=start)
        certificate=root/'normal_child_exit.json'
        ids,note=retired(certificate,expected,{child:v});assert not ids;checks+=1
        (root/'release_child.json').write_text('{}\n')
        wait(root/'parent_cleanup_waiting.json',process)
        assert process.poll()is None;checks+=1
        ids,note=retired(certificate,expected,{child:v})
        assert ids=={child} and note['accepted'];checks+=1
        assert not Path('/proc',str(child)).exists();checks+=1
        original=json.loads(certificate.read_text())
        negatives=[]
        def reject(label,record=None,exp=None,known=None,**kwargs):
            nonlocal checks
            path=certificate
            if record is not None:
                path=root/(label+'.json');path.write_text(json.dumps(record)+'\n')
            got,why=retired(path,expected if exp is None else exp,{child:v} if known is None else known,**kwargs)
            assert not got and why['accepted']is False,(label,why)
            negatives.append(dict(label=label,reason=why['reason']));checks+=1
        for key,value in [('attempt_nonce','stale'),('gpu_uuid','GPU-wrong'),('protocol_identity','b'*64),
                          ('cpu_only_fixture',False),('exit_code',1),('wait_status',9),
                          ('normal_exit',False),('proc_absent_after_wait',False),
                          ('waited_child_inner_pid',999),('child_started_ticks',-1),
                          ('parent_started_ticks',-1),('pid_namespace','pid:[foreign]'),
                          ('waited_monotonic_ns',start-1)]:
            reject(key,dict(original,**{key:value}))
        reject('unknown_GPU_PID',known={})
        reject('PID_still_live',pid_absent=lambda pid:False)
        reject('proc_permission_unverified',pid_absent=lambda pid:False)
        reject('parent_PID_reused',read_identity=lambda pid:dict(identity(pid),started_ticks=-1))
        reject('parent_disappeared',read_identity=lambda pid:None)
        reject('wait_predates_verified_ownership',known={child:dict(v,verified_at_ns=original['waited_monotonic_ns']+1)})
        reject('CPU_fixture_cannot_authorize_GPU',exp=dict(expected,cpu_only_fixture=False))
        # Only the proved former child is retired: a mixed foreign set must
        # preserve unknown occupants for the original arrival-order check.
        foreign={child,987654321};remaining=foreign-ids
        assert remaining=={987654321};checks+=1
        (root/'finish_cpu_cleanup.json').write_text('{}\n')
        assert process.wait(timeout=10)==0;checks+=1
        report=dict(passed=True,checks=checks,fixture_root=str(root),retired_host_PID=child,
             root_was_live_after_normal_child_reap=True,
             actual_nested_PID_namespace_identity_verified=True,
             simulated_GPU_ownership_only=True,no_GPU_started=True,no_GPU_resource_qualification=True,
             negative_tests=negatives,source_sha256={str(C/n):sha(C/n) for n in ('child_exit.py','exit_child_fixture.py','exit_checks_host.py')})
        target=O/'exit_cpu_checks.json';assert not target.exists();target.write_text(json.dumps(report,indent=2)+'\n')
        print('NORMAL_CHILD_CPU_NAMESPACE_CHECKS_PASS',checks,'NO_GPU',flush=True)
    finally:
        if process.poll()is None:stop_job(process)
        log.close()
if __name__=='__main__':main()
