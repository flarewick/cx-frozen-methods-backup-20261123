"""Run original/new real guard loops against CPU children and fake GPU telemetry.
No NVIDIA command, allocation or GPU resource qualification is performed.
"""
import importlib.util,json,os
from pathlib import Path
import subprocess,sys,time,uuid
from types import SimpleNamespace
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1]
sys.path.insert(0,str(P/'deployment'));sys.path.insert(0,str(T/'resource_guard_v1'))
import user_container as container
import guard as original
import strict_guard as patched
from child_exit import identity
def sha(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()
def wait(path,process):
    start=time.monotonic()
    while not path.exists():
        if process.poll()is not None:raise RuntimeError('CPU fixture prematurely exited')
        if time.monotonic()-start>15:raise TimeoutError(path)
        time.sleep(.02)
def run_case(module,mode,root):
    root.mkdir(parents=True);nonce=uuid.uuid4().hex
    canonical=lambda p:str(container.CANONICAL/'VLA-Adapter/cx_new_direction_guide_TTA'/p.relative_to(P))
    cmd=container.command(['bash',canonical(T/'native_material_audit_v1/scripts/cpu.sh'),
            canonical(C/'exit_child_fixture.py'),'--root',canonical(root),'--nonce',nonce],mount_project=True)
    log=(root/'console.log').open('x');process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    real_table=module.process_table;real_clock=module.time
    saved_doctor=module.gpu_doctor
    saved_modules={k:sys.modules.get(k) for k in ('gpu_policy','gpu_metrics_strict')}
    count=0;frame=0;retired=False;released=False
    expected=dict(attempt_nonce='wrong' if mode=='wrong_nonce' else nonce,
        gpu_uuid='GPU-CPU-fixture',protocol_identity='a'*64,cpu_only_fixture=True,
        attempt_started_ns=time.monotonic_ns())
    try:
        wait(root/'child_ready.json',process)
        ready=json.loads((root/'child_ready.json').read_text())
        table=real_table();own=original.descendants(process.pid,table)
        candidates=[pid for pid in own if (identity(pid) is not None and
              identity(pid)['nspid'][-1]==ready['child']['pid'] and
              identity(pid)['started_ticks']==ready['child']['started_ticks'])]
        assert len(candidates)==1;child=candidates[0]
        def table_stub():
            real=real_table()
            for r in real.values():r.update(cpu=200.,rss=64*2**20)
            return real
        def release():
            nonlocal released
            if not released:
                (root/'release_child.json').write_text('{}\n');released=True
                wait(root/'parent_cleanup_waiting.json',process)
        def query(args):
            nonlocal frame
            if '--query-gpu=memory.total'in args:return '24564\n'
            if '--query-gpu=utilization.gpu,utilization.memory,memory.used'in args:
                frame+=1
                return '80, 70, '+str(23000 if mode=='global_VRAM' and frame>=2 else 17765)+'\n'
            if '--query-compute-apps=gpu_uuid,pid,used_gpu_memory'in args:
                ids=[child]+([os.getpid()] if mode in ('mixed_unknown','foreign_first') and (frame>=2 or mode=='foreign_first') else [])
                return ''.join('GPU-CPU-fixture, '+str(pid)+', 16384\n' for pid in ids)
            if '--query-compute-apps=gpu_uuid,pid'in args:return 'GPU-CPU-fixture, '+str(child)+'\n'
            raise AssertionError(args)
        def fleet():
            ids=[child]+([os.getpid()] if mode in ('mixed_unknown','foreign_first') and (frame>=2 or mode=='foreign_first') else [])
            return dict(cards=[dict(uuid='GPU-CPU-fixture',pids=ids)],
                own_gpu_count=4 if mode=='global_account' and frame>=2 else 1,
                available_indices=[])
        def pmon(gpu):
            if frame==1:return {child:80}
            if mode=='metric_race':release()
            return {} if mode in ('metric_race','live_missing') else {child:80}
        def monotonic():
            nonlocal count
            count+=1
            return {1:0.,2:1.,3:2.}.get(count,281.+count)
        def sample(row):
            nonlocal retired
            if frame==1:
                if mode not in ('metric_race','live_missing','foreign_first'):release()
            else:
                if row.get('normal_child_exit_proof',{}).get('accepted'):
                    retired=True
                    if mode not in ('mixed_unknown','global_account','global_VRAM','wrong_nonce'):
                        (root/'finish_cpu_cleanup.json').write_text('{}\n')
                        assert process.wait(timeout=5)==0
            return None
        module.process_table=table_stub
        module.gpu_doctor=SimpleNamespace(query=query)
        module.time=SimpleNamespace(monotonic=monotonic,sleep=lambda s:time.sleep(.02))
        sys.modules['gpu_policy']=SimpleNamespace(inventory=fleet,MAX_GPUS=3)
        sys.modules['gpu_metrics_strict']=SimpleNamespace(process_utilization=pmon,diagnostics=lambda:dict(CPU_STUB=True))
        error=None
        try:
            kwargs=dict(gpu=0,deadline=7200,on_sample=sample,
                min_cumulative_utilization=40,max_memory_mib=22528,prepared_process=process)
            if module is patched:
                kwargs.update(child_exit_proof=root/'normal_child_exit.json',child_exit_identity=expected)
            module.guarded_command(cmd,dict(uuid='GPU-CPU-fixture'),root/'resources.jsonl',**kwargs)
        except (RuntimeError,SystemExit,ValueError) as exc:error=type(exc).__name__+': '+str(exc)
        telemetry=[json.loads(line) for line in (root/'resources.jsonl').read_text().splitlines()]
        stopped=[r['stopped'] for r in telemetry if 'stopped'in r]
        if module is patched and mode in ('normal','metric_race'):
            assert error is None and retired and not stopped,(mode,error,stopped)
        else:
            assert error is not None and stopped,(mode,error,stopped)
            if mode=='live_missing':assert any('unverified or below'in x for x in stopped)
            if mode in ('wrong_nonce','mixed_unknown','foreign_first') or module is original:
                assert any('occupied before'in x for x in stopped),(mode,stopped)
            if mode=='global_account':assert any('three-GPU'in x for x in stopped)
            if mode=='global_VRAM':assert any('memory exceeded'in x for x in stopped)
        return dict(guard='patched' if module is patched else 'original',mode=mode,
            expected_success=module is patched and mode in ('normal','metric_race'),
            error=error,stopped=stopped,CPU_ONLY_FAKE_GPU_METRICS=True,
            certificate_retirement_observed=retired)
    finally:
        module.process_table=real_table;module.gpu_doctor=saved_doctor;module.time=real_clock
        for k,v in saved_modules.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v
        if process.poll()is None:original.stop_job(process)
        log.close()
def main():
    O=T/'outputs/native_nextchunk_v1';root=O/'guard_fixtures'/uuid.uuid4().hex
    records=[]
    for i,(mod,mode) in enumerate([(original,'normal'),(patched,'normal'),(patched,'metric_race'),
          (patched,'live_missing'),(patched,'wrong_nonce'),(patched,'mixed_unknown'),
          (patched,'foreign_first'),(patched,'global_account'),(patched,'global_VRAM')]):
        records.append(run_case(mod,mode,root/('case'+str(i))))
        print('CPU_GUARD_SCENARIO',mod.__name__,mode,'PASS',flush=True)
    report=dict(passed=True,scenarios=len(records),records=records,
        no_GPU_started=True,NO_GPU_RESOURCE_QUALIFICATION=True,
        source_sha256={str(C/n):sha(C/n) for n in ('strict_guard.py','child_exit.py','guard_checks_host.py','exit_child_fixture.py')},
        scope='Original/new full guard function on actual CPU PID namespaces/normalwait proof with explicitly synthetic GPU/CPU utilization and memory. Validates exit-race handling and fail-closed arrival/live/unknown/nonce/account/VRAM controls, not NVIDIA telemetry or actual server resource performance.')
    path=O/'guard_cpu_checks.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
    print('CHILD_EXIT_FULL_GUARD_CPU_SCENARIOS_PASS',len(records),'NO_GPU',flush=True)
if __name__=='__main__':main()
