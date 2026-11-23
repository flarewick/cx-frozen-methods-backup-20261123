"""Current-policy CPU admission waiter and detached guarded official baseline host."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from policy import L, verify_rules, require_admission
from observer import admission_snapshot
sys.path.insert(0,str(L/'official_baseline_v14'))
sys.path.insert(0,str(L/'scripts'))
from common import read,write,sha,append
from host import checkpoints
from physical_release_evidence_v14 import restore_admitted_parent
from gpu_guard_bounded_metrics import stop_job

STATE=L/'outputs/official_baseline_live_v15.json'
child=None;stopping=False


def emit(phase,**values):
    root=Path(read(L/'outputs/official_baseline_latest.json')['root'])
    row=dict(phase=phase,campaign=root.name,wall_time=time.time(),controller_pid=os.getpid(),
             complete=len(list((root/'results').rglob('result.json'))),total=2000,**values)
    write(STATE,row);append(STATE.with_suffix('.jsonl'),row);print(json.dumps(row),flush=True)


def stop(sig,frame):
    global stopping
    stopping=True
    if child and child.poll()is None:
        child.send_signal(signal.SIGINT)


def main():
    global child
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    lock=(L/'outputs/official_baseline_live.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=(L/'outputs/official_baseline_pending.lock').open('a');fcntl.flock(pending,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic()
    while not stopping and time.monotonic()-start<86400:
        verify_rules()
        root=restore_admitted_parent(Path(read(L/'outputs/official_baseline_latest.json')['root']),checkpoints)
        points,count=checkpoints(root,read(root/'manifest.json'))
        if count==2000:
            emit('CPU_finalization_needed');return 0
        try:
            fleet=admission_snapshot()
        except (RuntimeError,FileNotFoundError)as exc:
            emit('waiting_observer',reason=str(exc));time.sleep(3);continue
        bound=read(root/'GPU_binding.json')['uuid'];selected=None;reasons=[]
        for c in sorted(fleet['cards'],key=lambda c:(c['uuid']!=bound,c['index'])):
            if not c['available']:continue
            try:require_admission(fleet,c['uuid']);selected=c;break
            except RuntimeError as exc:reasons.append(str(exc))
        if selected is None:
            emit('waiting_current_rule_admission',available_GPUs=fleet['available_indices'],
                 account_GPUs=fleet['own_gpu_count'],reasons=reasons,no_GPU_reserved=True)
            for _ in range(20):
                if stopping:break
                time.sleep(.5)
            continue
        if selected['uuid']!=bound:
            log=L/'logs'/('official_v15_migration_%d.log'%time.time_ns())
            with log.open('x')as f:child=subprocess.Popen([sys.executable,'-B',str(L/'resource_guard_v15/migrate.py'),'--gpu',str(selected['index'])],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            code=child.wait();child=None
            if code:raise RuntimeError('CPU physical carry failed: '+str(log))
            root=Path(read(L/'outputs/official_baseline_latest.json')['root'])
        if stopping:break
        log=L/'logs'/('official_v15_host_%d.log'%time.time_ns())
        with log.open('x')as f:child=subprocess.Popen([sys.executable,'-B',str(L/'resource_guard_v15/host.py'),'--gpu',str(selected['index']),'--resume'],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        emit('host_started',host_pid=child.pid,GPU=selected['index'],host_log=str(log))
        last=0
        while child.poll()is None:
            if stopping:
                try:child.wait(timeout=12)
                except subprocess.TimeoutExpired:stop_job(child)
                break
            if time.monotonic()-last>20:
                emit('monitoring',host_pid=child.pid,GPU=selected['index'],host_phase=read(root/'phase.json'),host_log=str(log));last=time.monotonic()
            time.sleep(1)
        code=child.wait();child=None
        if stopping:break
        at=Path(read(root/'current_attempt.json')['attempt'])
        if code==0:
            assert checkpoints(root,read(root/'manifest.json'))[1]==2000
            emit('final_CPU_audit')
            log=L/'logs'/('official_v15_final_audit_%d.log'%time.time_ns())
            with log.open('x')as f:child=subprocess.Popen(['bash',str(L/'scripts/run_cpu.sh'),'openpi','scripts/audit_official_baseline.py'],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            code=child.wait();child=None;assert code==0,str(log)
            emit('complete',final_audit=str(root/'final_audit.json'));return 0
        # CPU-only admission races are safe to reconsider; actual server-limit
        # stops, missing metrics and external signals never blindly restart.
        if not(at/'GPU_process.json').exists():
            failure=read(at/'failure.json')['error']
            if failure in ('Selected GPU is not idle','All-hours user cap of3 GPUs',
                           'Rule8: existing GPUs need180s paired utilization AND VRAM thresholds'):
                emit('CPU_admission_race',reason=failure);continue
            if failure=='Fleet changed since180s observation; wait for fresh observer':
                emit('CPU_admission_race',reason=failure);continue
        else:
            samples=[readrow for readrow in (json.loads(x)for x in(at/'resources.jsonl').read_text().splitlines())]
            actual=[v for v in samples if v.get('schema')=='governor180_v15']
            if samples[-1].get('own_gpu_released')and actual:
                reasons=actual[-1]['server_verdict']['reasons']
                if reasons==['Rule8: more than one user GPU fails paired180s thresholds']:
                    # Admission now requires ALL remaining account GPUs to meet
                    # the paired tier. A changed external workload is required.
                    emit('waiting_changed_rule8_admission',reason=reasons[0]);continue
        emit('diagnosis_required',error=read(at/'failure.json'),host_log=str(log));return 1
    emit('stopped');return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:emit('diagnosis_required',error=str(exc),traceback=traceback.format_exc());raise
