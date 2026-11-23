"""CPU preflight/plan by default; --run executes one strictly guarded native job."""
import argparse,fcntl,json,os,subprocess,sys,time,uuid
from pathlib import Path
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1]
sys.path.insert(0,str(P/'deployment'));sys.path.insert(0,str(T/'resource_guard_v1'));sys.path.insert(0,str(T/'native_nextchunk_v1'))
sys.path.insert(0,str(T))
import user_container as container
import gpu_doctor
from native_resource_policy_v1 import guard
def read(p):return json.loads(Path(p).read_text())
def canonical(p):return str(container.CANONICAL/'VLA-Adapter/cx_new_direction_guide_TTA'/Path(p).resolve().relative_to(P))
def cpu(module,*args):
    cmd=container.command(['bash',canonical(C/'scripts/cpu.sh'),'-m',module,*args],mount_project=True)
    result=subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(result.stdout,end='',flush=True);return result.stdout
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--run',action='store_true');p.add_argument('--cpu-proof',action='store_true');a=p.parse_args()
    from native_feedback_confirm_v5.cpu_placement import apply
    apply(read(T/'outputs/native_feedback_confirm_v5/cpu_placement_plan.json'))
    info=gpu_doctor.gpu_snapshot(a.gpu)
    result=cpu('native_expert_full_cache_v5.prepare','--gpu-uuid',info['uuid'])
    plan=json.loads(next(s.split(' ',1)[1] for s in result.splitlines() if s.startswith('FACTOR_PLAN ')))
    root=P/Path(plan['root']).relative_to(container.CANONICAL/'VLA-Adapter/cx_new_direction_guide_TTA')
    if not plan['remaining']:cpu('native_expert_full_cache_v5.audit','--dataset',plan['root']);return
    print('INITIAL_FRAME_GPU_PLAN',json.dumps(dict(gpu=info,remaining=plan['remaining'],total=96,reused_complete_original_cases=4,recovered_terminal_cases=28,recovered_initialization_failures=1,new_GPU_cases=68,
         actual_VLA_CPU_preload_required=True,simulation_lanes=6,original_expert_native_actions=True,new_independent_scenes=0,GPU_started=False)),flush=True)
    if a.cpu_proof:
        proof_command=container.command(['bash',canonical(C/'scripts/gpu.sh'),'--gpu-uuid',info['uuid'],'--cpu-proof'],mount_project=True)
        raise SystemExit(subprocess.run(proof_command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void')).returncode)
    if not a.run:return
    spec=read(T/'outputs/native_expert_full_cache_v5/protocol.json')
    assert info['uuid']==spec['required_gpu_uuid'],'No hardware mixing with prior source'
    proof=read(T/'outputs/native_expert_full_cache_v5/cpu_preload_proof.json')
    assert proof['passed'] and proof['cpu_fork_child_passed'] and proof['source_sha256']==spec['source_sha256']
    if (root/'host_attempts').exists():raise ValueError('Initial-frame attempt exists; inspect rather than retry')
    if (root/'resource_blocked.json').exists():raise ValueError('No unchanged retry after failed resource qualification')
    cert=container.WORK/('gpu-doctor-'+str(a.gpu)+'.json')
    if not cert.exists():cert=T/'outputs/native_feedback_confirm_v5'/('gpu-doctor-'+str(a.gpu)+'.json')
    certificate=read(cert)
    assert certificate['uuid']==info['uuid'] and certificate['build_mtime_ns']==(container.WORK/'stage-bootstrap.ok').stat().st_mtime_ns
    from native_feedback_confirm_v5.cpu_placement import apply
    placement=read(T/'outputs/native_feedback_confirm_v5/cpu_placement_plan.json')
    assert placement['gpu_uuid']==info['uuid'] and proof['actual_VLA_loaded_on_CPU']
    assert proof['CPU_affinity']==placement['selected_logical_CPUs']
    apply(placement)
    locks=container.WORK/'gpu_locks';locks.mkdir(exist_ok=True)
    with (locks/'account_admission.lock').open('a') as admission,(locks/('gpu_'+str(a.gpu)+'.lock')).open('a') as card:
        fcntl.flock(admission,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(card,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_doctor.check_admission(gpu_doctor.gpu_snapshot(a.gpu))
        nonce=uuid.uuid4().hex;attempt=root/'host_attempts'/nonce;attempt.mkdir(parents=True);barrier=attempt/'barrier'
        cmd=container.command(['bash',canonical(C/'scripts/gpu.sh'),'--gpu-uuid',info['uuid'],
            '--barrier',canonical(barrier),'--nonce',nonce],mount_project=True)
        split=cmd.index('--');cmd[split:split]=gpu_doctor.driver_options(info)
        console=(attempt/'console.log').open('x');process=subprocess.Popen(cmd,stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
        began=time.monotonic();started_ns=time.monotonic_ns()
        try:
            from gpu_policy import inventory,require_admission
            while not (barrier/'ready.json').exists():
                if process.poll()is not None:raise ValueError('CPU preloading failed before GPU admission')
                require_admission(inventory(),info['uuid'])
                if time.monotonic()-began>240:raise TimeoutError('CPU preparation deadline')
                time.sleep(.5)
            ready=read(barrier/'ready.json')
            assert ready['os_threads']==1 and not ready['cuda_initialized'] and ready['cpu_fork_child_passed']
            assert ready['protocol_identity']==spec['identity'] and ready['source_sha256']==spec['source_sha256']
            (attempt/'launch.json').write_text(json.dumps(dict(command=cmd,gpu=info,protocol_identity=spec['identity'],data_identity=plan['identity'],attempt_nonce=nonce),indent=2)+'\n')
            def permit():
                gpu_doctor.check_admission(gpu_doctor.gpu_snapshot(a.gpu))
                (barrier/'permit.json').write_text(json.dumps(dict(gpu_uuid=info['uuid'],attempt_nonce=nonce,protocol_identity=spec['identity']))+'\n')
            released=False;owned_permitted=False
            def sample(row):
                nonlocal released,owned_permitted
                if len(row['gpu_pids'])>1:return 'Exactly one GPU process required'
                if row['gpu_pids'] and not released:fcntl.flock(admission,fcntl.LOCK_UN);released=True
                if row['gpu_pids'] and not owned_permitted and (barrier/'cuda_ready.json').exists():
                    (barrier/'owned_pid_permit.json').write_text(json.dumps(dict(gpu_uuid=info['uuid'],host_GPU_PID=row['gpu_pids'][0]))+'\n')
                    owned_permitted=True
            guard.guarded_command(cmd,info,attempt/'resources.jsonl',gpu=a.gpu,deadline=7200,
                on_sample=sample,min_cumulative_utilization=34,max_memory_mib=22528,
                prepared_process=process,before_monitor=permit,child_exit_proof=barrier/'normal_child_exit.json',
                child_exit_identity=dict(attempt_nonce=nonce,gpu_uuid=info['uuid'],protocol_identity=spec['identity'],cpu_only_fixture=False,attempt_started_ns=started_ns))
        except BaseException as exc:
            (attempt/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,reason=str(exc),automatic_retry=False))+'\n')
            if any(term in str(exc).lower() for term in ('utilization','vram','memory','cpu utilization','rss')):
                (root/'resource_blocked.json').write_text(json.dumps(dict(reason=str(exc),attempt=str(attempt),automatic_retry=False))+'\n')
            if process.poll()is None:
                known=guard.stop_job(process);guard.record_release(process,info,began,attempt/'failed_release.jsonl',known)
            raise
        finally:console.close()
    cpu('native_expert_full_cache_v5.audit','--dataset',plan['root'])
if __name__=='__main__':main()
