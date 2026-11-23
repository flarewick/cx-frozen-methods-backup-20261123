"""Actual frozen-model CPU preload, one-thread fork proof, guarded GPU child."""
import argparse,os,sys,time,traceback,uuid
from pathlib import Path
from native_expert_full_cache_v8.paths import O,P,read,write,verify,dataset,sha
from native_expert_full_cache_v8.prepare import prepare as freeze

def prepare(spec,root):
    import torch,numpy as np
    from native_feedback_confirm_v5.cpu_placement import apply
    from cx_tta_online.native_entry_storage import NativeEntryStorage
    from cx_tta.evaluate_b2 import resolve_tokenizer
    from adapter_cache_cpu import load_cpu_policy,state_digest
    from cx_tta_feedback_confirm5.query import build_head
    from cx_tta_feedback_confirm5.schedule import instrument
    apply(read(spec['CPU_placement']));torch.set_num_threads(1)
    session=root/'sessions'/('process_'+uuid.uuid4().hex[:12]);session.mkdir(parents=True)
    class Storage(NativeEntryStorage):
        def prepare_runtime(self,folder):
            runtime=super().prepare_runtime(folder);file=runtime/'envs/_base_task.py';file.write_text(instrument(file.read_text()));return runtime
    storage=Storage(runtime_source=Path(spec['runtime_source']))
    try:
        runtime=storage.prepare_runtime(session);storage.configure_runtime(session,runtime,resolve_tokenizer())
        policy,info=load_cpu_policy(verify_state=True)
        proof=next(__import__('json').loads(l.split(' ',1)[1])for l in(P/'deployment/reports/target_adapter_cache_cpu_arrow_nvml_20261024.log').read_text().splitlines()if l.startswith('ADAPTER_CACHE_CPU_PROOF '))
        assert proof['passed']and proof['helper_sha256']==sha(P/'deployment/adapter_cache_cpu.py')
        converted=state_digest(policy._model);assert converted==proof['converted_state_sha256']
        before=torch.random.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):heads=[build_head(spec['training'])for _ in range(6)]
        assert torch.equal(before,torch.random.get_rng_state())
        with np.load(spec['projection'],allow_pickle=False)as z:projection={k:z[k].copy()for k in z.files}
        # Hash immutable recovery files before acquiring a CUDA context. Child inherits
        # these verified CPU JSON rows; final CPU audit rechecks actual files again.
        from native_expert_full_cache_v8.recovery import recovered_rows
        recovery_started=time.monotonic();recovery_rows=recovered_rows(spec,root)
        recovery_seconds=time.monotonic()-recovery_started
        assert not torch.cuda.is_initialized()and len(recovery_rows)==70
        print('RECOVERY_VERIFIED_BEFORE_CUDA',len(recovery_rows),recovery_seconds,flush=True)
        import sapien
        from native_telemetry_batch_v1.capture import load as load_telemetry
        telemetry_native=load_telemetry()
        from native_expert_full_cache_v8.runtime import execute
        from native_expert_full_cache_v8.audit import audit
        assert not torch.cuda.is_initialized()and len(list(Path('/proc/self/task').iterdir()))==1
        return dict(policy=policy,heads=heads,projection=projection,storage=storage,runtime=runtime,
                    session=session,converted_state_sha256=converted,recovered_rows=recovery_rows,
                    recovery_prepared_before_CUDA=True,recovery_CPU_CUDA_initialized=False,recovery_CPU_seconds=recovery_seconds,telemetry_native=telemetry_native)
    except BaseException:storage.cleanup();raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);p.add_argument('--barrier',type=Path)
    p.add_argument('--nonce');p.add_argument('--cpu-proof',action='store_true');a=p.parse_args()
    selected=os.environ.get('CUDA_VISIBLE_DEVICES','');os.environ['CUDA_VISIBLE_DEVICES']=''
    spec=freeze();root,manifest=dataset(spec,a.gpu_uuid);verify(spec)
    assert not(root/'GPU_native_complete.json').exists(),'Preserve existing cache'
    import torch
    prepared=prepare(spec,root)
    try:
        child=os.fork()
        if child==0:
            assert not torch.cuda.is_initialized() and not torch.cuda._is_in_bad_fork() and int(torch.arange(10).sum())==45
            os._exit(0)
        _,status=os.waitpid(child,0);assert os.waitstatus_to_exitcode(status)==0
        ready=dict(passed=True,protocol_identity=spec['identity'],source_sha256=spec['source_sha256'],
            os_threads=len(list(Path('/proc/self/task').iterdir())),cuda_initialized=torch.cuda.is_initialized(),
            cpu_fork_child_passed=True,CPU_affinity=sorted(os.sched_getaffinity(0)),actual_VLA_loaded_on_CPU=True,
            converted_state_sha256=prepared['converted_state_sha256'],original_cases=96,new_GPU_cases=26,recovered_terminal_cases=70,six_CPU_heads_loaded=True,teacher_labels_excluded_from_policy_API=True,recovery_prepared_before_CUDA=prepared['recovery_prepared_before_CUDA'],
            recovery_CPU_seconds=prepared['recovery_CPU_seconds'],recovery_CPU_CUDA_initialized=prepared['recovery_CPU_CUDA_initialized'],
            recovered_terminal_rows=len(prepared['recovered_rows']))
        if a.cpu_proof:
            write(O/'cpu_preload_proof.json',ready);print('EXPERT_POLICY_CPU_PREFLIGHT_PASS',flush=True);return
        assert a.barrier is not None and a.nonce;write(a.barrier/'ready.json',ready);began=time.monotonic()
        while not(a.barrier/'permit.json').exists():
            if time.monotonic()-began>240:raise TimeoutError('No fresh GPU admission permit')
            time.sleep(.05)
        permit=read(a.barrier/'permit.json')
        assert permit['gpu_uuid']==a.gpu_uuid and permit['attempt_nonce']==a.nonce and permit['protocol_identity']==spec['identity']
        child=os.fork()
        if child==0:
            try:
                os.environ['CUDA_VISIBLE_DEVICES']=selected
                assert not torch.cuda.is_initialized() and not torch.cuda._is_in_bad_fork()
                from native_expert_full_cache_v8.runtime import execute
                execute(spec,root,manifest,prepared,a.barrier)
            except BaseException:traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
            os._exit(0)
        from native_nextchunk_v1.child_exit import wait_and_record
        code=wait_and_record(child,a.barrier/'normal_child_exit.json',attempt_nonce=a.nonce,gpu_uuid=a.gpu_uuid,
                             protocol_identity=spec['identity'],cpu_only_fixture=False)
        if code!=0:raise SystemExit(code)
        assert not torch.cuda.is_initialized()
        from native_expert_full_cache_v8.audit import audit
        check=audit(spec,root,*prepared['heads'][0],prepared['projection'])
        from native_expert_full_cache_v8.telemetry_audit import audit_bridge
        bridge_check=audit_bridge(spec,root)
        verify(spec)
        write(prepared['session']/'source_storage_audit.json',prepared['storage'].verify_source())
        write(root/'result.json',dict(complete=True,protocol_identity=spec['identity'],manifest_identity=manifest['identity'],
            CPU_parent_CUDA_initialized=False,CPU_audit=check,readout_bridge_audit=bridge_check,actual_VLA_loaded=True,original96_native_capture=True,
            full_resource_qualification_requires_host_guard_terminal=True,causal_feedback_still_missing=True,native_training_cache_ready=False,
            training_steps=0,online_updates=0,TTA_utility_proven=False))
    finally:prepared['storage'].cleanup()

if __name__=='__main__':main()
