"""Actual one-thread CPU weights/runtime preload; fork proof; guarded GPU child."""
import argparse,importlib,json,os,sys,time,traceback,uuid
from pathlib import Path
from .paths import PROJECT,OUTPUT,read,write,output,verify,digest
from .contract import dataset,load_journal

def prepare(spec,root):
    from cpu_placement import apply,pages
    placement=read(OUTPUT/'cpu_placement_plan.json')
    assert placement['source_sha256']==digest(PROJECT/'multi_event_gate_v1/tta_event_v1/native_feedback_confirm_v5/cpu_placement.py')
    apply(placement)
    import torch
    torch.set_num_threads(1)
    from cx_tta_online.native_entry_storage import NativeEntryStorage
    from cx_tta.evaluate_b2 import resolve_tokenizer
    from .schedule import instrument
    from adapter_cache_cpu import load_cpu_policy,state_digest
    session=output(root/'sessions'/('process_'+uuid.uuid4().hex[:12]));session.mkdir(parents=True)
    class Storage(NativeEntryStorage):
        def prepare_runtime(self,folder):
            value=super().prepare_runtime(folder);p=value/'envs/_base_task.py'
            p.write_text(instrument(p.read_text()));return value
    storage=Storage(runtime_source=Path(spec['runtime_source']))
    try:
        runtime=storage.prepare_runtime(session);storage.configure_runtime(session,runtime,resolve_tokenizer())
        for module in ('cx_tta_feedback_confirm5.parallel_runtime','cx_tta_feedback_confirm5.current_case','cx_tta_feedback_confirm5.legacy_case',
             'native_nextchunk_v1.sampler_current','native_nextchunk_v1.child_exit',
             'cx_tta_feedback_confirm5.parent_callbacks','cx_tta_feedback_confirm5.actor_trace','parent_validation',
             'native_physics','collection_native_step','native_generic_physics'):
            importlib.import_module(module)
        proof=next(json.loads(line.split(' ',1)[1]) for line in (PROJECT/'deployment/reports/target_adapter_cache_cpu_arrow_nvml_20261024.log').read_text().splitlines() if line.startswith('ADAPTER_CACHE_CPU_PROOF '))
        assert proof['passed'] and proof['helper_sha256']==digest(PROJECT/'deployment/adapter_cache_cpu.py')
        base,info=load_cpu_policy(verify_state=True);converted=state_digest(base._model)
        from .query import build_head
        lane_heads=[];head_digests=[];before=torch.random.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            for lane in range(4):
                head,robots=build_head(spec['training'])
                assert all(v.device.type=='cpu'for v in head.state_dict().values())
                lane_heads.append((head,robots));head_digests.append(state_digest(head))
        assert torch.equal(before,torch.random.get_rng_state())and len(set(head_digests))==1
        from query_object_v1.stream import video
        from native_delayed_query_v1.primitives import configuration
        from native_delayed_query_v1.predictor import FrozenPredictor
        with torch.random.fork_rng(devices=[]):mask_model=video.build('cpu')
        mask_digest=state_digest(mask_model)
        teacher=read(PROJECT/'multi_event_gate_v1/tta_event_v1/outputs/native_arrived_mask_v3/cpu_models.json')
        assert mask_digest==teacher['model_state_sha256']['SAM2']
        assert torch.equal(before,torch.random.get_rng_state())
        cfg,camera,_=configuration()
        seeds={}
        import numpy as np
        for item in spec['scenes']:
            seed=item['earliest_seed']
            if seed['available']:
                assert digest(seed['mask_file'])==seed['mask_sha256']
                with np.load(seed['mask_file'],allow_pickle=False)as z:seeds[item['case_id']]=z['masks'].copy()
            else:seeds[item['case_id']]=None
        predictor=FrozenPredictor()
        placement_pages=pages()
        assert converted==proof['converted_state_sha256']
        assert not torch.cuda.is_initialized() and len(list(Path('/proc/self/task').iterdir()))==1
        write(session/'cpu_preload.json',dict(passed=True,cuda_initialized=False,threads=1,converted_state_sha256=converted,CPU_lane_heads=4,head_state_sha256=head_digests,CPU_affinity=sorted(os.sched_getaffinity(0)),anonymous_pages_by_NUMA=placement_pages,CPU_head_preparation_preserves_Torch_RNG=True,SAM2_state_sha256=mask_digest,SAM2_parameter_count=sum(p.numel()for p in mask_model.parameters()),**info))
        return dict(storage=storage,runtime=runtime,session=session,base=base,CPU_lane_heads=lane_heads,CPU_placement=placement,CPU_pages=placement_pages,mask_model=mask_model,feedback_hardware=(cfg,camera),seed_masks=seeds,predictor=predictor)
    except BaseException:storage.cleanup();raise

def state_digest_for_ready(model):
    from adapter_cache_cpu import state_digest
    return state_digest(model)

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);p.add_argument('--barrier',type=Path)
    p.add_argument('--nonce');p.add_argument('--cpu-proof',action='store_true');a=p.parse_args()
    selected=os.environ.get('CUDA_VISIBLE_DEVICES','');os.environ['CUDA_VISIBLE_DEVICES']=''
    from .prepare import prepare as freeze
    spec=freeze(cpu_fixture=a.cpu_proof);verify(spec);root,manifest=dataset(spec,a.gpu_uuid);j=load_journal(root,manifest,spec)
    import torch
    prepared=prepare(spec,root)
    try:
        child=os.fork()
        if child==0:
            assert not torch.cuda.is_initialized() and not torch.cuda._is_in_bad_fork() and int(torch.arange(10).sum())==45
            os._exit(0)
        _,status=os.waitpid(child,0);assert os.waitstatus_to_exitcode(status)==0
        ready=dict(CPU_admission_deadline_monotonic=time.monotonic()+spec['CPU_admission_wait_seconds'],protocol_identity=spec['identity'],source_sha256=spec['source_sha256'],
            os_threads=len(list(Path('/proc/self/task').iterdir())),cuda_initialized=torch.cuda.is_initialized(),cpu_fork_child_passed=True,CPU_affinity=sorted(os.sched_getaffinity(0)),CPU_lane_heads=len(prepared['CPU_lane_heads']),anonymous_pages_by_NUMA=prepared['CPU_pages'],SAM2_state_sha256=state_digest_for_ready(prepared['mask_model']))
        if a.cpu_proof:
            write(OUTPUT/'cpu_preload_proof.json',dict(passed=True,**ready));print('CURRENT_STREAM_ACTUAL_CPU_PRELOAD_FORK_PASS',flush=True);return
        if a.barrier is None or not a.nonce:raise ValueError('Fresh guarded host barrier required')
        write(a.barrier/'ready.json',ready);began=time.monotonic()
        while not (a.barrier/'permit.json').exists():
            if time.monotonic()>ready['CPU_admission_deadline_monotonic']:raise TimeoutError('CPU-only prepared GPU admission deadline')
            time.sleep(.05)
        permit=read(a.barrier/'permit.json')
        assert permit['gpu_uuid']==a.gpu_uuid and permit['attempt_nonce']==a.nonce and permit['protocol_identity']==spec['identity']
        initial=len(j['cases']);ipc=prepared['session']/'cpu_validation'
        prepared.update(barrier=a.barrier,source_bound_GPU_permit=True,ipc=ipc,nonce=a.nonce)
        from parent_validation import ParentValidator,wait_for_parent,atomic
        from .parent_callbacks import callbacks
        validate,accepted=callbacks(spec,root,manifest,j,prepared['session'])
        def CPU_only():
            if torch.cuda.is_initialized():raise ValueError('CPU validator initialized CUDA')
        service=ParentValidator(root,ipc,[v['case_id']for v in spec['scenes'][initial:]],
            nonce=a.nonce,protocol_identity=spec['identity'],bridge_count=max(0,spec['bridge_count']-initial),
            validate=validate,commit=accepted,assert_cpu=CPU_only)
        child=os.fork()
        if child==0:
            try:
                os.environ['CUDA_VISIBLE_DEVICES']=selected
                assert not torch.cuda.is_initialized() and not torch.cuda._is_in_bad_fork()
                wait_for_parent(ipc,nonce=a.nonce,protocol_identity=spec['identity'])
                from .parallel_runtime import execute
                execute(spec,root,manifest,j,prepared=prepared)
            except BaseException:traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
            os._exit(0)
        # This is the first validator thread; it starts only AFTER the CUDA child's fork.
        try:
            service.start_after_fork(child)
            atomic(ipc/'parent_validator_ready.json',dict(nonce=a.nonce,protocol_identity=spec['identity'],child_pid=child))
        except BaseException:
            # This is our own unreaped fork child; it cannot yet have a reused PID.
            try:os.kill(child,__import__('signal').SIGTERM)
            except ProcessLookupError:pass
            _,status=os.waitpid(child,0);service.child_exited(os.waitstatus_to_exitcode(status))
            if service.thread is not None:
                try:service.join()
                except BaseException:pass
            raise
        from native_nextchunk_v1.child_exit import wait_and_record
        try:
            code=wait_and_record(child,a.barrier/'normal_child_exit.json',attempt_nonce=a.nonce,gpu_uuid=a.gpu_uuid,protocol_identity=spec['identity'],cpu_only_fixture=False)
        except BaseException:
            service.cancel_child()
            try:
                _,status=os.waitpid(child,0);code=os.waitstatus_to_exitcode(status)
            except ChildProcessError:code=-1
            service.child_exited(code)
            try:service.join()
            except BaseException:pass
            raise
        service.child_exited(code)
        validation=service.join()
        CPU_only()
        if code!=0:raise SystemExit(code)
        assert validation['all_cases_validated']and len(j['cases'])==len(spec['scenes'])
        verify(spec)
        write(prepared['session']/'source_storage_audit.json',prepared['storage'].verify_source())
        write(root/'result.json',dict(complete=True,identity=manifest['identity'],protocol_identity=spec['identity'],
           bundles=len(j['cases']),journal_sha256=digest(root/'journal.json'),gpu_uuid=a.gpu_uuid,
           CPU_parent_validation=True,parent_CUDA_initialized=False,persistent_lanes=4,
           old_resource_failure_preserved=True,online_updates=0,full_task_episodes=0,TTA_utility_proven=False))
        raise SystemExit(0)
    finally:prepared['storage'].cleanup()
if __name__=='__main__':main()
