"""Host cache selection, CPU audit, guarded per-task two-condition evaluation."""
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,uuid
from pathlib import Path
import user_container as container
import gpu_doctor,gpu_guard_bounded_metrics as guard
from gpu_policy import inventory,require_admission
from compiler_bundle import read_bundle
from evaluate import CANONICAL_PROJECT,host,add_env,cleanup_runs
from native_cache_seed import NativeCacheSeed
from shared6_native_cache_seed import Shared6NativeCacheSeed
from context_cache_seed import ContextCacheSeed
D=Path(__file__).resolve().parent;P=D/'reports'
from shared6_eval_resume import install as install_resume
install_resume()
# Host has no Torch; expose exactly the container runtime source inventory only.
import shared6_eval_backend as _candidate_backend
_candidate_old_sources=_candidate_backend.sources
def _candidate_sources():
    return dict(_candidate_old_sources(),**{name:hashlib.sha256((D/name).read_bytes()).hexdigest() for name in ('candidate_eval_runtime.py','candidate_eval_policy.py','candidate_eval_reference.py','candidate_eval_checkpoint.py')})
_candidate_backend.sources=_candidate_sources

def context_proof_rank(file):
    value=json.loads(file.read_text()).get('files',{})
    return (len(value)if isinstance(value,dict)else 0,file.stat().st_mtime_ns)

def select_cache(gpu_uuid,training_run):
    plan=json.loads((training_run/'training_plan.json').read_text());options=[]
    proofs=list(P.glob('*native*cpu*.json'))
    for file in(container.WORK/'compiler_bundles').glob('*/*/bundle.json'):
        meta=json.loads(file.read_text());f=meta['fingerprint'];base=f['policy_identity'].get('live_identity',f['policy_identity'])
        if f['gpu_uuid']!=gpu_uuid or f['gemm_backends']!='ATEN'or any(base[k]!=plan['policy_identity'][k]for k in('checkpoint','base_checkpoint_sha256','normalization_sha256','tokenizer_sha256')):continue
        native=None
        for proof in proofs:
            try:
                value=json.loads(proof.read_text())
                if value.get('passed')and value.get('artifact_sha256')==meta['artifact_sha256']:
                    cls=Shared6NativeCacheSeed if 'source_count'in value else NativeCacheSeed
                    native=cls(proof,file.parent);break
            except(ValueError,FileNotFoundError):continue
        context=None
        for proof in sorted(P.glob('target_context_cache*.json'),key=context_proof_rank):
            value=json.loads(proof.read_text())
            if value.get('artifact_sha256')==meta['artifact_sha256']and value.get('fingerprint_sha256')==meta['fingerprint_sha256']:context=ContextCacheSeed(proof,file.parent);break
        options.append((native is None,bool(native and native.proof.get('unselected')),f.get('model_cache_scope')!='trained_shared6_frozen_sampler',file.parent,native,context))
    if not options:return None,None,None
    _,_,_,bundle,native,context=sorted(options,key=lambda row:row[:3])[0];return bundle,native,context

def canonical(path):return str(CANONICAL_PROJECT/Path(path).resolve().relative_to(container.PROJECT))

def cpu_finalize(plan):
    program='''import json\nfrom pathlib import Path\nfrom task_adapters.eval_resume import final_result\nfrom cx_tta.paths import replace_json\nplan=json.loads(Path(PLAN_FILE).read_text());rows=[]\nfor job in plan['jobs']:\n result=final_result(Path(job['run']),job['manifest']);replace_json(Path(job['run'])/'result.json',result);replace_json(Path(job['run'])/'progress.json',dict(phase='finished',**result));rows.append(result)\nif rows[0]['valid_seeds']!=rows[1]['valid_seeds']:raise RuntimeError('Original zero/full eligible seed groups differ')\nprint('SHARED6_TASK_FULL100_CPU_AUDITED',json.dumps(dict(task=plan['task'],conditions=rows,matched_accepted_seeds=True)))\n'''
    file=P/('target_shared6_final_plan_'+str(time.time_ns())+'.json');file.write_text(json.dumps(plan)+'\n');program=program.replace('PLAN_FILE',repr(canonical(file)))
    completed=subprocess.run(container.command(['bash',str(CANONICAL_PROJECT/'deployment/task_adapter_python.sh'),'-c',program],mount_project=True),check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(completed.stdout,end='',flush=True)
    return json.loads([line.split(' ',1)[1]for line in completed.stdout.splitlines()if line.startswith('SHARED6_TASK_FULL100_CPU_AUDITED ')][-1])

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--gpu',type=int,choices=(0,1,2),required=True);ap.add_argument('--task',required=True);ap.add_argument('--training-run',type=Path,required=True);ap.add_argument('--run',action='store_true');ap.add_argument('--lanes',type=int,choices=(4,6),default=4);ap.add_argument('--cpu-start',type=int,default=0);ap.add_argument('--cold-prepare',action='store_true');ap.add_argument('--trace-signals',action='store_true');ap.add_argument('--status-file',type=Path);a=ap.parse_args()
    status_record=dict(pid=os.getpid(),task=a.task,gpu=a.gpu,phase='checking',jobs=[])
    def status(phase,**fields):
        status_record.update(phase=phase,**fields)
        if a.status_file:
            if not a.status_file.resolve().is_relative_to(P):raise RuntimeError('Status file outside reports')
            temporary=a.status_file.with_suffix('.tmp');temporary.write_text(json.dumps(status_record,indent=2)+'\n');temporary.replace(a.status_file)
    status('checking')
    available=sorted(os.sched_getaffinity(0));chosen=available[a.cpu_start:a.cpu_start+16]
    if len(chosen)!=16:raise RuntimeError('Need16 allowed logical cores for this card')
    os.sched_setaffinity(0,chosen);training=a.training_run.resolve();info=gpu_doctor.gpu_snapshot(a.gpu);bundle,native,context=select_cache(info['uuid'],training)
    common=['--task',a.task,'--training-run',canonical(training),'--gpu-uuid',info['uuid'],'--lanes',str(a.lanes)]
    if bundle:common+=['--bundle',canonical(bundle)]
    environment=dict(PYTHONFAULTHANDLER='1',JE_ARROW_MALLOC_CONF='background_thread:false',MALLOC_CONF='background_thread:false',ARROW_DEFAULT_MEMORY_POOL='system',PYTORCH_NVML_BASED_CUDA_CHECK='1',TORCH_WARM_POOL='0',TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS='ATEN',TORCHINDUCTOR_COMPILE_THREADS='16',CX_TARGET_GPU_UUID=info['uuid'],CX_TARGET_DRIVER_VERSION=gpu_doctor.query(['-i',str(a.gpu),'--query-gpu=driver_version']).strip())
    if not a.run:
        cmd=container.command(['bash',str(CANONICAL_PROJECT/'deployment/task_adapter_python.sh'),'-c',f"import sys;sys.path.insert(0,{str(CANONICAL_PROJECT/'deployment')!r});from candidate_eval_runtime import install;install();from shared6_eval_worker import main;main()",'--mode','plan',*common],mount_project=True);add_env(cmd,environment);subprocess.run(cmd,check=True);return
    proof=json.loads((P/'target_candidate_eval_prepared_cpu_20261025.json').read_text())
    from shared6_eval_backend import sources
    if not proof['passed']or proof['source_sha256']!=sources():raise RuntimeError('Full source-bound one-thread CPU/fork proof missing/stale')
    locks=container.WORK/'gpu_locks';locks.mkdir(exist_ok=True)
    with(locks/'account_admission.lock').open('a')as admission,(locks/('gpu_'+str(a.gpu)+'.lock')).open('a')as selected:
        fcntl.flock(admission,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(selected,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_doctor.check_admission(info)
        certificate=json.loads((container.WORK/('gpu-doctor-'+str(a.gpu)+'.json')).read_text())
        if certificate['uuid']!=info['uuid']or certificate['build_mtime_ns']!=(container.WORK/'stage-bootstrap.ok').stat().st_mtime_ns:raise RuntimeError('GPU CUDA/Vulkan certificate differs')
        barrier=training.parent/'evaluation_controls'/('shared6_control_'+uuid.uuid4().hex[:12]);barrier.mkdir(parents=True);cmd=container.command(['bash',str(CANONICAL_PROJECT/'deployment/task_adapter_python.sh'),str(CANONICAL_PROJECT/'deployment/candidate_eval_runtime.py'),'--barrier',canonical(barrier),*common],mount_project=True);split=cmd.index('--');cmd[split:split]=gpu_doctor.driver_options(info);add_env(cmd,environment)
        trace=None
        if a.trace_signals:
            trace=P/('target_shared6_signal_gpu'+str(a.gpu)+'_'+a.task+'_'+str(time.time_ns())+'.log')
            cmd=['strace','-f','--seccomp-bpf','-qq','-e','trace=exit,exit_group,kill,tgkill','-e','signal=all','-o',str(trace),'--',*cmd]
        process=subprocess.Popen(cmd,start_new_session=True);status('cpu_preparing',cpu_pid=process.pid,signal_trace=str(trace)if trace else None);began=time.monotonic();report=P/('target_shared6_eval_gpu'+str(a.gpu)+'_'+a.task+'_'+str(time.time_ns())+'.jsonl');plan=None;token=uuid.uuid4().hex
        try:
            while not(barrier/'ready.json').exists():
                if process.poll()is not None:raise RuntimeError('CPU prelude failed before GPU admission')
                fleet=inventory();card=next(c for c in fleet['cards']if c['uuid']==info['uuid'])
                if card['own_pids']:
                    table=guard.process_table();owned=guard.descendants(process.pid,table)
                    if set(card['own_pids'])&owned:raise RuntimeError('CPU prelude unexpectedly touched GPU')
                    raise RuntimeError('Selected GPU is no longer available; preserve existing jobs')
                if time.monotonic()-began>=240:raise TimeoutError('CPU preparation deadline')
                time.sleep(1)
            ready=json.loads((barrier/'ready.json').read_text());plan=ready['plan'];status('cpu_ready',jobs=[{k:j[k]for k in('condition','run','identity','completed')}for j in plan['jobs']],telemetry=str(report),barrier=str(barrier))
            if ready['os_threads']!=1 or ready['cuda_initialized']or ready['source_sha256']!=sources():raise RuntimeError('Unsafe or changed CPU prelude')
            if ready['gpu_needed']:
                session=host(ready['master_session'])
                if native:print('SHARED6_NATIVE_CACHE_SEEDED',json.dumps(native.seed(session)),flush=True)
                if context:print('SHARED6_CONTEXT_CACHE_SEEDED',json.dumps(context.seed(session)),flush=True)
            gpu_doctor.check_admission(gpu_doctor.gpu_snapshot(a.gpu))
        except BaseException:
            known=guard.stop_job(process);guard.record_release(process,info,began,report,known)
            if plan:cleanup_runs(plan,token,'CPU preparation failed',process.returncode)
            raise
        released=False
        def sample(row):
            nonlocal released
            if row['gpu_pids']and not released:fcntl.flock(admission,fcntl.LOCK_UN);released=True
            status('running',resources=row)
            if len(row['gpu_pids'])>1:return 'Expected oneGPU process for all independent scenes'
            for job in plan['jobs']:
                for session in(host(job['run'])/'sessions').glob('process_*'):
                    if session.name in job['previous_sessions']:continue
                    if(session/'failure.json').exists():return 'Original scene failed; preserve whole episodes and stop this card'
        def permit():
            gpu_doctor.check_admission(gpu_doctor.gpu_snapshot(a.gpu));(barrier/'permit.json').write_text(json.dumps(dict(gpu_uuid=info['uuid']))+'\n')
        print('SHARED6_EVAL_PLAN',json.dumps(plan),flush=True);print('EVAL_GPU_RUN',json.dumps(dict(task=a.task,gpu=a.gpu,telemetry=str(report),cold_prepare=a.cold_prepare)),flush=True)
        def cleanup(reason,code):
            try:
                from shared6_context_cache_export import export_context
                result=export_context(plan,bundle,report,context.files if context else None);print('SHARED6_CONTEXT_CACHE_PUBLISHED',json.dumps(result),flush=True)
            except Exception as error:print('SHARED6_CONTEXT_CACHE_EXPORT_FAILED',repr(error),flush=True)
            finally:cleanup_runs(plan,token,reason,code)
        failure=None
        try:guard.guarded_command(cmd,info,report,gpu=a.gpu,deadline=270 if a.cold_prepare else 21600,prepared_process=process,before_monitor=permit,on_sample=sample,max_memory_mib=24563,min_cumulative_utilization=33,steady_min_utilization=33,steady_window_seconds=300,cleanup=cleanup)
        except BaseException as error:
            failure=error;status('guard_stopped',error=str(error))
        finally:
            from bundle_store import publish
            for job in plan['jobs']:
                for session in(host(job['run'])/'sessions').glob('process_*'):
                    export=session/'deployment_compiler_bundle'
                    if session.name not in job['previous_sessions']and export.exists():print('SHARED6_COMPILED_ARTIFACT_PUBLISHED',str(publish(export)),flush=True)
        if failure is not None:
            if isinstance(failure,(KeyboardInterrupt,SystemExit)):raise failure
            complete=True
            for job in plan['jobs']:
                result=host(job['run'])/'result.json'
                complete=complete and result.exists()and json.loads(result.read_text()).get('complete')is True
            if not complete:raise failure
            rows=[json.loads(line)for line in report.read_text().splitlines()if line.strip()]
            if not any(row.get('own_gpu_released')is True for row in rows):raise failure
            print('SHARED6_COMPLETE_RESULTS_REQUIRE_CPU_RECOVERY_AFTER_GUARD_EXIT',flush=True)
        status('final_cpu_audit');result=cpu_finalize(plan)
        from candidate_eval_reference import load as _reference
        expected=_reference()['eligible_seeds']
        if any(row['valid_seeds']!=expected for row in result['conditions']):raise RuntimeError('Candidate accepted scenes differ from original baseline100')
        status('complete_cpu_audited',result=result,error=None,historical_guard_error=str(failure)if failure else None)
if __name__=='__main__':main()
