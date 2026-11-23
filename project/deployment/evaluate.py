"""Server-guarded baseline/full comparison; original complete-episode resumption."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import gpu_baseline as guard
import gpu_doctor
import user_container as container
from bundle_store import validated_store_path

CANONICAL_PROJECT=container.CANONICAL/'VLA-Adapter/cx_new_direction_guide_TTA'


def host(path):
    relative=Path(path).relative_to(CANONICAL_PROJECT)
    return container.PROJECT/relative


def add_env(cmd,values):
    split=cmd.index('--');options=[]
    for name,value in values.items():options+=['--setenv',name,str(value)]
    cmd[split:split]=options


def cleanup_runs(plan,token,reason,code):
    from finish_scratch import finish
    completed=[]
    for job in plan['jobs']:
        for session in (host(job['run'])/'sessions').glob('process_*'):
            if session.name in job['previous_sessions']:continue
            queue=session/'storage_cleanup.json'
            if not queue.exists() and (session/'storage.json').exists():
                storage=json.loads((session/'storage.json').read_text());temp=json.loads((session/'compiler_temp.json').read_text())
                if Path(temp['path']).parent!=Path(storage['scratch']):raise RuntimeError('Recovery scratch manifests disagree')
                with queue.open('x') as f:
                    json.dump(dict(scratch=storage['scratch'],phase='deferred_to_cpu_launcher',removed=False,gpu_uuid=plan['gpu_uuid'],launcher_token=token,record_origin='CPU recovery after confirmed GPU release'),f,indent=2)
            if queue.exists():
                record=json.loads(queue.read_text())
                if record.get('launcher_token')!=token:raise RuntimeError('Unexpected session cleanup token')
                completed.append(dict(session=str(session),cleanup=finish(queue)))
    print('EVAL_CPU_CLEANUP '+json.dumps(completed),flush=True)


def run(args):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    bundle=validated_store_path(args.cache_bundle) if args.cache_bundle else None
    native_seed = None
    if args.native_cache_proof:
        from native_cache_seed import NativeCacheSeed
        native_seed = NativeCacheSeed(args.native_cache_proof, bundle)
    scope='migration_acceptance' if args.acceptance else 'frozen_evaluation'
    if args.cache_mode == 'export':
        # Export has no imported fingerprint yet. Isolate its preparation run by
        # actual GPU UUID so CPU resumption cannot reuse another card's export.
        scope += '_cache_prepare_' + gpu_doctor.gpu_snapshot(args.gpu)['uuid']
    env=dict(CX_EVAL_COORDINATED='1',CX_EVAL_SCOPE=scope,CX_COMPILER_VARIANT='gemm_aten',
        TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS='ATEN',TORCHINDUCTOR_COMPILE_THREADS=16,
        CX_COMPILER_CACHE_MODE=args.cache_mode,CX_DEFER_SCRATCH_CLEANUP='1',CX_ROLLOUT_TIMING='1')
    if args.defer_observation_materialization:env['CX_OBSERVATION_MATERIALIZATION']='1'
    if args.defer_render_sync:env['CX_DEFER_RENDER_SYNC']='1'
    if args.defer_light_setters:env['CX_DEFER_LIGHT_SETTERS']='1'
    context_seed=None
    if args.context_cache_proof:
        from context_cache_seed import ContextCacheSeed
        context_seed=ContextCacheSeed(args.context_cache_proof,bundle)
    parity=None
    if args.query_parity_reference:
        from query_parity_guard import QueryParityGuard
        parity=QueryParityGuard(args.query_parity_reference)
    key=hashlib.sha256(json.dumps(dict(task=args.task,entry=args.entry,seed=args.seed,episodes=args.episodes,scope=scope,task_config=args.task_config),sort_keys=True).encode()).hexdigest()
    locks=container.PROJECT/'multi_event_gate_v1/outputs/deployment_eval_locks';locks.mkdir(parents=True,exist_ok=True)
    with (locks/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cmd=container.command(['bash','multi_event_gate_v1/scripts/cpu.sh',str(CANONICAL_PROJECT/'deployment/evaluation_plan.py'),
            '--task',args.task,'--entry',args.entry,'--seed',str(args.seed),'--episodes',str(args.episodes),
            '--max-attempts',str(args.max_attempts),'--task-config',args.task_config],mount_project=True)
        cmd[cmd.index('--'):cmd.index('--')]=['--tmpfs','/run']+(['--ro-bind',str(bundle),'/run/cxcache_bundle'] if bundle else [])
        add_env(cmd,env)
        result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError('CPU original preflight failed: '+result.stderr[-2500:])
        rows=[line[len('EVAL_PLAN '):] for line in result.stdout.splitlines() if line.startswith('EVAL_PLAN ')]
        if len(rows)!=1:raise RuntimeError('CPU planner did not return one audited plan: '+result.stderr[-1500:])
        plan=json.loads(rows[0]);print('EVAL_PLAN '+json.dumps(plan),flush=True)
        plan_file=container.HERE/'reports'/('target_comparison_plan_'+str(time.time_ns())+'.json')
        plan_file.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
        print('EVAL_PLAN_FILE',str(plan_file),flush=True)
        pending=[j for j in plan['jobs'] if not j['complete']]
        if not args.run or not pending:
            print('EVAL_CPU_ONLY '+json.dumps(dict(complete=not pending,gpu_started=False,pending=len(pending))),flush=True)
            return False
        info=gpu_doctor.gpu_snapshot(args.gpu)
        certificate=json.loads((container.WORK/('gpu-doctor-'+str(args.gpu)+'.json')).read_text())
        if certificate['uuid']!=info['uuid'] or certificate['build_mtime_ns']!=(container.WORK/'stage-bootstrap.ok').stat().st_mtime_ns:
            raise RuntimeError('GPU doctor identity/environment certificate changed')
        gpu_doctor.check_admission(info)
        if args.serial:pending=pending[:1]
        owner=pending[0];token=uuid.uuid4().hex
        env.update(CX_SHARED_MODEL_ROLE='owner' if len(pending)==2 else 'solo',CX_EVAL_EXPECTED_ID=owner['identity'],
            CX_TARGET_GPU_UUID=info['uuid'],CX_TARGET_DRIVER_VERSION=gpu_doctor.query(['-i',str(args.gpu),'--query-gpu=driver_version']).strip(),CX_CLEANUP_TOKEN=token,
            CX_PYTHON=str(CANONICAL_PROJECT/('deployment/light_state_python.sh' if args.defer_light_setters else ('deployment/render_sync_python.sh' if args.defer_render_sync else ('deployment/observation_python.sh' if args.defer_observation_materialization else 'deployment/startup_python.sh')))))
        if len(pending)==2:env.update(CX_EVAL_PEER_ARGUMENTS=json.dumps(pending[1]['arguments']),CX_EVAL_PEER_ID=pending[1]['identity'])
        if args.verify_interrupt:
            if len(pending)!=2 or args.entry!='both' or any(j['completed'] for j in pending):raise ValueError('Interruption verification requires a fresh two-condition acceptance dataset')
            env['CX_RESUME_TEST_HOLD']='M_event_full'
        cmd=container.command(['bash',str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/eval_event_full_worker.sh'),*owner['arguments']],mount_project=True)
        split=cmd.index('--');cmd[split:split]=gpu_doctor.driver_options(info)+(['--ro-bind',str(bundle),'/run/cxcache_bundle'] if bundle else [])
        add_env(cmd,env)
        report=container.HERE/'reports'/('target_comparison_'+str(time.time_ns())+'.jsonl')
        plan['gpu_uuid']=info['uuid']
        print('EVAL_GPU_RUN '+json.dumps(dict(owner=owner['condition'],workers=len(pending),telemetry=str(report))),flush=True)
        saw_pair=False
        seeded_sessions=set()
        def sample(row):
            nonlocal saw_pair
            if row['gpu_pids'] and args.admission_lock is not None:
                fcntl.flock(args.admission_lock,fcntl.LOCK_UN)
                args.admission_lock=None
            saw_pair=saw_pair or len(row['gpu_pids'])==2
            unfinished=0
            for job in pending:
                result_file=host(job['run'])/'result.json'
                done=result_file.exists() and json.loads(result_file.read_text()).get('complete') is True
                if not done:unfinished+=1
                sessions=[s for s in (host(job['run'])/'sessions').glob('process_*') if s.name not in job['previous_sessions']]
                if native_seed:
                    for session in sessions:
                        if session not in seeded_sessions and (session/'storage.json').exists():
                            if context_seed:
                                print('EVAL_CONTEXT_CACHE_SEEDED '+json.dumps(context_seed.seed(session)),flush=True)
                            result = native_seed.seed(session)
                            seeded_sessions.add(session)
                            print('EVAL_NATIVE_CACHE_SEEDED '+json.dumps(dict(session=str(session),**result)),flush=True)
                if parity:
                    error=parity.check(host(job['run']))
                    if error:return error
                if any((s/'failure.json').exists() for s in sessions):
                    return 'An evaluation worker failed; preserve committed episodes and stop the pair'
            if saw_pair and len(row['gpu_pids'])<2 and unfinished==2:
                return 'A GPU worker disappeared before its evaluation completed'
            if not args.verify_interrupt:return None
            baseline=next(j for j in plan['jobs'] if j['condition']=='Z_zero');full=next(j for j in plan['jobs'] if j['condition']=='M_event_full')
            done=host(baseline['run'])/'result.json'
            sessions=[s for s in (host(full['run'])/'sessions').glob('process_*') if s.name not in full['previous_sessions']]
            progress=sessions[0]/'progress.json' if len(sessions)==1 else Path('/nonexistent_progress')
            if done.exists() and progress.exists():
                a=json.loads(done.read_text());b=json.loads(progress.read_text())
                if a.get('complete') and b.get('phase')=='rollout' and b.get('query',0)>=2:
                    return 'TEST_INTERRUPT: baseline complete; M_event_full episode incomplete'
        guard.guarded_command(cmd,info,report,gpu=args.gpu,deadline=args.max_seconds,on_sample=sample,
            cleanup=lambda reason,code:cleanup_runs(plan,token,reason,code))
        print('EVAL_WORKERS_COMPLETED; rerun the same command for CPU-only final audits',flush=True)
        if args.cache_mode=='export':
            from bundle_store import publish
            for session in (host(owner['run'])/'sessions').glob('process_*'):
                exported=session/'deployment_compiler_bundle'
                if session.name not in owner['previous_sessions'] and exported.exists():
                    print('EVAL_PUBLISHED_CACHE',publish(exported),flush=True)
        return True


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--task',required=True);p.add_argument('--entry',choices=('baseline','full','both'),default='both')
    p.add_argument('--task-config',choices=('demo_randomized','demo_clean'),default='demo_randomized')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--max-attempts',type=int,default=1000);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--cache-mode',choices=('import','export'),default='import')
    p.add_argument('--cache-bundle',type=Path);p.add_argument('--run',action='store_true')
    p.add_argument('--context-cache-proof',type=Path,help='Checked same-card driver/NVRTC cache, private materialization only')
    p.add_argument('--defer-light-setters',action='store_true',help='Independent float32 light-state staging variant')
    p.add_argument('--defer-render-sync',action='store_true',help='Independent CPU-proven visual-sync variant; requires observation optimization and query parity guard')
    p.add_argument('--query-parity-reference',type=Path,help='Stop independent observation variant on any non-timing query/outcome difference')
    p.add_argument('--defer-observation-materialization',action='store_true',help='Independent performance identity; preserve every render, defer unused CPU readback')
    p.add_argument('--native-cache-proof',type=Path,help='Verified CPU-only native precompilation proof; preserves original worker identity')
    p.add_argument('--serial',action='store_true',help='Run pending conditions sequentially on the same GPU, preserving their original identities')
    p.add_argument('--acceptance',action='store_true');p.add_argument('--verify-interrupt',action='store_true')
    p.add_argument('--max-seconds',type=int,default=21600)
    args=p.parse_args()
    if args.context_cache_proof and not args.native_cache_proof:p.error('Context seeding requires native seeding at the verified storage boundary')
    if args.defer_light_setters and not args.defer_observation_materialization:p.error('Light-state staging requires observation optimization and query parity')
    if args.defer_render_sync and not args.defer_observation_materialization:p.error('--defer-render-sync requires --defer-observation-materialization')
    if args.defer_observation_materialization and (args.entry not in ('baseline','full') or not args.query_parity_reference):p.error('Observation variant requires one condition and a retained query parity reference')
    if (args.cache_mode=='import') != bool(args.cache_bundle):p.error('--cache-bundle is required exactly with cache import')
    if args.cache_mode=='export' and (not args.acceptance or args.entry!='baseline' or args.episodes!=1):p.error('Cache preparation requires --acceptance --entry baseline --episodes 1')
    if args.cache_mode=='export':args.max_seconds=min(args.max_seconds,270)
    if args.verify_interrupt and (not args.acceptance or not args.run):p.error('Fault injection requires --acceptance --run')
    if args.acceptance and args.episodes>3:p.error('Acceptance is bounded to three episodes per condition')
    if not 1<=args.max_seconds<=47*3600:p.error('Runtime must be at most 47 hours')
    if args.acceptance:args.max_seconds=min(args.max_seconds,480)
    # Serialize launch decisions for this account and reserve the selected GPU
    # until cleanup. Admission mutex is released once NVML sees our GPU worker.
    locks=container.WORK/'gpu_locks';locks.mkdir(exist_ok=True)
    with (locks/'account_admission.lock').open('a') as admission:
        fcntl.flock(admission,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (locks/('gpu_'+str(args.gpu)+'.lock')).open('a') as selected:
            fcntl.flock(selected,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for _ in range(3 if args.serial else 1):
                fcntl.flock(admission,fcntl.LOCK_EX|fcntl.LOCK_NB)
                args.admission_lock=admission
                if not run(args):break

if __name__=='__main__':main()
