"""Ramp independent expert collectors after real complete-scene warmup.

No unmeasured generic training jobs here. Initial reservations are conservative
engineering limits, updated from sampled usage; they are not guaranteed peaks.
Whole-card free VRAM includes Vulkan/graphics and foreign jobs. Unattributed
usage remains charged via free memory rather than being assumed to be zero.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

from .audit import digest
from .paths import WORKSPACE_ROOT, new_run, write_json, replace_json, output_path
from .gpu_pool import gpu_snapshot, host_resources, group_usage, cleanup_scratch
from .collection_jobs import read, build_jobs, validate_complete, collection_root
from .gpu_monitor import GpuMonitor


def options(argv=None):
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--tasks',default='bowls,blocks',help='bowls,blocks or full task names; comma-separated')
    ap.add_argument('--splits',default='train,val',help='train,val or adapter_fit,adapter_validation')
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--max-workers',type=int,default=4)
    ap.add_argument('--warmup-scenes',type=int,default=2)
    ap.add_argument('--min-samples',type=int,default=3)
    ap.add_argument('--job-gb',type=float,default=12,help='Initial reservation floor per expert process, not measured peak')
    ap.add_argument('--job-ram-gb',type=float,default=16)
    ap.add_argument('--job-disk-gb',type=float,default=2)
    ap.add_argument('--reserve-gb',type=float,default=6)
    ap.add_argument('--ram-reserve-gb',type=float,default=8)
    ap.add_argument('--startup-extra-gb',type=float,default=1)
    ap.add_argument('--peak-headroom',type=float,default=.3)
    ap.add_argument('--cpus-per-job',type=int,default=4)
    ap.add_argument('--min-disk-gb',type=int,default=10)
    ap.add_argument('--poll-seconds',type=float,default=2)
    ap.add_argument('--admission-timeout',type=float,default=600)
    ap.add_argument('--retry-incomplete',action='store_true',help='Compatibility alias; whole-episode resume is now automatic, never starts over')
    ap.add_argument('--dry-run',action='store_true',help='CPU task/data audit only, no GPU query or jobs')
    a = ap.parse_args(argv)
    if a.gpu<0 or not 1<=a.max_workers<=16 or min(a.cpus_per_job,a.warmup_scenes,a.min_samples,a.min_disk_gb)<1:
        ap.error('Invalid GPU/worker/warmup/resource count')
    for k in ('job_gb','job_ram_gb','job_disk_gb','reserve_gb','ram_reserve_gb',
              'startup_extra_gb','poll_seconds','admission_timeout'):
        if not math.isfinite(getattr(a,k)) or getattr(a,k)<=0:
            ap.error('Positive finite '+k+' required')
    if not math.isfinite(a.peak_headroom) or not 0<=a.peak_headroom<=2:
        ap.error('Invalid peak headroom')
    return a


def reservations(args, task, profiles):
    p = profiles.get(task,{})
    return dict(vram=max(args.job_gb*1024,p.get('vram',0)*(1+args.peak_headroom)),
                ram=max(args.job_ram_gb*1024,p.get('ram',0)*(1+args.peak_headroom)))


def admission(args, running, gpu, host, proposed, disk_free_gb):
    if len(running)>=args.max_workers:return False,'worker_limit'
    if (len(running)+1)*args.cpus_per_job>host['cpus']:return False,'cpu_limit'
    if any(not j['ready'] for j in running):return False,'waiting_validated_warmup_scenes'
    # Per-process attribution can be incomplete (graphics or PID namespaces).
    # Keep those bytes in whole-card usage AND keep unmatched reservations.
    unfilled = sum(max(0,j['reserve_vram_mib']-j['used_mib']) for j in running)
    if gpu['free_mib'] < unfilled+proposed['vram']+(args.reserve_gb+args.startup_extra_gb)*1024:
        return False,'whole_card_vram_and_reservations'
    unfilled_ram=sum(max(0,j['reserve_ram_mib']-j['rss_mib']) for j in running)
    if host['available_ram_mib']<unfilled_ram+proposed['ram']+args.ram_reserve_gb*1024:
        return False,'ram_reservation'
    needed_disk=args.min_disk_gb+args.job_disk_gb+sum(j['remaining_disk_gb'] for j in running)
    if disk_free_gb<needed_disk:return False,'disk_headroom'
    return True,'admitted'


def read_progress(status, spec):
    if not status.get('run'):
        return dict(accepted=0,attempts=0)
    run=output_path(Path(status['run']))
    if run.parent!=collection_root(spec).resolve():
        if status.get('phase')=='failed':return dict(accepted=0,attempts=0)
        raise ValueError('Heartbeat points outside expected collection')
    result=read(run/'result.json')
    progress=read(run/'progress.json')
    n=progress.get('accepted',result.get('accepted_episodes',0))
    if type(n)is not int or not 0<=n<=spec['episodes']:
        raise ValueError('Invalid collection progress')
    return dict(accepted=n,attempts=progress.get('attempts',result.get('attempts',0)))


def group_alive(pid):
    try:os.killpg(pid,0);return True
    except ProcessLookupError:return False


def validate_worker(job, status, rc):
    if rc!=0 or status.get('phase')!='finished' or not status.get('result'):
        raise ValueError('Worker failed or missing finished collection result')
    run=output_path(Path(status['run']))
    if output_path(Path(status['result']))!=run/'result.json':
        raise ValueError('Wrong collection result path')
    return validate_complete(run,job['spec'])


def launch(args, spec, pool_run, resources):
    folder=pool_run/spec['id'];folder.mkdir()
    write_json(folder/'job.json',spec)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),CX_GPU_JOB_STATUS=str(folder/'worker_status.json'),
               CX_GPU_JOB_ID=spec['id'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='1')
    scratch=Path(tempfile.mkdtemp(prefix='cxgp_',dir='/tmp'))
    for key in ('XDG_CACHE_HOME','HF_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TMPDIR'):
        path=scratch/key.lower();path.mkdir();env[key]=str(path)
    env['TMP']=env['TEMP']=env['TMPDIR']
    write_json(folder/'scratch.json',dict(path=str(scratch)))
    argv=['bash',str(WORKSPACE_ROOT/'scripts/collection_worker.sh'),'--job-spec',str(folder/'job.json'),'--gpu',str(args.gpu)]
    log=(folder/'console.log').open('w',buffering=1)
    try:
        proc=subprocess.Popen(argv,cwd=WORKSPACE_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException:
        log.close();shutil.rmtree(scratch);raise
    return dict(spec=spec,process=proc,log=log,pid=proc.pid,job_dir=str(folder),scratch=str(scratch),
                state='running',started_epoch=time.time(),started_mono=time.monotonic(),
                ready=False,samples=0,accepted=0,initial_accepted=0,attempts=0,used_mib=0,rss_mib=0,
                peak_mib=0,peak_rss_mib=0,reserve_vram_mib=resources['vram'],reserve_ram_mib=resources['ram'],
                remaining_disk_gb=args.job_disk_gb,scratch_removed=False)


def update_usage(j,args,snap,profiles):
    used,rss=group_usage(j['pid'],snap['process_mib'])
    j.update(used_mib=used,rss_mib=rss,peak_mib=max(j['peak_mib'],used),
             peak_rss_mib=max(j['peak_rss_mib'],rss),samples=j['samples']+1)
    j['reserve_vram_mib']=max(j['reserve_vram_mib'],j['peak_mib']*(1+args.peak_headroom))
    j['reserve_ram_mib']=max(j['reserve_ram_mib'],j['peak_rss_mib']*(1+args.peak_headroom))
    status=read(Path(j['job_dir'])/'worker_status.json')
    j['worker_status']=status
    j['initial_accepted']=status.get('initial_accepted',j.get('initial_accepted',0))
    progress=read_progress(status,j['spec'])
    if progress['accepted']<j['accepted']:
        raise ValueError('Accepted count went backwards')
    changed=progress['accepted']>j['accepted']
    j.update(progress)
    session_accepted=status.get('session_accepted',j['accepted']-j['initial_accepted'])
    j['ready']=(status.get('phase')=='collecting' and session_accepted>=args.warmup_scenes
                and j['samples']>=args.min_samples)
    if changed:
        print(f"\033[32m{j['spec']['id']}: {j['accepted']}/{j['spec']['episodes']} accepted "
              f"(kept {j['initial_accepted']}, new {j['accepted']-j['initial_accepted']}), "
              f"{j['attempts']} attempts\033[0m",flush=True)
        run=Path(status['run'])
        rows=[r for r in read(run/'attempts.json') if r.get('accepted')]
        if rows:
            biggest=max(output_path(Path(r['hdf5'])).stat().st_size for r in rows)/2**30
            j['remaining_disk_gb']=max(args.job_disk_gb,
                biggest*(j['spec']['episodes']-j['accepted'])*(1+args.peak_headroom)+1)
    if j['ready']:
        p=profiles.setdefault(j['spec']['task'],dict(vram=0,ram=0))
        p.update(vram=max(p['vram'],j['peak_mib']),ram=max(p['ram'],j['peak_rss_mib']))
    return status


def execute(args,specs,evidence):
    reused=[s['existing']['reuse'] for s in specs if s['existing']['reuse']]
    pending=[s for s in specs if not s['existing']['reuse']]
    run=new_run('pool',group=('collection_pools',));print('COLLECTION_POOL',run,flush=True)
    started=time.monotonic();jobs=[];profiles={};halt=False;reason='';error=None
    peak_workers=0;peak_gpu=0;last_print=0;idle_since=None;lock=None;previous={}
    monitor=GpuMonitor(lambda:gpu_snapshot(args.gpu))
    def serializable(j):return {k:v for k,v in j.items() if k not in ('process','log')}
    try:
        initial=monitor.initial() if pending else None
        if initial:
            lock=(Path('/tmp')/f"cx_gpu_pool_{os.getuid()}_{initial['uuid']}.lock").open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise RuntimeError('Another scheduler already manages this GPU; use a free card') from exc
        files=[Path(__file__),Path(__file__).with_name('gpu_monitor.py'),Path(__file__).with_name('collection_jobs.py'),Path(__file__).with_name('collection_worker.py'),
               Path(__file__).with_name('collection_resume.py'),Path(__file__).with_name('collect_stacking_resumable.py'),
               Path(__file__).with_name('gpu_pool.py'),WORKSPACE_ROOT/'scripts/collect_stacking_parallel.sh',
               WORKSPACE_ROOT/'scripts/collection_worker.sh']
        write_json(run/'manifest.json',dict(schema=1,options=vars(args),jobs=specs,reused=reused,evidence=evidence,
            gpu=initial,source_sha256={str(f):digest(f) for f in files},
            policy='Warm on real formal scenes; preserve episode targets, seeds, teacher, RNG and data split',
            measurement='Whole-card free VRAM includes graphics; attributed group peaks may be incomplete; untracked usage conservatively remains charged'))
        def cancel(*_):raise KeyboardInterrupt('Collection pool cancelled')
        previous={sig:signal.signal(sig,cancel) for sig in (signal.SIGINT,signal.SIGTERM)}
        with (run/'resources.jsonl').open('w',buffering=1) as telemetry:
            while pending or any(j['state']=='running' for j in jobs):
                snap=monitor.observe();host=host_resources();now=time.monotonic()
                for j in jobs:
                    if j['state']!='running':continue
                    status=update_usage(j,args,snap,profiles)
                    if status.get('phase')=='failed':halt=True;reason='worker_failed_no_automatic_retry'
                    rc=j['process'].poll()
                    if rc is None:continue
                    if group_alive(j['pid']):
                        j['ready']=False
                        j.setdefault('exit_wait_started',now)
                        if now-j['exit_wait_started']>30:raise RuntimeError('Worker descendants did not exit: '+j['spec']['id'])
                        continue
                    try:
                        # A child may publish its final result between the usage
                        # sample above and poll(). Read again after it has exited.
                        status=update_usage(j,args,snap,profiles)
                        result=validate_worker(j,status,rc)
                        j.update(state='complete',result=result)
                    except (ValueError,KeyError,OSError) as exc:
                        j.update(state='failed',validation_error=str(exc));halt=True;reason='worker_failed_no_automatic_retry'
                    j.update(returncode=rc,elapsed_s=now-j['started_mono'])
                    j['log'].close();cleanup_scratch(j)
                    print(j['spec']['id'],j['state'],flush=True)
                running=[j for j in jobs if j['state']=='running']
                why=reason or 'all_jobs_launched'
                if pending and not halt:
                    proposed=reservations(args,pending[0]['task'],profiles)
                    disk=min(shutil.disk_usage(p).free/2**30 for p in (WORKSPACE_ROOT,'/tmp'))
                    ok,why=admission(args,running,snap,host,proposed,disk)
                    if not snap['telemetry_fresh']:ok=False;why='telemetry_unavailable_existing_workers_continue'
                    if ok:
                        spec=pending[0]
                        job=launch(args,spec,run,proposed)
                        pending.pop(0);jobs.append(job)
                        print(f"\033[36mSTART {spec['id']}, reservation={proposed['vram']/1024:.1f} GiB; "
                              f"log={run/spec['id']/'console.log'}\033[0m",flush=True)
                        idle_since=None
                    elif not running:
                        if idle_since is None:idle_since=now
                        if now-idle_since>=args.admission_timeout:halt=True;reason='admission_timeout_'+why
                active=[j for j in jobs if j['state']=='running']
                peak_workers=max(peak_workers,len(active));peak_gpu=max(peak_gpu,snap['used_mib'])
                record=dict(epoch=time.time(),elapsed_s=now-started,active=len(active),pending=len(pending),
                    gpu={k:v for k,v in snap.items() if k!='process_mib'},host=host,admission=why,
                    workers=[{k:j[k] for k in ('pid','accepted','ready','used_mib','peak_mib','reserve_vram_mib','rss_mib')} for j in active])
                telemetry.write(json.dumps(record)+'\n')
                replace_json(run/'status.json',dict(phase='draining_after_failure' if halt else 'running',reason=reason,
                    jobs=[serializable(j) for j in jobs],pending=[s['id'] for s in pending],reused=reused,resource=record))
                if now-last_print>=30:
                    print(f"POOL active={len(active)}/{args.max_workers} pending={len(pending)} "
                          f"GPU={snap['used_mib']/1024:.1f}/{snap['total_mib']/1024:.1f} GiB "
                          f"util={snap['utilization']:.0f}% waiting={why}",flush=True);last_print=now
                if halt and not active:break
                if pending or active:time.sleep(args.poll_seconds)
    except BaseException as exc:
        error=repr(exc);halt=True;reason='scheduler_failed_or_interrupted'
        print('COLLECTION_POOL_ERROR',error,flush=True)
    finally:
        # Cancel only groups this invocation created, including orphan descendants.
        active=[j for j in jobs if group_alive(j['pid'])]
        for j in active:
            try:os.killpg(j['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
        deadline=time.monotonic()+20
        for j in active:
            try:j['process'].wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:pass
            if group_alive(j['pid']):
                try:os.killpg(j['pid'],signal.SIGKILL)
                except ProcessLookupError:pass
            j['process'].wait()
            j.update(state='interrupted',returncode=j['process'].returncode)
        for j in jobs:
            if not j['log'].closed:j['log'].close()
            if not j.get('scratch_removed'):
                try:cleanup_scratch(j)
                except OSError as exc:j['scratch_cleanup_error']=str(exc)
        for sig,handler in previous.items():signal.signal(sig,handler)
        elapsed=time.monotonic()-started
        accepted=sum(j['accepted']-j['initial_accepted'] for j in jobs if not j.get('worker_status',{}).get('reused'))
        complete=not halt and not pending and all(j['state']=='complete' for j in jobs)
        result=dict(complete=complete,reason=reason,error=error,elapsed_s=elapsed,
            new_accepted_episodes=accepted,new_accepted_per_hour=accepted/elapsed*3600 if elapsed else None,
            max_concurrent_workers=peak_workers,peak_whole_card_vram_mib=peak_gpu,
            telemetry_failures=monitor.total_failures,
            sampled_profiles=profiles,reused=reused,jobs=[serializable(j) for j in jobs],
            not_started=[s['id'] for s in pending],training_ready=False,
            limitation='Observed peaks are samples; no GPU speedup or OOM-free guarantee without an actual run. Resume only audited whole episodes within one dataset; no partial-scene splicing.')
        write_json(run/'result.json',result)
        replace_json(run/'status.json',dict(phase='complete' if complete else 'incomplete',**result))
        if lock is not None:lock.close()
        print('COLLECTION_POOL_COMPLETE' if complete else 'COLLECTION_POOL_INCOMPLETE',run,flush=True)
    return 0 if result['complete'] else 1


def main(argv=None):
    a=options(argv);specs,evidence=build_jobs(a)
    if a.dry_run:
        print(json.dumps(dict(cpu_only=True,gpu_started=False,jobs=specs,
            total_requested=sum(s['episodes'] for s in specs),options=vars(a)),ensure_ascii=False,indent=2))
        return 0
    return execute(a,specs,evidence)


if __name__=='__main__':
    raise SystemExit(main())
