"""Resource-aware single-GPU process pool. No model batching or RNG changes.

Built-in evaluation workers must finish one whole episode before another worker
starts. Generic training/collection workers require a declared measured peak.
Memory admission is a reservation estimate, not an allocator or an OOM guarantee.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time

from .paths import WORKSPACE_ROOT, new_run, output_path, replace_json, write_json
from .official_eval_console import episode_output
from .gpu_monitor import GpuMonitor


def read_json(path):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return {}


def cleanup_scratch(job):
    """Only our short private cache, only after the entire worker group exits."""
    try:
        os.killpg(job['pid'],0)
        return False
    except ProcessLookupError: pass
    path=Path(job['scratch'])
    if path.is_symlink() or path.parent!=Path('/tmp') or not path.name.startswith('cxgp_'):
        raise ValueError('Invalid scheduler-owned scratch path')
    if path.exists():shutil.rmtree(path,ignore_errors=False)
    job['scratch_removed']=True
    return True


def gpu_snapshot(gpu):
    def query(fields, category='gpu'):
        out = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-'+category+'='+fields,
            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
        return list(csv.reader(out.stdout.strip().splitlines(), skipinitialspace=True))
    rows = query('uuid,memory.total,memory.used,memory.free,utilization.gpu')
    if len(rows) != 1: raise ValueError('Exactly one physical GPU must be selected')
    uuid,total,used,free,util = rows[0]
    apps = {}
    for row in query('pid,used_gpu_memory', 'compute-apps'):
        if len(row) != 2: continue
        try: pid,mb = int(row[0]),float(row[1])
        except ValueError: continue
        # Driver may list compute+graphics contexts twice for the same PID.
        apps[pid] = max(apps.get(pid, 0), mb)
    return dict(uuid=uuid, total_mib=float(total), used_mib=float(used),
                free_mib=float(free), utilization=float(util), process_mib=apps)


def host_resources():
    cpus = len(os.sched_getaffinity(0))
    try:
        quota,period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max': cpus = min(cpus, max(1, int(int(quota)/int(period))))
    except (OSError,ValueError): pass
    info = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    free = int(info['MemAvailable'].split()[0])/1024
    try:
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit != 'max':
            current = int(Path('/sys/fs/cgroup/memory.current').read_text())
            # Inactive file cache is reclaimable; counting it as fixed model RSS
            # would unnecessarily serialize workers after reading model files.
            stats = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
            reclaimable = int(stats.get('inactive_file',0))
            free = min(free, max(0,(int(limit)-current+reclaimable)/1024**2))
    except (OSError,ValueError): pass
    return dict(cpus=cpus, available_ram_mib=free)


def group_usage(pgid, apps):
    vram = 0; rss = 0
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit(): continue
        try:
            pid = int(folder.name)
            if os.getpgid(pid) != pgid: continue
            vram += apps.get(pid,0)
            for line in (folder/'status').read_text().splitlines():
                if line.startswith('VmRSS:'): rss += int(line.split()[1])/1024
        except (ProcessLookupError,PermissionError,FileNotFoundError): continue
    return vram,rss


def reservation(job, headroom):
    return max(job['spec']['peak_vram_mib'], math.ceil(job.get('peak_mib',0)*(1+headroom)))


def admission(spec, running, gpu, host, *, headroom, reserve_mib, startup_mib,
              ram_reserve_mib, cpus_per_job, max_workers):
    if len(running) >= max_workers: return False,'worker_limit'
    if (len(running)+1)*cpus_per_job > host['cpus']: return False,'cpu_limit'
    if any(not j['ready'] for j in running): return False,'waiting_first_episode_or_profile_stagger'
    pending = sum(max(0,reservation(j,headroom)-j.get('used_mib',0)) for j in running)
    if gpu['free_mib'] < pending+spec['peak_vram_mib']+reserve_mib+startup_mib:
        return False,'vram_reservation'
    pending_ram = sum(max(0,j['spec']['ram_mib']-j.get('rss_mib',0)) for j in running)
    if host['available_ram_mib'] < pending_ram+spec['ram_mib']+ram_reserve_mib:
        return False,'ram_reservation'
    return True,'admitted'


def normalize_jobs(document, *, allow_unprofiled=False):
    if document.get('schema') != 1 or not isinstance(document.get('jobs'),list) or not document['jobs']:
        raise ValueError('Manifest needs schema=1 and a nonempty jobs list')
    jobs=[];names=set()
    for value in document['jobs']:
        j=dict(value);name=j.get('id','')
        if not re.fullmatch(r'[a-zA-Z0-9_]+',name) or name in names:
            raise ValueError('Unique alphanumeric job ids required')
        names.add(name)
        if not isinstance(j.get('argv'),list) or not j['argv'] or not all(isinstance(x,str) for x in j['argv']):
            raise ValueError('argv must be an explicit string array; shell expressions are not evaluated')
        if j.get('kind') not in ('evaluation','collection','training'):
            raise ValueError('kind must be evaluation, collection or training')
        if j.get('readiness') not in ('stacking_status','profiled'):
            raise ValueError('readiness must be stacking_status or profiled')
        for k in ('peak_vram_mib','ram_mib'):
            if not isinstance(j.get(k),(float,int)) or not math.isfinite(j[k]) or j[k]<=0:
                raise ValueError('Positive finite resource reservation required: '+k)
        if j['readiness']=='profiled' and j.get('peak_was_measured') is not True and not allow_unprofiled:
            raise ValueError('Training/collection requires a previously measured peak, not an eval estimate')
        if j['readiness']=='stacking_status' and (j['kind']!='evaluation' or type(j.get('episodes')) is not int or j['episodes']<1):
            raise ValueError('Stacking evaluation needs a positive completed-episode target')
        cwd=Path(j.get('cwd',WORKSPACE_ROOT)).resolve()
        if not cwd.is_relative_to(WORKSPACE_ROOT) or not cwd.is_dir():
            raise ValueError('Worker cwd must be an existing directory in the isolated workspace')
        j['cwd']=str(cwd);jobs.append(j)
    return jobs


def completed_job(spec, status, returncode):
    if returncode != 0: return False,'nonzero_exit',None
    if spec['readiness']!='stacking_status':
        return True,'process_exit_zero_experiment_not_independently_validated',None
    if status.get('phase')!='finished' or not status.get('result'):
        return False,'missing_finished_result',None
    result=read_json(output_path(Path(status['result'])))
    valid=(result.get('complete') is True and result.get('completed')==spec['episodes']
           and result.get('requested')==spec['episodes'] and result.get('censored')==0
           and result.get('task')==spec['task'] and result.get('condition')==spec['condition'])
    return valid,'validated_complete_eval' if valid else 'incomplete_or_wrong_eval',result


def build_eval_jobs(args):
    new_adapter = bool(getattr(args,'training_run',None))
    event_cache_limit = getattr(args,'event_cache_limit',None)
    if event_cache_limit is None:
        event_cache_limit = 0 if new_adapter else 6
    if new_adapter:
        from .stacking_eval_resume import options, preflight
    else:
        from .evaluate_stacking import options, preflight
    jobs=[];preflights={}
    tasks=args.tasks.split(',');conditions=args.conditions.split(',')
    if len(set(tasks))!=len(tasks) or len(set(conditions))!=len(conditions):
        raise ValueError('Duplicate tasks/conditions are not independent repeats')
    for task in tasks:
        for cond in conditions:
            argv=['--task',task,'--condition',cond,'--gpu',str(args.gpu),
                  '--episodes',str(args.episodes),'--task-config','demo_randomized','--instruction-type','unseen',
                  '--event-cache-limit',str(event_cache_limit)]
            argv += (['--training-run',str(args.training_run),'--seed-start',str(args.seed_start)] if new_adapter
                     else ['--seed',str(args.seed)])
            opts=options(argv)
            if task not in preflights:
                _,_,_,manifest=preflight(opts)
                preflights[task]=manifest
            jobs.append(dict(id=task+'_'+cond,kind='evaluation',readiness='stacking_status',
                task=task,condition=cond,episodes=args.episodes,peak_vram_mib=args.job_gb*1024,
                ram_mib=args.job_ram_gb*1024,cwd=str(WORKSPACE_ROOT),
                argv=['bash',str(WORKSPACE_ROOT/'scripts'/('eval_stacking_adapter.sh' if new_adapter else 'eval_stacking.sh'))]+argv))
    return normalize_jobs(dict(schema=1,jobs=jobs)),preflights


def execute_pool(args, specs, preflights):
    monitor=GpuMonitor(lambda:gpu_snapshot(args.gpu))
    initial=monitor.initial()
    lock_path=Path('/tmp')/('cx_gpu_pool_'+str(os.getuid())+'_'+initial['uuid']+'.lock')
    lock=lock_path.open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another pool already manages this GPU on this host')
    run=new_run('pool',group=('gpu_pools',));print('POOL',run,flush=True)
    host=host_resources();limit=args.max_workers or min(len(specs),max(1,host['cpus']//args.cpus_per_job))
    write_json(run/'manifest.json',dict(schema=1,gpu=args.gpu,gpu_uuid=initial['uuid'],jobs=specs,
        options=vars(args),max_workers_effective=limit,host=host,preflights=preflights,
        comparison='Independent model/simulator/compiler processes; no exact within-process pairing claim',
        resource_claim='Admission estimate only; real concurrent GPU validation and throughput comparison still required'))
    jobs=[];pending=list(specs);halt=False;reason='';started=time.monotonic();last_print=0;no_work_since=None
    peak_gpu=initial['used_mib'];peak_workers=0;failure=None
    def cancel(signum,frame): raise KeyboardInterrupt('Pool cancellation')
    previous={sig:signal.signal(sig,cancel) for sig in (signal.SIGINT,signal.SIGTERM)}
    def serializable(j):
        return {k:v for k,v in j.items() if k not in ('process','log')}
    try:
        with (run/'resources.jsonl').open('w',buffering=1) as telemetry:
            while pending or any(j['state']=='running' for j in jobs):
                snap=monitor.observe();host=host_resources();now=time.monotonic()
                for j in jobs:
                    if j['state']!='running': continue
                    used,rss=group_usage(j['pid'],snap['process_mib']);j.update(used_mib=used,rss_mib=rss,peak_mib=max(j['peak_mib'],used))
                    status=read_json(Path(j['job_dir'])/'worker_status.json');j['worker_status']=status
                    j['initial_completed']=max(j.get('initial_completed',0),status.get('initial_completed',0))
                    if j['spec']['readiness']=='stacking_status':
                        j['ready']=bool(status.get('ready') and status.get('completed',0)>=1)
                        completed=status.get('completed',0)
                        if completed>j.get('announced',0) and status.get('last_episode'):
                            ep=status['last_episode'];j['announced']=completed
                            print(episode_output(j['spec']['task'],'stacking/'+j['spec']['condition'],
                                'demo_randomized','stacking_adapter_seed0' if getattr(args,'training_run',None) else 'frozen_A',ep['success'],status['successes'],completed,ep['seed']),end='',flush=True)
                    else:j['ready']=now-j['started_mono']>=args.stagger_seconds
                    rc=j['process'].poll()
                    if rc is not None:
                        status=read_json(Path(j['job_dir'])/'worker_status.json');j['worker_status']=status
                        j['initial_completed']=max(j.get('initial_completed',0),status.get('initial_completed',0))
                        ok,label,result=completed_job(j['spec'],status,rc)
                        j.update(state='complete' if ok else 'failed',returncode=rc,validation=label,
                                 result=result,elapsed_s=now-j['started_mono']);j['log'].close()
                        try:cleanup_scratch(j)
                        except OSError as exc:j['scratch_cleanup_error']=str(exc)
                        print(f"[{j['spec']['id']}] {j['state']}: {label}; peak {j['peak_mib']/1024:.1f} GiB",flush=True)
                        if not ok:halt=True;reason='worker_failed_no_automatic_retry'
                running=[j for j in jobs if j['state']=='running']
                if pending and not halt:
                    ok,why=admission(pending[0],running,snap,host,headroom=args.peak_headroom,
                        reserve_mib=args.reserve_gb*1024,startup_mib=args.startup_extra_gb*1024,
                        ram_reserve_mib=8192,cpus_per_job=args.cpus_per_job,max_workers=limit)
                    if not snap['telemetry_fresh']:ok=False;why='telemetry_unavailable_existing_workers_continue'
                    for folder in (WORKSPACE_ROOT,'/tmp'):
                        if shutil.disk_usage(folder).free<args.min_disk_gb*1024**3:ok=False;why='disk_headroom'
                    if ok:
                        spec=pending.pop(0);folder=run/spec['id'];folder.mkdir()
                        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),
                            CX_GPU_JOB_STATUS=str(folder/'worker_status.json'),CX_GPU_JOB_ID=spec['id'],
                            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                            OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
                        scratch=Path(tempfile.mkdtemp(prefix='cxgp_',dir='/tmp'))
                        for key in ('XDG_CACHE_HOME','HF_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TMPDIR'):
                            cache=scratch/key.lower();cache.mkdir();env[key]=str(cache)
                        env['TMP']=env['TEMP']=env['TMPDIR']
                        write_json(folder/'scratch.json',dict(path=str(scratch),environment={k:env[k] for k in ('TMPDIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR')}))
                        argv=[x.replace('{gpu}',str(args.gpu)).replace('{job_dir}',str(folder)).replace('{workspace}',str(WORKSPACE_ROOT)) for x in spec['argv']]
                        log=(folder/'console.log').open('w',buffering=1)
                        try:proc=subprocess.Popen(argv,cwd=spec['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        except BaseException:log.close();shutil.rmtree(scratch);raise
                        jobs.append(dict(spec=spec,state='running',process=proc,log=log,pid=proc.pid,
                            job_dir=str(folder),scratch=str(scratch),scratch_removed=False,
                            started_mono=now,started_epoch=time.time(),ready=False,peak_mib=0,used_mib=0,rss_mib=0))
                        print(f"START {spec['id']} pid={proc.pid}, reservation={spec['peak_vram_mib']/1024:g} GiB; log={folder/'console.log'}",flush=True)
                        no_work_since=None
                    elif not running:
                        if no_work_since is None:no_work_since=now
                        if now-no_work_since>=args.admission_timeout:
                            halt=True;reason='admission_timeout_'+why
                else:why=reason or 'all_jobs_launched'
                running=[j for j in jobs if j['state']=='running'];peak_workers=max(peak_workers,len(running));peak_gpu=max(peak_gpu,snap['used_mib'])
                record=dict(epoch=time.time(),elapsed_s=now-started,running=len(running),pending=len(pending),
                    gpu={k:v for k,v in snap.items() if k!='process_mib'},host=host,admission=why,
                    workers=[dict(id=j['spec']['id'],used_mib=j['used_mib'],peak_mib=j['peak_mib'],ready=j['ready']) for j in running])
                telemetry.write(json.dumps(record)+'\n')
                replace_json(run/'status.json',dict(phase='draining_after_failure' if halt else 'running',reason=reason,
                    jobs=[serializable(j) for j in jobs],pending=[s['id'] for s in pending],resource=record))
                if now-last_print>=30:
                    print(f"POOL active={len(running)}/{limit} pending={len(pending)} GPU={snap['used_mib']/1024:.1f}/{snap['total_mib']/1024:.1f} GiB util={snap['utilization']:.0f}% admission={why}",flush=True);last_print=now
                if halt and not running:break
                if pending or running:time.sleep(args.poll_seconds)
    except BaseException as exc:
        failure=repr(exc);halt=True;reason='scheduler_interrupted_or_failed'
    finally:
        active=[j for j in jobs if j['process'].poll() is None]
        for j in active:
            try:os.killpg(j['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
        deadline=time.monotonic()+20
        for j in active:
            try:j['process'].wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
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
        episodes=sum(max(0,j.get('worker_status',{}).get('completed',0)-j.get('initial_completed',0)) for j in jobs if j['spec']['readiness']=='stacking_status')
        outcomes=[]
        for j in jobs:
            r=j.get('result')
            if r and j['spec']['readiness']=='stacking_status':
                outcomes.append(dict(task=r['task'],condition=r['condition'],complete=r['complete'],
                    successes=r.get('successes'),completed=r['completed'],success_rate=r.get('success_rate'),
                    valid_seeds=r.get('valid_seeds'),run=j['worker_status'].get('run')))
                print(f"{r['task']} / {r['condition']}: {r.get('successes')}/{r['completed']} ({r.get('success_rate',0):.1%})",flush=True)
        result=dict(complete=not halt and not pending and all(j['state']=='complete' for j in jobs),reason=reason,
            error=failure,elapsed_s=elapsed,completed_eval_episodes=episodes,
            episodes_per_hour_including_startup=episodes/elapsed*3600 if elapsed else None,
            max_concurrent_workers=peak_workers,peak_gpu_mib=peak_gpu,
            telemetry_failures=monitor.total_failures,
            jobs=[serializable(j) for j in jobs],eval_outcomes=outcomes,not_started=[s['id'] for s in pending],
            throughput_note='No serial speedup claim without measured comparable serial workload; GPU saturation is not success',
            resume_note='No silent retries or episode splicing; retain partial outputs and audit before a new protocol')
        write_json(run/'result.json',result);replace_json(run/'status.json',dict(phase='finished' if result['complete'] else 'incomplete',**result))
        if getattr(args,'profile_only',False):
            write_json(run/'measured_profiles.json',dict(
                validation='Sampled peaks of completed commands; reserve extra memory for missed transient/future peaks',
                jobs=[dict(id=j['spec']['id'],sampled_peak_vram_mib=j['peak_mib'],
                           recommended_reservation_mib=math.ceil(j['peak_mib']*(1+args.peak_headroom)+1024),
                           original_spec=j['spec']) for j in jobs if j['state']=='complete' and j['peak_mib']>0]))
        print('POOL RESULT',run/'result.json',flush=True);lock.close()
    return 0 if result['complete'] else 1


def main(argv=None):
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--manifest',type=Path,help='Generic jobs JSON; training/collection requires measured peaks')
    ap.add_argument('--tasks',default='stack_blocks_two')
    ap.add_argument('--conditions')
    ap.add_argument('--training-run',help='New stacking adapter; automatically resumes matching whole-episode evaluations')
    ap.add_argument('--event-cache-limit',type=int,help='Optional saved event packets per episode; 0 keeps query metadata only (default: new adapter 0, old adapter 6)')
    ap.add_argument('--seed-start',type=int,default=7105000)
    ap.add_argument('--episodes',type=int,default=100);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--max-workers',type=int,default=0,help='0 = automatic within CPU/memory constraints')
    ap.add_argument('--job-gb',type=float,default=18);ap.add_argument('--job-ram-gb',type=float,default=32)
    ap.add_argument('--reserve-gb',type=float,default=6);ap.add_argument('--startup-extra-gb',type=float,default=1)
    ap.add_argument('--peak-headroom',type=float,default=.2);ap.add_argument('--cpus-per-job',type=int,default=4)
    ap.add_argument('--min-disk-gb',type=float,default=10);ap.add_argument('--stagger-seconds',type=float,default=60)
    ap.add_argument('--poll-seconds',type=float,default=5);ap.add_argument('--admission-timeout',type=float,default=600)
    ap.add_argument('--dry-run',action='store_true',help='CPU preflight only; never queries GPU or launches jobs')
    ap.add_argument('--profile-only',action='store_true',help='Run declared jobs serially, allowing unmeasured generic jobs; records peaks, never alters training settings')
    args=ap.parse_args(argv)
    if args.conditions is None:
        args.conditions='Z_zero,M_first_full,M_event_full' + ('' if args.training_run else ',M_legacy_candidate')
    if args.training_run:args.training_run=str(output_path(Path(args.training_run).resolve()))
    if args.gpu<0 or args.max_workers<0 or args.cpus_per_job<1:ap.error('Invalid GPU/worker/CPU count')
    if args.event_cache_limit is not None and not 0<=args.event_cache_limit<=100:ap.error('Event cache limit must be 0..100')
    for key in ('job_gb','job_ram_gb','reserve_gb','startup_extra_gb','min_disk_gb','stagger_seconds','poll_seconds','admission_timeout'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<=0:ap.error('Positive finite '+key+' required')
    if not math.isfinite(args.peak_headroom) or not 0<=args.peak_headroom<=2:ap.error('Invalid peak headroom')
    if args.profile_only:args.max_workers=1
    if args.manifest:specs=normalize_jobs(read_json(args.manifest),allow_unprofiled=args.profile_only);preflights={};args.manifest=str(args.manifest.resolve())
    else:specs,preflights=build_eval_jobs(args)
    if args.dry_run:
        print(json.dumps(dict(validation='CPU only; no GPU rollout/concurrency validation',jobs=specs,
            native_budgets={k:v['official_task_action_budget'] for k,v in preflights.items()},
            memory_policy=dict(reserve_gb=args.reserve_gb,startup_extra_gb=args.startup_extra_gb,peak_headroom=args.peak_headroom),
            gpu=args.gpu,host=host_resources()),ensure_ascii=False,indent=2));return
    raise SystemExit(execute_pool(args,specs,preflights))


if __name__=='__main__':main()
