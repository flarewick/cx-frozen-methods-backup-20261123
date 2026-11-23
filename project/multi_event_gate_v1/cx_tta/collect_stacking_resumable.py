"""Same expert plan/replay contract, durable accepted scenes and isolated restart sessions."""
import atexit
import copy
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import shutil
import signal
import sys
import traceback

from .audit import digest
from .paths import new_run, output_path, write_json, replace_json
from .collection_jobs import read
from .collection_resume import resume_state, check_unmanaged_writer, local_process_identity
from .collect_stacking_experts import options, cleanup_frames, preflight as preflight_v1
from .expert_screening import run_expert_plan, ExpertPlanningFailure
from .gpu_job_status import publish_job


def checkpoint(run, rows, requested):
    accepted = sum(r.get('accepted') is True for r in rows)
    replace_json(run/'attempts.json', rows)
    replace_json(run/'progress.json', dict(accepted=accepted, requested=requested, attempts=len(rows)))
    return accepted


def collect(task, config, args, p, run, session, state, seed_all, unstable_type, export, qualify):
    spec = p['splits'][args.split]
    rows = read(run/'attempts.json') if (run/'attempts.json').exists() else []
    initial = accepted = state['accepted']
    terminal = set(state['terminal_seeds'])
    sha = digest(args.protocol)
    for seed in range(spec['start'], spec['stop']):
        if accepted >= spec['episodes']:
            break
        if seed in terminal:
            continue
        if shutil.disk_usage(run).free < args.min_free_gb * 2**30:
            raise OSError('Insufficient free disk; stopped before next scene')
        # Interrupted attempts retain their evidence. Never overwrite an old
        # HDF5 or append frames from a new simulator process to an old episode.
        attempt = run/'expert_data'/f'seed_{seed}'/session.name
        row = dict(task=args.task, seed=seed, split=args.split, episode_index=accepted,
            accepted=False, plan_success=False, replay_success=False, session=session.name)
        opened = False
        try:
            for replay in (False, True):
                seed_all(seed)
                opts = copy.deepcopy(config)
                opts.update(need_plan=not replay, save_data=replay, save_path=str(attempt))
                opened = True
                task.setup_demo(now_ep_num=accepted, seed=seed, **opts)
                if replay:
                    traj = task.load_tran_data(accepted)
                    opts.update(left_joint_path=traj['left_joint_path'], right_joint_path=traj['right_joint_path'])
                    task.set_path_lst(opts)
                info = task.play_once() if replay else run_expert_plan(task)
                success = bool(task.plan_success and task.check_success())
                row['replay_success' if replay else 'plan_success'] = success
                if not replay and success:
                    task.save_traj_data(accepted)
                if replay and success:
                    h5 = export(task, config['cx_runtime'], args.task, info, seed, accepted)
                    import h5py
                    with h5py.File(h5, 'r+') as h:
                        h.attrs.update(stacking_split=args.split, stacking_protocol_sha256=sha,
                                       instruction_split='seen', data_role=spec['role'])
                    row.update(hdf5=str(h5), capture=qualify(h5,args.task,seed,args.split,sha),
                               scene_info=info, accepted=True)
                    # Persist before cache cleanup/env teardown; a completed
                    # readback remains recoverable even if teardown is interrupted.
                    checkpoint(run, rows+[row], spec['episodes'])
                if replay:
                    cleanup_frames(task, attempt)
                task.close_env(); opened = False
                if not success:
                    row['excluded_reason'] = 'expert_replay_failed' if replay else 'expert_plan_failed'
                    break
        except (ExpertPlanningFailure, unstable_type) as exc:
            row.update(excluded_reason='expert_missing_grasp_target' if isinstance(exc,ExpertPlanningFailure)
                else 'unstable_scene', error_type=type(exc).__name__, error=str(exc))
        except BaseException:
            row['error'] = traceback.format_exc()
            rows.append(row); checkpoint(run,rows,spec['episodes'])
            raise
        finally:
            if opened:
                try:task.close_env()
                except Exception:pass
        rows.append(row)
        accepted = checkpoint(run, rows, spec['episodes'])
        publish_job(run,'collecting',ready=accepted>initial,completed=accepted,requested=spec['episodes'],
                    initial_accepted=initial,session_accepted=accepted-initial)
        print(f'\033[32mEXPERT DATA {args.task}/{args.split}: {accepted}/{spec["episodes"]} accepted '
              f'(kept {initial}, new {accepted-initial}), {len(rows)} attempts\033[0m',flush=True)
    return dict(status='collection_complete' if accepted==spec['episodes'] else 'insufficient_expert_successes',
        task=args.task,split=args.split,accepted_episodes=accepted,attempts=len(rows),
        initially_accepted=initial,new_accepted=accepted-initial,
        residual_training_ready=False,gate_training_ready=False,policy_evaluation=False)


def execute(a,p,source,tokenizer,run,session,state,revision):
    from .native_entry_storage import NativeEntryStorage
    from .collect_smoke import make_config
    from .recording import instrument_runtime
    from .stacking_training_data import qualify_capture
    if revision=='v2':
        from .stacking_recording import install_recording
        from .collect_stacking_experts_v2 import export_recording as export
    else:
        from .recording import install_recording
        from .capture import export_capture as export
    storage=NativeEntryStorage(runtime_source=source)
    atexit.register(storage.cleanup)
    runtime=storage.prepare_runtime(session)
    original=digest(runtime/'envs/_base_task.py')
    count=instrument_runtime(runtime)
    write_json(session/'instrumentation.json',dict(original_sha256=original,
        instrumented_sha256=digest(runtime/'envs/_base_task.py'),counted_step_calls=count,
        recorder_revision=revision))
    storage.configure_runtime(session,runtime,tokenizer)
    import torch
    import torch.multiprocessing as mp
    from .evaluate_b2 import seed_all
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; collection not started')
    mp.set_start_method('spawn',force=True)
    a.task_config=p['task_config']
    config=make_config(runtime,a,session)
    config.update(eval_mode=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
    write_json(session/'resolved_config.json',config)
    task=getattr(importlib.import_module('envs.'+a.task),a.task)()
    from envs.utils.create_actor import UnStableError
    install_recording(task,effect_schema=True)
    result=collect(task,config,a,p,run,session,state,seed_all,UnStableError,export,qualify_capture)
    write_json(session/'source_runtime_audit.json',storage.verify_source())
    from .stacking_training_design import load_design
    load_design(a.protocol)
    return result


def run_job(spec,gpu,existing):
    """Caller holds the shared task/split scene lock throughout this function."""
    a=options(['--protocol',spec['protocol'],'--task',spec['task'],'--split',spec['split'],
               '--gpu',str(gpu),'--min-free-gb',str(spec['min_free_gb'])])
    if spec['collector_version']=='v2':
        from .collect_stacking_experts_v2 import preflight
    else:preflight=preflight_v1
    p,source,tokenizer,manifest=preflight(a)
    if existing.get('resume'):
        run=output_path(Path(existing['resume']['run']))
        check_unmanaged_writer(run)
        state=resume_state(run,spec,decode=True)
    else:
        run=new_run('collect',group=('stacking_training','expert',a.task,a.split))
        manifest.update(protocol=str(a.protocol),protocol_sha256=digest(a.protocol),resumable=True)
        write_json(run/'manifest.json',manifest)
        state=resume_state(run,spec)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    session=run/'sessions'/('session_'+stamp);session.mkdir(parents=True)
    files=[Path(__file__),Path(__file__).with_name('collection_resume.py'),
           Path(__file__).with_name('collection_worker.py')]
    write_json(session/'manifest.json',dict(dataset=str(run),protocol_sha256=digest(a.protocol),
        collector_revision=spec['collector_version'],initial=state,writer=local_process_identity(),
        source_sha256={str(f):digest(f) for f in files}))
    # Keep all previous status/error records instead of overwriting history.
    for name in ('result.json','data_audit.json','failure.json','writer.json'):
        path=run/name
        if path.exists():shutil.copy2(path,session/('previous_'+name))
    replace_json(run/'writer.json',dict(local_process_identity(),session=str(session)))
    rows=read(run/'attempts.json') if (run/'attempts.json').exists() else []
    checkpoint(run,rows,spec['episodes'])
    initial=state['accepted']
    publish_job(run,'initializing',ready=False,initial_accepted=initial,session_accepted=0)
    print('RESUME' if initial else 'RUN',run,f'{initial}/{spec["episodes"]} accepted',flush=True)
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    from .evaluate_stacking import Tee
    old_out,old_err=sys.stdout,sys.stderr
    try:
        with (session/'console.log').open('x') as log:
            sys.stdout,sys.stderr=Tee(old_out,log),Tee(old_err,log)
            try:
                if state['remaining']:
                    result=execute(a,p,source,tokenizer,run,session,state,spec['collector_version'])
                else:
                    result=dict(status='collection_complete',task=a.task,split=a.split,
                        accepted_episodes=initial,attempts=len(rows),new_accepted=0,
                        residual_training_ready=False,gate_training_ready=False,policy_evaluation=False)
                replace_json(run/'result.json',result)
                if result['status']!='collection_complete':
                    publish_job(run,'failed',ready=False,reason=result['status'],initial_accepted=initial)
                    return 2
                from .stacking_training_data import audit_collection
                audit=audit_collection(run,a.protocol)
                replace_json(run/'data_audit.json',audit)
                write_json(session/'result.json',dict(result=result,audit=audit))
                if (run/'failure.json').exists():
                    (run/'failure.json').unlink()  # Exact previous copy retained above.
                publish_job(run,'finished',ready=False,result=str(run/'result.json'),initial_accepted=initial,
                    session_accepted=result['accepted_episodes']-initial)
                print('COLLECTION COMPLETE',run,flush=True)
                return 0
            except BaseException:
                failure=dict(traceback=traceback.format_exc(),session=str(session))
                replace_json(run/'failure.json',failure)
                write_json(session/'failure.json',failure)
                publish_job(run,'failed',ready=False,initial_accepted=initial,**failure)
                raise
            finally:
                sys.stdout.close();sys.stderr.close()
    finally:
        sys.stdout,sys.stderr=old_out,old_err
