"""S1b: one fixed state, fresh compiled base, zero sham, teacher horizon and projection."""
import argparse
import hashlib
import importlib
import json
import signal
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .audit import digest
from .collect_recovery import DEFAULT_PROTOCOL, signature, save_observation, infer, rng_json
from .eval_diagnostics import state_snapshot, array_hash, json_hash
from .evaluate_b2 import configure_runtime, resolve_tokenizer, seed_all
from .paths import ROOT, new_run, write_json, replace_json
from .probe_action_replay import load_source, source_runtime, source_scene_index, prime_warp_cache, compare_physical
from .probe_recovery_teacher import diagnostics
from .recovery_contract import rng_capture, rng_restore, rng_fingerprint, terminal_label
from .teacher_projection import ProjectionExecutor, ReplanningTeacher, IsolatedQuery, CAPS, ProjectionStop, PlanningFailed, replay_teacher_prefix

from .compatible_teacher import CompatibleTeacher

BRANCHES = ('base','sham_zero','projected','compatible')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--point',type=int,choices=(4,5,6,7),default=4)
    parser.add_argument('--policy-seed',type=int,default=910000)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if not args.gpu.isdigit():parser.error('GPU index required')
    protocol=json.loads(DEFAULT_PROTOCOL.read_text());point=protocol['points'][args.point]
    source=Path(point['source']);manifest,episode,rows=load_source(source,point['seed'])
    for path,sha in ((source/'manifest.json',point['source_manifest_sha256']),
                     (source/'resolved_config.json',point['source_config_sha256']),
                     (source/'traces'/f"r0_seed{point['seed']}.jsonl",point['source_trace_sha256'])):
        if digest(path)!=sha:raise ValueError('Changed source '+str(path))
    runtime=source_runtime(source);config=json.loads((source/'resolved_config.json').read_text())
    native_limit=500 if point['task']=='place_shoe' else 400
    job=dict(point=args.point,task=point['task'],scene_seed=point['seed'],policy_seed=args.policy_seed,
        branches=list(BRANCHES),prefix_actions=point['prefix_action_count'],native_limit=native_limit,
        caps=CAPS.tolist(),planner_stride=config['save_freq'],video=False,privileged_teacher=True,
        protocol_version='S1g measured native compatibility with explicit alignment; full continuation',
        short_branch_prefix='none; all branches use common initial measured fork and base block; subsequent queries are fresh',
        inference='official compiled sample_actions, internal noise, 32 callbacks/observations per query',
        rng='separate saved policy RNG stream; planner RNG preserved across queries; common initial query reused',
        replanning='current measured state at every model query boundary and each semantic phase; no stored future replay',
        phase_goal='official semantic target fixed within phase; not arbitrary object-tracking or optimal expert',
        phase_advance='compatible branch: measured native EE position <=5mm and orientation <=3deg; gripper endpoint command with measured fingers logged, not a grasp certificate; controls retain old rule',
        fallback='teacher horizon / planning failure / script finished -> L0 in remaining budget; unexpected errors censored',
        selected='preselected developer checkpoint, first policy seed; not a holdout or trained adapter score',
        stop_gates='Stop before candidates if full base/sham trajectory differs; S1d unbounded feasibility is historical, not a current-run branch',
        protocol=str(DEFAULT_PROTOCOL),protocol_sha256=digest(DEFAULT_PROTOCOL),source=str(source))
    if args.dry_run:print(json.dumps(job,ensure_ascii=False,indent=2));return
    run=new_run('gate',group=('compatible_continuations',point['task']));print(run,flush=True)
    files=[Path(__file__),ROOT/'cx_tta/compatible_teacher.py',ROOT/'cx_tta/constraint_teacher.py',ROOT/'cx_tta/teacher_projection.py',ROOT/'cx_tta/native_teacher.py',
           ROOT/'cx_tta/collect_recovery.py',ROOT/'cx_tta/evaluate_b2.py',ROOT/'cx_tta/recovery_contract.py',
           runtime/'envs/_base_task.py',runtime/'envs'/f"{point['task']}.py"]
    write_json(run/'manifest.json',dict(**job,code_sha256={str(p):digest(p) for p in files}))
    active=executor=None;reports=[];started=time.monotonic()
    def timeout(*_):raise TimeoutError('Bounded gate wall limit 5400 seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(5400)
    try:
        prime_warp_cache(source,run);configure_runtime(run,runtime,resolve_tokenizer())
        import torch
        from .cache_expert import create_base_policy
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
        torch.set_num_threads(1)
        replace_json(run/'progress.json',dict(phase='loading_policy',reports=reports))
        policy=create_base_policy()
        if torch._dynamo.config.disable or not hasattr(policy._model.sample_actions,'_torchdynamo_orig_callable'):
            raise RuntimeError('Official compiled policy required')
        task_cls=getattr(importlib.import_module('envs.'+point['task']),point['task'])
        config.update(save_path=str(run/'runtime_data'),cx_runtime=str(runtime),
            save_data=False,eval_video_log=False,eval_video_save_dir=None)
        def replay():
            nonlocal active
            active=task_cls();seed_all(point['seed'])
            active.setup_demo(now_ep_num=source_scene_index(manifest,point['seed']),seed=point['seed'],
                              **dict(config,need_plan=True,is_test=True))
            active.set_instruction(instruction=point['instruction']);obs=active.get_obs()
            for c in range(point['chunk']):
                if signature(obs)!=rows[c]['observation_components']:raise RuntimeError('Prefix observation mismatch')
                for a in rows[c]['base_executed']:
                    n=active.take_action_cnt;active.take_action(np.asarray(a,np.float32));obs=active.get_obs()
                    if active.take_action_cnt!=n+1:raise RuntimeError('Prefix stopped')
            physical=compare_physical(rows[point['chunk']]['before'],state_snapshot(active),1e-5)
            if not physical['matched'] or signature(obs)!=point['observation_signature']:
                raise RuntimeError('Fork mismatch')
            if active.take_action_cnt!=job['prefix_actions'] or active.step_lim!=native_limit or active.eval_success:
                raise RuntimeError('Fork terminal or budget changed')
            return obs,physical
        obs,physical=replay();planner_rng=rng_capture();policy.reset()
        replace_json(run/'progress.json',dict(phase='warming_compiled_policy',reports=reports))
        seed_all(args.policy_seed);infer(policy,obs,point['instruction'])
        seed_all(args.policy_seed);pre=rng_capture();base=infer(policy,obs,point['instruction']);post=rng_capture()
        write_json(run/'common_rng.json',dict(before=rng_json(pre),after=rng_json(post)))
        np.savez_compressed(run/'common_base.npz',base=base)
        rng_restore(planner_rng)
        for index,branch in enumerate(BRANCHES):
            if index:obs,physical=replay()
            planner_rng=rng_capture();policy.reset()
            folder=run/branch;folder.mkdir();save_observation(folder/'fork.npz',obs)
            write_json(folder/'fork.json',dict(observation=signature(obs),physical=physical,
                native_limit=native_limit,prefix_actions=job['prefix_actions'],diagnostics=diagnostics(active)))
            caps=np.zeros(14,np.float32) if branch=='sham_zero' else CAPS if branch in ('projected','projected32','compatible') else None
            first_rng=dict(before=rng_fingerprint(pre),after=rng_fingerprint(post))
            query=IsolatedQuery(lambda o:infer(policy,o,point['instruction']),post)
            trajectory=hashlib.sha256();before_obs=signature(obs);branch_start=time.monotonic()
            query_records=[]
            def on_query(q,observation,actions,rng):
                save_observation(folder/f'observation_{q:04d}.npz',observation)
                np.savez_compressed(folder/f'base_{q:04d}.npz',actions=actions)
                entry=dict(query=q,observation=signature(observation),rng=rng,base_sha256=array_hash(actions),
                    native_count=int(active.take_action_cnt),physical=state_snapshot(active))
                query_records.append(entry)
                with (folder/'queries.jsonl').open('a') as stream:stream.write(json.dumps(entry)+'\n')
                replace_json(run/'progress.json',dict(phase='running',branch=branch,query=q,
                    action_count=int(active.take_action_cnt),reports=reports,seconds=time.monotonic()-started))
            def on_action(row,observation):
                nonlocal before_obs
                row.update(before_observation=before_obs,after_observation=signature(observation),
                    offline_only=diagnostics(active))
                before_obs=row['after_observation']
                comparable={k:row[k] for k in ('query','slot','action','advanced','count','padding',
                                              'success','after_observation')}
                comparable['physical']=row['offline_only']['physical']
                trajectory.update(json_hash(comparable).encode())
                with (folder/'trace.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            def on_plan(pid,dense,indices,meta):
                np.savez_compressed(folder/f'plan_{pid:04d}.npz',dense_commands=dense,indices=indices)
                write_json(folder/f'plan_{pid:04d}.json',dict(**meta,physical=state_snapshot(active),
                    observation=signature(executor.obs)))
            def on_attempt(aid,meta):
                entry=dict(attempt=aid,**meta,physical=state_snapshot(active),
                    observation=signature(executor.obs),offline_only=diagnostics(active))
                with (folder/'planning_attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(entry,allow_nan=False)+'\n')
                if not meta['returned_plan'] or not meta['plan_success']:
                    save_observation(folder/f'planning_failure_{aid:04d}.npz',executor.obs)
            executor=ProjectionExecutor(active,obs,query,first_base=base,first_rng=first_rng,
                on_query=on_query,on_action=on_action,horizon=32 if branch in ('teacher32','projected32') else None,caps=caps)
            teacher=None;reason='base';fallback=False
            prefix_source=None
            if branch in ('teacher32','projected32'):
                parent='teacher_full' if branch=='teacher32' else 'projected'
                parent_folder=run/parent
                prefix_rows=[json.loads(l) for l in (parent_folder/'trace.jsonl').read_text().splitlines()][:32]
                parent_fork=json.loads((parent_folder/'fork.json').read_text())
                prefix_source=dict(branch=parent,trace_sha256=digest(parent_folder/'trace.jsonl'),
                    result_sha256=digest(parent_folder/'result.json'),actions=32)
                write_json(folder/'prefix_source.json',prefix_source)
                def verify_before(i,row):
                    expected=parent_fork['diagnostics']['physical'] if i==0 else prefix_rows[i-1]['offline_only']['physical']
                    if state_snapshot(active)!=expected or signature(executor.obs)!=row['before_observation']:
                        raise RuntimeError('Common teacher prefix predecessor mismatch')
                def verify_after(i,row):
                    if state_snapshot(active)!=row['offline_only']['physical'] or signature(executor.obs)!=row['after_observation']:
                        raise RuntimeError('Common teacher prefix reached-state mismatch')
                def replay_plan(pid):
                    meta=json.loads((parent_folder/f'plan_{pid:04d}.json').read_text())
                    for key in ('physical','observation'):meta.pop(key)
                    with np.load(parent_folder/f'plan_{pid:04d}.npz',allow_pickle=False) as z:
                        on_plan(pid,z['dense_commands'],z['indices'],meta)
                count=replay_teacher_prefix(executor,prefix_rows,verify_before,verify_after,replay_plan)
                teacher=SimpleNamespace(plans=count,phase=prefix_rows[-1]['phase'],attempts=0)
                reason='teacher_horizon';fallback=True
            elif branch!='base':
                def on_measurement(record):
                    with (folder/'measurements.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
                teacher=(CompatibleTeacher(active,executor,config['save_freq'],on_plan,on_attempt,on_measurement=on_measurement) if branch=='compatible' else ReplanningTeacher(active,executor,config['save_freq'],on_plan,on_attempt))
                try:
                    active.play_once();reason='teacher_script_finished'
                except (ProjectionStop,PlanningFailed) as exc:reason=str(exc)
                fallback=not executor.terminal
            executor.continue_base()
            outcome=terminal_label(int(active.take_action_cnt),native_limit,bool(active.eval_success))
            if not outcome['complete']:raise RuntimeError('Unfinished branch')
            save_observation(folder/'terminal.npz',executor.obs)
            record=dict(branch=branch,**outcome,teacher_exit=reason,l0_fallback=fallback,
                action_count=int(active.take_action_cnt),prefix_actions=job['prefix_actions'],native_limit=native_limit,
                continuation_actions=executor.executed,teacher_actions=executor.teacher_actions,
                queries=executor.queries,observation_updates=executor.attempts,physics_steps=executor.guard.steps,
                teacher_plans=0 if teacher is None else teacher.plans,completed_phases=0 if teacher is None else teacher.phase,
                planning_attempts=0 if teacher is None else teacher.attempts,
                planning_attempts_sha256=digest(folder/'planning_attempts.jsonl') if (folder/'planning_attempts.jsonl').exists() else None,
                prefix_source=prefix_source,
                physics_dt=float(active.scene.get_timestep()),trajectory_sha256=trajectory.hexdigest(),
                trace_sha256=digest(folder/'trace.jsonl'),queries_sha256=digest(folder/'queries.jsonl'),
                final_diagnostics=diagnostics(active),seconds=time.monotonic()-branch_start)
            write_json(folder/'result.json',record);reports.append(record)
            active.scene=executor.guard.scene;active.close_env(clear_cache=False);active=None;executor=None
            print('BRANCH_COMPLETE',branch,'success',record['success'],'actions',record['continuation_actions'],
                  'teacher_exit',reason,flush=True)
            if branch=='sham_zero':
                sham=reports[0]['trajectory_sha256']==record['trajectory_sha256']
                write_json(run/'sham_check.json',dict(passed=sham,base=reports[0]['trajectory_sha256'],sham=record['trajectory_sha256']))
                if not sham:raise RuntimeError('Base/zero-sham mismatch; do not compare candidates')
            if branch=='teacher_full' and (not record['success'] or fallback):
                raise RuntimeError('Unbounded replanning teacher control failed; projection interpretation blocked')
        write_json(run/'result.json',dict(status='complete',reports=reports,
            scope='One developer checkpoint and one policy seed; privileged feasibility, no student/generalization claim'))
        replace_json(run/'progress.json',dict(phase='complete',reports=reports,seconds=time.monotonic()-started))
        print('COMPLETE',str(run),flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed_branches=len(reports)))
        replace_json(run/'progress.json',dict(phase='failed_or_interrupted',reports=reports))
        raise
    finally:
        signal.alarm(0)
        if active is not None and hasattr(active,'scene'):
            if executor is not None:active.scene=executor.guard.scene
            active.close_env(clear_cache=False)


if __name__=='__main__':main()
