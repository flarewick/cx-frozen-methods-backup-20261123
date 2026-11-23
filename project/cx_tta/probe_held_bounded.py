"""S1k: matched held-entry continuation with original caps and fresh compiled VLA."""
import argparse
import hashlib
import importlib
import json
import signal
import shutil
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
from .held_teacher import enable_held_continuation

BRANCHES = ('base','sham_zero','compatible','held_compatible')
SUPPORT = ROOT/'outputs/handoff_support/place_shoe/probe_20261001_154658_701256'
NATIVE_HELD = ROOT/'outputs/recovery_stage_controls/place_shoe/probe_20261001_155532_413918'
CACHE_SOURCE = ROOT/'outputs/compatible_continuations/place_shoe/gate_20261001_141842_178427'


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--point',type=int,choices=(4,),default=4)
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
    support=json.loads((SUPPORT/'contact_reanalysis/result.json').read_text())
    support_manifest=json.loads((SUPPORT/'manifest.json').read_text())
    support_fork=json.loads((SUPPORT/'hold_native/fork.json').read_text())
    native_audit=json.loads((NATIVE_HELD/'audit.json').read_text())
    if not support['passed'] or not all(x['local_support_passed'] for x in support['reports']) or not native_audit['passed']:
        raise ValueError('Verified support and unbounded held continuation required')
    if str(source)!=support_manifest['source'] or point['seed']!=support_manifest['scene_seed']:
        raise ValueError('Support scene/source mismatch')
    runtime=source_runtime(source);config=json.loads((source/'resolved_config.json').read_text())
    native_limit=500 if point['task']=='place_shoe' else 400
    job=dict(point=args.point,task=point['task'],scene_seed=point['seed'],policy_seed=args.policy_seed,
        branches=list(BRANCHES),prefix_actions=point['prefix_action_count'],native_limit=native_limit,
        caps=CAPS.tolist(),planner_stride=config['save_freq'],video=False,privileged_teacher=True,
        protocol_version='S1k original caps, fresh official compiled queries, paired held-entry vs restart entry',
        short_branch_prefix='none; all branches use common initial measured fork and base block; subsequent queries are fresh',
        inference='official compiled sample_actions, internal noise, 32 callbacks/observations per query',
        rng='separate saved policy RNG stream; planner RNG preserved across queries; common initial query reused',
        replanning='current measured state at every model query boundary and each semantic phase; no stored future replay',
        phase_goal='official semantic target fixed within phase; not arbitrary object-tracking or optimal expert',
        phase_advance='both compatible teacher branches: measured native EE position <=5mm and orientation <=3deg; gripper endpoint command with measured fingers logged, not a grasp certificate; controls retain old rule',
        fallback='teacher horizon / planning failure / script finished -> L0 in remaining budget; unexpected errors censored',
        selected='preselected developer checkpoint, first policy seed; not a holdout or trained adapter score',
        stop_gates='Stop before candidates if full base/sham trajectory differs; exact verified held fork required',
        support_run=str(SUPPORT),support_result_sha256=digest(SUPPORT/'contact_reanalysis/result.json'),
        support_fork_sha256=digest(SUPPORT/'hold_native/fork.json'),
        native_held_run=str(NATIVE_HELD),native_held_audit_sha256=digest(NATIVE_HELD/'audit.json'),
        intervention='compatible and held_compatible differ only in grasp entry; held entry retains closure and skips repeated approach',
        compiler_cache_source=str(CACHE_SOURCE),compiler_cache_protocol='private copied same-policy cache; fresh inference, RNG and observations; no action replay',
        protocol=str(DEFAULT_PROTOCOL),protocol_sha256=digest(DEFAULT_PROTOCOL),source=str(source))
    if args.dry_run:print(json.dumps(job,ensure_ascii=False,indent=2));return
    run=new_run('gate',group=('held_bounded_continuations',point['task']));print(run,flush=True)
    files=[Path(__file__),ROOT/'cx_tta/held_teacher.py',ROOT/'cx_tta/compatible_teacher.py',ROOT/'cx_tta/constraint_teacher.py',ROOT/'cx_tta/teacher_projection.py',ROOT/'cx_tta/native_teacher.py',
           ROOT/'cx_tta/collect_recovery.py',ROOT/'cx_tta/evaluate_b2.py',ROOT/'cx_tta/recovery_contract.py',
           runtime/'envs/_base_task.py',runtime/'envs'/f"{point['task']}.py"]
    write_json(run/'manifest.json',dict(**job,code_sha256={str(p):digest(p) for p in files}))
    snapshot=run/'source_snapshot';snapshot.mkdir()
    source_entries=[]
    for i,path in enumerate(files):
        dest=snapshot/f'code_{i}_{path.name}';shutil.copy2(path,dest)
        source_entries.append(dict(source=str(path),snapshot=str(dest),sha256=digest(path)))
    write_json(run/'source_snapshot.json',source_entries)
    native_paths=[Path(x['source']) for x in json.loads((CACHE_SOURCE/'native_sources.json').read_text())]
    native_entries=[]
    for i,path in enumerate(native_paths):
        dest=snapshot/f'native_{i}_{path.name}';shutil.copy2(path,dest)
        native_entries.append(dict(source=str(path),snapshot=str(dest),sha256=digest(path)))
    write_json(run/'native_sources.json',native_entries)
    cache_audit=json.loads((CACHE_SOURCE/'audit.json').read_text())
    if not cache_audit['passed']:raise ValueError('Cache source audit failed')
    cache_entries=[]
    for name in ('torchinductor_cache_dir','triton_cache_dir'):
        old=CACHE_SOURCE/'cache'/name;dest=run/'cache'/name
        shutil.copytree(old,dest)
        for path in sorted(old.rglob('*')):
            if path.is_file():
                copied=dest/path.relative_to(old);sha=digest(path)
                if digest(copied)!=sha:raise RuntimeError('Private compiler cache copy differs')
                cache_entries.append(dict(source=str(path),copy=str(copied),sha256=sha,bytes=path.stat().st_size))
    write_json(run/'compiler_cache_provenance.json',dict(source=str(CACHE_SOURCE),source_audit_sha256=digest(CACHE_SOURCE/'audit.json'),entries=cache_entries))
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
            if state_snapshot(active)!=support_fork['diagnostics']['physical'] or signature(obs)!=support_fork['observation']:
                raise RuntimeError('Verified held fork changed')
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
            caps=np.zeros(14,np.float32) if branch=='sham_zero' else CAPS if branch in ('compatible','held_compatible') else None
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
                on_query=on_query,on_action=on_action,horizon=None,caps=caps)
            teacher=None;reason='base';fallback=False
            prefix_source=None
            if branch!='base':
                def on_measurement(record):
                    with (folder/'measurements.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
                teacher=(CompatibleTeacher(active,executor,config['save_freq'],on_plan,on_attempt,on_measurement=on_measurement) if branch in ('compatible','held_compatible') else ReplanningTeacher(active,executor,config['save_freq'],on_plan,on_attempt))
                if branch=='held_compatible':
                    enable_held_continuation(active,support_fork['object_name'],'right',lambda entry:write_json(folder/'stage_entry.json',entry))
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
