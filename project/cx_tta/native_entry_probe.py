"""One teacher-prefix intervention then live official VLA continuation.

This is a privileged target-feasibility diagnostic, not an adapter evaluation.
Existing teacher commands are fixed, so repeated planning is not a confound.
"""
import argparse
import importlib
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from .audit import digest
from .paths import ROOT,new_run,write_json,replace_json,output_path
from .native_history_cache import read,arrays,verify,require,ACCEPTED
from .native_target_qualification import CAPS,arm_direction_projection

CONDITIONS=('L0','teacher_prefix_raw','teacher_prefix_clip','teacher_prefix_arm_direction')
TASKS=('place_shoe','turn_switch')


def select_entry(rows,task):
    eligible=[x for x in rows if x['task']==task and x['target_alignment_and_native_domain_passed']]
    require(bool(eligible),'No aligned teacher entry')
    return min(eligible,key=lambda x:(-x['new_base_metrics']['fully_inside_caps_actions']/
        x['new_base_metrics']['valid_actions'],x['seed'],x['native_count']))


def first_actions(base,target,valid,condition):
    require(condition in CONDITIONS,'Unknown teacher diagnostic condition')
    require(base.shape==target.shape==(32,14) and valid.shape==(32,),'Native block shapes')
    require(base.dtype==target.dtype==np.float32 and valid.dtype==bool,'Native block types')
    require(np.isfinite(base).all() and np.isfinite(target[valid]).all(),'Nonfinite valid command')
    if condition=='L0':return base.copy()
    desired=base.copy();desired[valid]=target[valid]
    if condition=='teacher_prefix_raw':return desired
    action=(base+np.clip(desired-base,-CAPS,CAPS) if condition=='teacher_prefix_clip'
            else arm_direction_projection(base,desired))
    action[~valid]=base[~valid]  # Preserve signed-zero and ignored teacher tail.
    require(not np.any(np.abs(action-base)>CAPS+2e-6),'Teacher projection exceeds original caps')
    return action


def decision(episodes,repeats,smoke):
    if smoke:return dict(qualified=False,reason='Interface was deliberately truncated; not complete benefit evidence')
    require(len(episodes)==len(CONDITIONS)*repeats,'All selected branches required')
    require(all(e['complete'] for e in episodes),'Incomplete continuation cannot qualify labels')
    by={(e['condition'],e['repeat']):e for e in episodes}
    require(len(by)==len(episodes),'Duplicate branch result')
    for condition in CONDITIONS:
        require(all(by[condition,r]['trajectory_sha256']==by[condition,0]['trajectory_sha256']
            and by[condition,r]['success']==by[condition,0]['success'] for r in range(repeats)),
            'Within-process repeats disagree; no recovery certificate')
    qualified={condition:all(not by['L0',r]['success'] and by[condition,r]['success'] for r in range(repeats))
        for condition in ('teacher_prefix_clip','teacher_prefix_arm_direction')}
    return dict(qualified=any(qualified.values()),by_bounded_condition=qualified,
        reason='Repeated bounded rescue against same-process L0' if any(qualified.values())
            else 'No repeated bounded rescue at the selected prefix; do not promote this target',
        scope='One selected training state; not adapter improvement, all targets or population success')


def prepare(cache,target_audit):
    cache=output_path(cache);target_audit=output_path(target_audit)
    require(read(cache/'audit.json')['passed'],'Audited native cache required')
    qa=read(target_audit/'result.json');verify(qa['source_sha256'])
    require(qa['cache_run']==str(cache) and qa['passed'],'Target/cache mismatch')
    cp=read(cache/'protocol.json');verify(cp['source_sha256'])
    points=[];sources={}
    def add(path):sources[str(path)]=digest(Path(path))
    for task in TASKS:
        picked=select_entry(qa['rows'],task);row=cp['rows'][picked['cache_row']]
        source=Path(read(ACCEPTED/'next_experiments.json')['native_source_runs'][task])
        target=row['teacher_entry'];require(row['split']=='train_candidate','Training source only')
        manifest=read(source/'manifest.json')
        require(manifest['role']=='train' and row['seed'] in manifest['seeds'],'Native TRAIN scene required')
        require(row['seed'] not in (421007 if task=='place_shoe' else 421004,),'Whole-scene holdout leaked')
        points.append(dict(task=task,seed=row['seed'],native_count=row['native_count'],chunk=row['chunk'],
            cache_row=picked['cache_row'],source=str(source),target=target['target'],
            target_sha256=target['target_sha256'],valid_teacher_prefix=picked['new_base_metrics']['valid_actions'],
            selection='Largest static supported-step fraction within each task; seed/count tie break. No closed-loop outcomes searched.',
            source_signature=row['signature']))
        for name in ('manifest.json','result.json','episodes.json','resolved_config.json','runtime_hashes.json','provenance.json'):
            add(source/name)
        add(source/'traces'/f"r0_seed{row['seed']}.jsonl");add(Path(target['target']))
        add(source/'observations'/f"seed_{row['seed']}"/'index.json')
        for chunk in range(row['chunk']+1):
            prefix=source/'observations'/f"seed_{row['seed']}"/f'chunk_{chunk:03d}'
            for suffix in ('.npz','.base.npz','.rng.json'):add(prefix.with_suffix(suffix))
    for path in (cache/'protocol.json',cache/'audit.json',cache/'inputs.npz',cache/'recorded_bases.npz',
        target_audit/'result.json',Path(cp['tokenizer']),Path(__file__),ROOT/'scripts/probe_native_entry.sh',
        ROOT/'tests/test_native_entry_probe.py',ROOT/'cx_tta/native_target_qualification.py',
        ROOT/'cx_tta/collect_recovery.py',ROOT/'cx_tta/recovery_contract.py',
        ROOT/'cx_tta/native_entry_storage.py',ROOT/'tests/test_native_entry_storage.py',
        Path('/data/cj/robo_pi/250010033/qfw/robotwin/setup_user_nvidia_graphics.env'),
        ROOT/'cx_tta/probe_action_replay.py',ROOT/'cx_tta/evaluate_structured_low_storage.py'):
        add(path)
    protocol=dict(stage='S2n_native_entry_target_feasibility',status='frozen_before_physics',
        cache_run=str(cache),target_audit=str(target_audit),points=points,conditions=list(CONDITIONS),
        repeats=2,condition_order='Forward then reverse',sampler_seed=920000,tokenizer=cp['tokenizer'],
        identity=cp['identity'],caps=CAPS.tolist(),scene_timeout_s=1800,source_sha256=sources,
        correction='Only first query valid teacher prefix<=32 commands. All later queries are fresh VLA at reached states.',
        no_teacher_replanning=True,no_teacher_geometry_in_deployable_actor=True,
        first_live_base='Shared warmed compiled policy; queried once per branch, same initial physical/RGB/RNG and base verified.',
        recorded_source_base='Used only to replay prefix and select static supported states; not substituted for a fresh branch query.',
        label_certificate='If repeated rescue qualifies, retain actual LIVE first-query base/target/valid and bounded commands together. '
            'Do not apply that certificate automatically to the old recorded-base candidate; its fresh-base source must be shared by both later training arms.',
        storage='No RGB/video/full prefix/backbone copy; numeric blocks and state/RNG hashes, source runtime immutable sharing, private scratch removed at exit.',
        excluded='Known shoe320006 projection gate and100380-100399 development evaluations are not rerun.',
        interpretation='Privileged prefix-target feasibility. No learned adapter/router/TTA, no population or task-generalization claim.')
    out=new_run('native_entry_bounded',group=('reports','protocols'));write_json(out/'protocol.json',protocol)
    print('PROTOCOL',out/'protocol.json',flush=True);return out/'protocol.json'


def preflight(protocol,task):
    p=read(protocol);verify(p['source_sha256'])
    require(p['stage']=='S2n_native_entry_target_feasibility' and p['conditions']==list(CONDITIONS),'Sealed diagnostic contract')
    require(p['repeats']==2 and p['caps']==CAPS.tolist(),'Budget/repetition/caps changed')
    matches=[x for x in p['points'] if x['task']==task];require(len(matches)==1,'Exactly one fixed point required')
    point=matches[0]
    from .probe_action_replay import load_source,source_runtime
    manifest,episode,trace=load_source(Path(point['source']),point['seed'])
    require(not episode['success'],'Recovery source must be an originally failed TRAIN scene')
    require(sum(x['executed'] for x in trace[:point['chunk']])==point['native_count'],'Fork action index changed')
    source_runtime(Path(point['source']))
    require(digest(Path(point['target']))==point['target_sha256'],'Teacher target changed')
    return p,point,manifest,episode,trace


def run_probe(protocol,task_name,gpu,smoke):
    p,point,manifest,episode,source_trace=preflight(protocol,task_name)
    os.environ['CUDA_VISIBLE_DEVICES']=gpu;os.environ['JAX_PLATFORMS']='cpu'
    from .native_entry_storage import NativeEntryStorage
    from .probe_action_replay import source_scene_index,compare_physical
    from .collect_recovery import signature,infer
    from .evaluate_b2 import seed_all
    from .eval_diagnostics import state_snapshot,array_hash,json_hash
    from .recovery_contract import rng_capture,rng_restore,rng_fingerprint,execute_chunk,terminal_label
    from .cache_expert import create_base_policy
    import atexit
    import hashlib
    import torch
    require(torch.cuda.is_available(),'CUDA device required')
    source=Path(point['source']);run=new_run('entry_smoke' if smoke else 'entry_paired',
        group=('diagnostics','native_entry_targets',task_name));print('RUN',run,flush=True)
    storage=NativeEntryStorage(runtime_source=source/'robotwin');atexit.register(storage.cleanup)
    try:
        runtime=storage.prepare_runtime(run);storage.configure_runtime(run,runtime,Path(p['tokenizer']))
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),phase='runtime_preparation',
            completed_branches=0,model_created=False,physics_started=False));raise
    config=read(source/'resolved_config.json');config.update(save_path=str(run/'runtime_data'),
        cx_runtime=str(runtime),save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
    write_json(run/'manifest.json',dict(protocol=str(protocol),protocol_sha256=digest(protocol),point=point,
        smoke=smoke,conditions=list(CONDITIONS),repeats=1 if smoke else 2,no_video=True,
        scope='Privileged native-entry target diagnostic; not adapter evaluation'))
    write_json(run/'resolved_config.json',config)
    target=arrays(point['target']);task=None;episodes=[];first_reference=None
    warm_base=None;pre_rng=None;sampler=None;began=time.monotonic()
    def timeout(*_):raise TimeoutError('Native-entry scene/compile exceeded1800s')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['scene_timeout_s'])
    try:
        seed_all(0);policy=create_base_policy();policy._model.eval().requires_grad_(False);sampler=policy._sample_actions
        require(hasattr(sampler,'_torchdynamo_orig_callable') and not torch._dynamo.config.disable,'Compiled policy required')
        cls=getattr(importlib.import_module('envs.'+task_name),task_name)
        for repeat in range(1 if smoke else p['repeats']):
            order=CONDITIONS if repeat==0 else tuple(reversed(CONDITIONS))
            for condition in order:
                signal.alarm(p['scene_timeout_s']);task=cls();seed_all(point['seed'])
                task.setup_demo(now_ep_num=source_scene_index(manifest,point['seed']),seed=point['seed'],
                    **dict(config,need_plan=True,is_test=True))
                task.set_instruction(instruction=episode['instruction']);obs=task.get_obs()
                for i,row in enumerate(source_trace[:point['chunk']]):
                    require(compare_physical(row['before'],state_snapshot(task))['matched']
                        and signature(obs)==row['observation_components'],'Source replay before-query mismatch')
                    base=arrays(source/'observations'/f"seed_{point['seed']}"/f'chunk_{i:03d}.base.npz')['base_action']
                    obs,executed=execute_chunk(task,base)
                    require(executed==row['executed'] and compare_physical(row['after'],state_snapshot(task))['matched'],
                        'Source replay after-query mismatch')
                require(int(task.take_action_cnt)==point['native_count'] and signature(obs)==point['source_signature'],
                    'Actual fork count/RGB differs from selected source')
                require(compare_physical(source_trace[point['chunk']]['before'],state_snapshot(task))['matched'],
                    'Actual physical fork differs from source')
                if pre_rng is None:
                    replace_json(run/'progress.json',dict(phase='warming_shared_compiled_policy',completed=0))
                    infer(policy,obs,episode['instruction']);infer(policy,obs,episode['instruction'])
                    seed_all(p['sampler_seed']);pre_rng=rng_capture()
                    warm_base=infer(policy,obs,episode['instruction']);post=rng_capture();rng_restore(pre_rng)
                    second=infer(policy,obs,episode['instruction'])
                    require(np.array_equal(warm_base,second) and rng_fingerprint(post)==rng_fingerprint(rng_capture()),
                        'Steady first-query probe mismatch')
                    write_json(run/'steady_probe.json',dict(passed=True,base_sha256=array_hash(warm_base),
                        rng_before=rng_fingerprint(pre_rng),rng_after=rng_fingerprint(post)))
                rng_restore(pre_rng);rows=[];bases=[];actions=[];query=0;traj=hashlib.sha256()
                prefix_success=None;trace_path=run/f'r{repeat}_{condition}.jsonl'
                with trace_path.open('x') as trace_file:
                    while not task.eval_success and task.take_action_cnt<task.step_lim:
                        if smoke and query>=1:break
                        before=state_snapshot(task);sig=signature(obs);pre=rng_fingerprint(rng_capture())
                        base=infer(policy,obs,episode['instruction']);post=rng_fingerprint(rng_capture())
                        action=first_actions(base,target['target'],target['valid'],condition) if query==0 else base.copy()
                        if query==0:
                            paired=dict(physical=json_hash(before),signature=sig,rng_before=pre,rng_after=post,
                                base_sha256=array_hash(base))
                            require(np.array_equal(base,warm_base),'Warmed first-query base differs')
                            if first_reference is None:first_reference=paired
                            else:require(paired==first_reference,'Initial branch pairing mismatch')
                        obs,executed=execute_chunk(task,action);after=state_snapshot(task)
                        after_physics=rng_fingerprint(rng_capture())
                        if query==0:prefix_success=bool(task.eval_success)
                        record=dict(query=query,native_before=before['native_count'] if 'native_count' in before else point['native_count']+32*query,
                            native_after=int(task.take_action_cnt),executed=executed,observation_signature=sig,
                            after_signature=signature(obs),before_physical_sha256=json_hash(before),
                            after_physical_sha256=json_hash(after),base_sha256=array_hash(base),
                            actions_sha256=array_hash(action),rng_before=pre,rng_after=post,
                            rng_after_physics=after_physics)
                        # Candidate recording and passive observation reads must not draw RNG.
                        require(after_physics==rng_fingerprint(rng_capture()),'Passive recording altered RNG')
                        encoded=__import__('json').dumps(record,sort_keys=True,separators=(',',':'))
                        trace_file.write(encoded+'\n');traj.update(encoded.encode());bases.append(base);actions.append(action)
                        query+=1
                        require(policy._sample_actions is sampler and not torch._dynamo.config.disable,'Sampler changed')
                label=terminal_label(int(task.take_action_cnt),int(task.step_lim),bool(task.eval_success),interrupted=smoke)
                result=dict(condition=condition,repeat=repeat,**label,prefix_native_success=prefix_success,
                    executed_actions=int(task.take_action_cnt),continuation_actions=int(task.take_action_cnt)-point['native_count'],
                    queries=query,trajectory_sha256=traj.hexdigest(),first_query_paired=True)
                np.savez_compressed(run/f'r{repeat}_{condition}.npz',base=np.stack(bases),actions=np.stack(actions))
                episodes.append(result);replace_json(run/'episodes.json',episodes)
                print(condition,repeat,'complete',result['complete'],'success',result['success'],flush=True)
                task.close_env();task=None
        cert=decision(episodes,p['repeats'],smoke)
        write_json(run/'result.json',dict(status='complete',point=point,episodes=episodes,decision=cert,
            first_query_pairing_passed=True,smoke=smoke,elapsed_s=time.monotonic()-began,
            learned_adapter_used=False,new_recovery_training_allowed=cert['qualified']))
        # Fresh base and actions must stay attached to a recovery qualification.
        np.savez_compressed(run/'first_live_base_target.npz',base=warm_base,target=target['target'],valid=target['valid'],
            source_native_counts=target['source_native_counts'],
            coordinate_clip=first_actions(warm_base,target['target'],target['valid'],'teacher_prefix_clip'),
            arm_direction=first_actions(warm_base,target['target'],target['valid'],'teacher_prefix_arm_direction'))
        write_json(run/'source_runtime_audit.json',storage.verify_source())
        print('QUALIFIED',cert['qualified'],'interface_only',smoke,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed_branches=len(episodes)));raise
    finally:
        signal.alarm(0)
        if task is not None and hasattr(task,'scene'):task.close_env()


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--cache-run',type=Path);parser.add_argument('--target-audit',type=Path)
    parser.add_argument('--protocol',type=Path);parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--gpu',default='0');parser.add_argument('--smoke',action='store_true');parser.add_argument('--dry-run',action='store_true')
    a=parser.parse_args()
    if not a.gpu.isdigit():parser.error('Numeric GPU required')
    if a.prepare_only:
        if a.cache_run is None or a.target_audit is None:parser.error('Cache and target audit required')
        prepare(a.cache_run.resolve(),a.target_audit.resolve());return
    if a.protocol is None or a.task is None:parser.error('Sealed --protocol and --task required')
    protocol=a.protocol.resolve();p,point,*_=preflight(protocol,a.task)
    if a.dry_run:
        print(__import__('json').dumps(dict(point=point,conditions=p['conditions'],
            repeats=1 if a.smoke else 2,queries_per_branch=1 if a.smoke else 'Full original native budget',GPU_started=False),indent=2));return
    run_probe(protocol,a.task,a.gpu,a.smoke)


if __name__=='__main__':main()
