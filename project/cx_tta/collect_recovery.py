"""One bounded intervention then full official L0 continuation; paired terminal probe."""
import argparse
import copy
import hashlib
import importlib
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from .audit import digest
from .deploy import unpack_observation,POLICY_CAMERA_KEYS
from .eval_diagnostics import array_hash,json_hash,state_snapshot
from .evaluate_b2 import configure_runtime,resolve_tokenizer,seed_all
from .paths import ROOT,new_run,write_json,replace_json
from .probe_action_replay import load_source,source_runtime,source_scene_index,prime_warp_cache,compare_physical
from .recovery_contract import rng_capture,rng_restore,rng_fingerprint,execute_chunk,terminal_label,comparison

DEFAULT_PROTOCOL=ROOT/'outputs/reports/protocols/recovery_protocol_20260930_154709_684366/protocol.json'


def signature(observation):
    return dict(images={k:array_hash(observation['observation'][k]['rgb']) for k in ('head_camera','right_camera','left_camera')},
        command_state=array_hash(observation['joint_action']['vector']))


def infer(policy,observation,instruction):
    images,state=unpack_observation(observation)
    request=dict(state=state.copy(),prompt=instruction,images={k:im.transpose(2,0,1).copy() for k,im in zip(POLICY_CAMERA_KEYS,images)})
    base=np.asarray(policy.infer(request)['actions'],np.float32).copy()
    if base.shape!=(32,14) or not np.isfinite(base).all():raise ValueError('Invalid base action')
    return base


def save_observation(path,obs):
    images,state=unpack_observation(obs)
    np.savez_compressed(path,head_rgb=images[0],right_rgb=images[1],left_rgb=images[2],
        command_state_raw=np.asarray(obs['joint_action']['vector']),command_state=state)


def rng_json(state):
    name,keys,pos,gauss,value=state['numpy']
    return dict(python=state['python'],numpy=[name,keys.tolist(),pos,gauss,value],
        torch_cpu=state['torch_cpu'].tolist(),torch_cuda=[s.tolist() for s in state['torch_cuda']],sha256=rng_fingerprint(state))


def preflight(args):
    protocol=json.loads(args.protocol.read_text());points=protocol['points']
    if args.point is None:
        choices=[(i,p) for i,p in enumerate(points) if not p['source_success_stratum']]
        selected=min(choices,key=lambda ip:hashlib.sha256(f'{ip[1]["task"]}:{ip[1]["seed"]}'.encode()).hexdigest())[0]
    else:selected=args.point
    if not 0<=selected<len(points):raise ValueError('Invalid point index')
    point=points[selected];source=Path(point['source'])
    for path,key in [(source/'manifest.json','source_manifest_sha256'),(source/'resolved_config.json','source_config_sha256'),
                     (source/'traces'/f'r0_seed{point["seed"]}.jsonl','source_trace_sha256'),(Path(point['checkpoint']),'checkpoint_sha256')]:
        if digest(path)!=point[key]:raise ValueError('Changed planned source: '+str(path))
    model=protocol['candidate_model']
    if digest(Path(model['path']))!=model['sha256'] or digest(Path(model['preprocessing']))!=model['preprocessing_sha256']:
        raise ValueError('Changed fixed proposal model')
    manifest,episode,rows=load_source(source,point['seed']);runtime=source_runtime(source)
    split_path=Path(protocol['split_source'])
    if digest(split_path)!=protocol['split_sha256']:raise ValueError('Changed original split')
    allowed={(r['task'],r['seed']) for r in json.loads(split_path.read_text())['scenes'] if r['split']=='train'}
    if (point['task'],point['seed']) not in allowed:raise ValueError('Not an original-training scene')
    import yaml
    limit_path=runtime/'task_config/_eval_step_limit.yml'
    limit=int(yaml.safe_load(limit_path.read_text()).get(point['task'],1000))
    if point['prefix_action_count']!=sum(r['executed'] for r in rows[:point['chunk']]) or not point['prefix_action_count']<limit:
        raise ValueError('Invalid fork budget')
    seeds=protocol['continuation_policy_seeds'][:2] if args.phase=='gate' else protocol['continuation_policy_seeds']
    if args.phase=='pilot':
        if args.gate_result is None:raise ValueError('Pilot requires a passing gate result')
        gate=json.loads(args.gate_result.read_text());gm=json.loads((args.gate_result.parent/'manifest.json').read_text())
        if gate['status']!='complete' or not gate['gate_passed'] or gm['phase']!='gate' or gm['protocol_sha256']!=digest(args.protocol):
            raise ValueError('Gate is not valid for this protocol')
        for rel,sha in gm['code_hashes'].items():
            if digest(ROOT/rel)!=sha:raise ValueError('Gate code changed; validate new version before pilot')
    if args.cache_from is not None:
        args.cache_from=args.cache_from.resolve()
        if not args.cache_from.is_relative_to(ROOT/'outputs/recovery_continuations'):raise ValueError('Cache source must be a recovery run in this project')
        cached=json.loads((args.cache_from/'manifest.json').read_text())
        if cached['protocol_sha256']!=digest(args.protocol):raise ValueError('Cache source protocol differs')
        for rel,sha in cached['code_hashes'].items():
            if rel!='cx_tta/collect_recovery.py' and digest(ROOT/rel)!=sha:raise ValueError('Cache runtime dependency changed: '+rel)
    return protocol,point,manifest,episode,rows,runtime,dict(point_index=selected,task=point['task'],scene_seed=point['seed'],
        chunk=point['chunk'],prefix_actions=point['prefix_action_count'],native_limit=limit,remaining_actions=limit-point['prefix_action_count'],
        limit_file_sha256=digest(limit_path),policy_seeds=seeds,continuations=len(seeds)*3+(len(seeds) if args.phase=='gate' else 1),
        phase=args.phase,protocol_sha256=digest(args.protocol),no_video=True,
        compile_cache_source=str(args.cache_from) if args.cache_from is not None else None)


def copy_compile_cache(source,run):
    """Copy, never share/mutate, prior local compiler artifacts after an interruption."""
    files={}
    for name in ('torchinductor_cache_dir','triton_cache_dir'):
        origin=source/'cache'/name
        if not origin.is_dir():raise ValueError('Missing compiler cache: '+str(origin))
        for path in origin.rglob('*'):
            if path.is_symlink():raise ValueError('Compiler cache symlink not allowed')
            if path.is_file():files[str(path.relative_to(source))]=digest(path)
        shutil.copytree(origin,run/'cache'/name)
    if not files:raise ValueError('Compiler cache is empty')
    if any(digest(run/rel)!=sha for rel,sha in files.items()):raise ValueError('Compiler cache copy mismatch')
    write_json(run/'compile_cache_copy.json',dict(source=str(source),source_manifest_sha256=digest(source/'manifest.json'),files=files,
        scope='Compiler artifacts only. All replay, query noninterference and full branch checks execute again.'))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL)
    parser.add_argument('--phase',choices=('gate','pilot'),default='gate');parser.add_argument('--point',type=int)
    parser.add_argument('--gpu',default='0');parser.add_argument('--gate-result',type=Path);parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--cache-from',type=Path,help='Copy compiler cache from a compatible local recovery run; never reuse outcomes')
    args=parser.parse_args();args.protocol=args.protocol.resolve()
    if args.gate_result:args.gate_result=args.gate_result.resolve()
    if not args.gpu.isdigit():parser.error('Numeric GPU index required')
    protocol,point,manifest,episode,rows,runtime,job=preflight(args)
    if args.dry_run:print(json.dumps(dict(status='dry_run_passed',**job),indent=2));return
    run=new_run(args.phase,group=('recovery_continuations',point['task']));print('RUN '+str(run),flush=True)
    code_paths=['cx_tta/'+name+'.py' for name in ('collect_recovery','recovery_contract','recovery_adapter','cache_joint_prefix','direction_adapter','probe_action_replay','evaluate_b2','deploy')]+['scripts/collect_recovery.sh']
    write_json(run/'manifest.json',dict(**job,protocol=str(args.protocol),point=point,candidate_model=protocol['candidate_model'],
        code_hashes={p:digest(ROOT/p) for p in code_paths},scope='Train-scene paired terminal-continuation feasibility, not final success-rate evaluation or TTA. Only first 32-action chunk corrected. Fresh policy RNG at fork; historical outcome only selects stratum.'))
    active=None;reports=[];pairs=[]
    try:
        prime_warp_cache(Path(point['source']),run)
        if args.cache_from is not None:copy_compile_cache(args.cache_from,run)
        configure_runtime(run,runtime,resolve_tokenizer())
        import torch
        from .cache_expert import create_base_policy
        from .recovery_adapter import RecoveryAdapter
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
        torch.set_num_threads(1)
        adapter=RecoveryAdapter(protocol['candidate_model']['path'],protocol['candidate_model']['preprocessing'])
        policy=create_base_policy()
        if torch._dynamo.config.disable or not hasattr(policy._model.sample_actions,'_torchdynamo_orig_callable'):
            raise RuntimeError('Official compiled sample_actions required')
        config=json.loads((Path(point['source'])/'resolved_config.json').read_text())
        config.update(save_path=str(run/'runtime_data'),cx_runtime=str(runtime),save_data=False,eval_video_log=False,eval_video_save_dir=None)
        task_cls=getattr(importlib.import_module('envs.'+point['task']),point['task']);scene_index=source_scene_index(manifest,point['seed'])

        def replay():
            nonlocal active
            active=task_cls();seed_all(point['seed'])
            active.setup_demo(now_ep_num=scene_index,seed=point['seed'],**dict(config,need_plan=True,is_test=True))
            active.set_instruction(instruction=point['instruction']);obs=active.get_obs();previous=None
            for chunk in range(point['chunk']):
                if chunk==point['chunk']-1:previous=copy.deepcopy(obs)
                expected=rows[chunk]
                if signature(obs)!=expected['observation_components']:raise RuntimeError('Replay prefix image/command mismatch')
                for action in expected['base_executed']:
                    before=active.take_action_cnt;active.take_action(np.asarray(action,np.float32));obs=active.get_obs()
                    if active.take_action_cnt!=before+1:raise RuntimeError('Replay prefix stopped early')
            physical=compare_physical(rows[point['chunk']]['before'],state_snapshot(active),1e-5)
            if not physical['matched'] or signature(obs)!=point['observation_signature']:
                raise RuntimeError('Fork replay mismatch: '+json.dumps(physical))
            if active.take_action_cnt!=job['prefix_actions'] or active.step_lim!=job['native_limit'] or active.eval_success:
                raise RuntimeError('Fork budget or terminal state mismatch')
            return obs,previous,physical

        for repeat,policy_seed in enumerate(job['policy_seeds']):
            obs,previous,physical=replay();policy.reset()
            # Warm compiled inference before measuring auxiliary noninterference.
            seed_all(policy_seed);infer(policy,obs,point['instruction'])
            seed_all(policy_seed);pre=rng_capture();started=time.perf_counter()
            base_control=infer(policy,obs,point['instruction']);post=rng_capture();base_seconds=time.perf_counter()-started
            rng_restore(pre)
            current,feature_meta=adapter.observable_features(policy,obs,point['instruction'])
            past,past_meta=(None,None) if previous is None else adapter.observable_features(policy,previous,point['instruction'])
            if rng_fingerprint(rng_capture())!=rng_fingerprint(pre):raise RuntimeError('Auxiliary changed RNG')
            base=infer(policy,obs,point['instruction'])
            if not np.array_equal(base,base_control) or rng_fingerprint(rng_capture())!=rng_fingerprint(post):
                raise RuntimeError('Auxiliary changed compiled base actions or post-query RNG')
            delta,context=adapter.propose(current,past,base)
            if rng_fingerprint(rng_capture())!=rng_fingerprint(post):raise RuntimeError('Adapter changed RNG')
            repeat_dir=run/f'policy_seed_{policy_seed}';repeat_dir.mkdir()
            save_observation(repeat_dir/'fork.npz',obs)
            if previous is not None:save_observation(repeat_dir/'previous_query.npz',previous)
            np.savez_compressed(repeat_dir/'proposal.npz',base=base,delta=delta,context=context,current_features=current,
                previous_features=past if past is not None else np.zeros_like(current))
            write_json(repeat_dir/'rng.json',dict(before=rng_json(pre),after_common_query=rng_json(post)))
            write_json(repeat_dir/'interface.json',dict(compiled_base_exact=True,rng_exact=True,physical=physical,
                feature=feature_meta,history_feature=past_meta,history_source_chunk=point['chunk']-1 if previous is not None else None,
                history_gap_actions=32 if previous is not None else None,base_seconds=base_seconds,max_abs_delta=float(np.abs(delta).max()),
                interpretation='History is previous real policy query, not future or GT. Expert training had irregular cached history gaps; deployment history generalization unproven.'))
            branch_names=['base','sham','plus','minus'] if args.phase=='gate' or repeat==0 else ['base','plus','minus']
            local={}
            for branch_index,name in enumerate(branch_names):
                if branch_index:
                    obs,_,_=replay()
                branch_dir=repeat_dir/name;branch_dir.mkdir();policy.reset();rng_restore(post)
                branch_start=rng_fingerprint(rng_capture());trajectory=hashlib.sha256();chunk=0;updates=0;start=time.perf_counter()
                first=base if name in ('base','sham') else base+(delta if name=='plus' else -delta)
                np.savez_compressed(branch_dir/'first_chunk.npz',actions=first,base=base,delta=first-base)
                with (branch_dir/'trace.jsonl').open('x') as stream:
                    while active.take_action_cnt<active.step_lim and not active.eval_success:
                        before=state_snapshot(active);obs_sig=signature(obs);save_observation(branch_dir/f'observation_{chunk:04d}.npz',obs)
                        before_rng=rng_fingerprint(rng_capture())
                        if chunk==0:actions=first;query_base=base
                        else:query_base=infer(policy,obs,point['instruction']);actions=query_base
                        per_action=[]
                        def observe(i,action,following,advanced):
                            per_action.append(dict(attempt=i,count=int(active.take_action_cnt),advanced=advanced,
                                observation=signature(following),success=bool(active.eval_success)))
                        obs,executed=execute_chunk(active,actions,observe);updates+=32
                        if executed==0:raise RuntimeError('Nonterminal chunk made no progress')
                        record=dict(chunk=chunk,observation=obs_sig,before=before,after=state_snapshot(active),base_sha256=array_hash(query_base),
                            actions=actions.tolist(),executed=executed,observation_updates=32,per_action=per_action,
                            before_query_rng=before_rng,after_chunk_rng=rng_fingerprint(rng_capture()))
                        trajectory.update(json_hash(record).encode());stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();chunk+=1
                outcome=terminal_label(int(active.take_action_cnt),int(active.step_lim),bool(active.eval_success))
                if not outcome['complete']:raise RuntimeError('Unfinished branch incorrectly exited loop')
                save_observation(branch_dir/'terminal.npz',obs)
                record=dict(policy_seed=policy_seed,branch=name,**outcome,action_count=int(active.take_action_cnt),
                    prefix_actions=job['prefix_actions'],continuation_actions=int(active.take_action_cnt)-job['prefix_actions'],
                    native_limit=int(active.step_lim),chunks=chunk,observation_updates=updates,start_rng=branch_start,
                    trajectory_sha256=trajectory.hexdigest(),trace_sha256=digest(branch_dir/'trace.jsonl'),seconds=time.perf_counter()-start)
                write_json(branch_dir/'result.json',record);local[name]=record;reports.append(record)
                active.close_env(clear_cache=False);active=None
                print('BRANCH_COMPLETE',policy_seed,name,outcome['success'],record['continuation_actions'],flush=True)
                if name=='sham':
                    sham=(local['base']['trajectory_sha256']==record['trajectory_sha256'] and local['base']['success']==record['success'] and local['base']['action_count']==record['action_count'])
                    write_json(repeat_dir/'sham_check.json',dict(passed=sham,base=local['base']['trajectory_sha256'],sham=record['trajectory_sha256']))
                    if not sham:raise RuntimeError('Base/sham mismatch; stop before candidate comparisons')
                replace_json(run/'progress.json',dict(status='running',reports=reports,pairs=pairs))
            sham_passed='sham' not in local or local['base']['trajectory_sha256']==local['sham']['trajectory_sha256']
            for name in ('plus','minus'):pairs.append(dict(policy_seed=policy_seed,branch=name,comparison=comparison(local['base'],local[name],sham_passed)))
        passed=all(r['complete'] for r in reports) and len(reports)==job['continuations']
        write_json(run/'result.json',dict(status='complete',gate_passed=passed,reports=reports,pairs=pairs,
            scope='Paired feasibility only; same checkpoint, multiple policy seeds. Not independent scene success rate or reliable preference labels.'))
        print('COMPLETE',str(run),'gate_passed',passed,'pairs',json.dumps(pairs),flush=True)
    except BaseException:
        failure=dict(status='failed_or_interrupted',gate_passed=False,success=None,reports=reports,pairs=pairs,traceback=traceback.format_exc(),
            partial_action_count=int(active.take_action_cnt) if active is not None and hasattr(active,'take_action_cnt') else None,
            interpretation='Incomplete/invalid comparisons are censored, not task failures.')
        replace_json(run/'result.json',failure);print(failure['traceback'],flush=True);raise
    finally:
        if active is not None and hasattr(active,'scene'):active.close_env(clear_cache=False)


if __name__=='__main__':main()
