"""Inventory all fixed attempts and freeze legal teacher cache inputs."""
import argparse,json
from pathlib import Path
import numpy as np
from .audit import digest
from .cache_recovery import FEATURES,FIT,read,verify_hashes,legal_observation
from .inventory_recovery_labels import signature_file
from .paths import ROOT,new_run,write_json


def prepare(protocol):
    p=read(protocol);batch=read(protocol.parent/'batch_result.json')
    verify_hashes({**p['source_sha256'],**p['code_sha256']})
    context_audit=ROOT/'outputs/feature_caches/recovery_context_20261002_075253_551153/audit.json'
    checked=read(context_audit)
    if not checked['passed'] or checked['schema']!='recovery_context_v2_state_div_pi':raise ValueError('Corrected training convention required')
    for name in ('semantic_preprocess.npz','coarse_preprocess.npz'):
        f=FIT/name
        if digest(f)!=checked['source_sha256'][str(f)]:raise ValueError('Original fixed expert preprocessing changed')
    if batch['status']!='complete' or len(batch['attempts'])!=15:raise ValueError('All fixed attempts must finish first')
    run=new_run('fixed_recovery',group=('feature_caches',));print(run,flush=True)
    targets=run/'executed_targets';targets.mkdir()
    sources=[protocol,protocol.parent/'batch_result.json',context_audit,FIT/'semantic_preprocess.npz',FIT/'coarse_preprocess.npz',
        FEATURES/'manifest.json',FEATURES/'result.json',FEATURES/'feature_audit.json',FEATURES/'checkpoint_audit.json']
    attempts=[];selected=[]
    all_attempts=[dict(kind='switch',point=4,run=p['reuse_gate_run'],exit_code=0,reused=True)]+batch['attempts']+p.get('held_teacher_attempts',[])
    for attempt in all_attempts:
        path=Path(attempt['run']);point=p['points'][attempt['point']]
        sources.append(path/'manifest.json');manifest=read(path/'manifest.json')
        if manifest['point']!=point['index'] or manifest['scene_seed']!=point['seed'] or manifest['source']!=point['source'] or manifest['prefix_actions']!=point['prefix_actions']:
            raise ValueError('Attempt identity mismatch')
        record=dict(point=point['index'],task=point['task'],seed=point['seed'],prefix=point['prefix_actions'],run=str(path),kind=attempt['kind'])
        if (path/'failure.json').exists():
            if not (path/'replay_mismatch.json').exists():raise ValueError('Unresolved interface failure')
            sources.extend([path/'failure.json',path/'replay_mismatch.json'])
            record.update(qualified=False,reason='exact replay mismatch; no teacher labels',replay=read(path/'replay_mismatch.json'))
            attempts.append(record);continue
        sources.append(path/'result.json');result=read(path/'result.json')
        if attempt['kind']=='shoe_support':
            record.update(qualified=False,support_qualified=result['held_qualified'],reason='support assay is not a completed teacher trajectory')
        else:
            q=read(path/'qualification.json');a=read(path/'audit.json');sources.extend([path/'qualification.json',path/'audit.json'])
            if not a['passed'] or a['result_sha256']!=digest(path/'result.json'):raise ValueError('Teacher audit changed')
            verify_hashes(manifest['code_sha256'])
            record.update(qualified=q['qualified'],reason=result['reason'],teacher_actions=result['teacher_actions'])
            if q['qualified']:
                if not result['teacher_success'] or not result['native_terminal'] or not result['plan_success'] or result['reason']!='native_success':raise ValueError('Inconsistent qualification')
                if attempt['kind']=='shoe_teacher':
                    support=Path(manifest['support_run']);sr=read(support/'result.json');sf=read(support/'hold/fork.json');fork=read(path/'fork.json')
                    if not sr['held_qualified'] or q['verified_owner']!=sr['owner'] or fork['diagnostics']['physical']!=sf['diagnostics']['physical'] or fork['observation']!=sf['observation']:raise ValueError('Unverified held entry')
                    stages=[read(path/f'semantic_plan_{i:03d}.json')['stage'] for i in range(result['teacher_plans'])]
                    if stages!=q['stages'] or stages[:5]!=['closure','lift','preplace','place','release'] or stages[-1]!='release':raise ValueError('Unfinished held semantic trajectory')
                    obj=np.asarray(result['final_diagnostics']['physical']['offline_only']['object_poses']['shoe']);quat=obj[3:].copy()
                    if quat[0]<0:quat*=-1
                    if not (np.all(np.abs(obj[:2]-[0,-.08])<[.05,.02]) and np.all(np.abs(quat-[.5,.5,-.5,-.5])<.07) and all(result['final_diagnostics']['grippers_open'])):raise ValueError('Shoe terminal predicate mismatch')
                    sources.extend([support/'audit.json',support/'result.json',support/'hold/fork.json',path/'stage_entry.json',*path.glob('semantic_plan_*.json')])
                selected.append((point,path))
        attempts.append(record)
    rows=[];feature_rows=[];feature_history=[];current_ids=[];history=[]
    for point,source in selected:
        result=read(source/'result.json');fork=read(source/'fork.json')
        trace=[json.loads(x) for x in (source/'trace.jsonl').read_text().splitlines()]
        with np.load(source/'native_actions.npz') as z:actions=z['actions'].copy()
        if actions.shape!=(len(trace),14) or len(trace)!=result['teacher_actions'] or digest(source/'trace.jsonl')!=result['trace_sha256']:
            raise ValueError('Teacher execution evidence mismatch')
        if not np.array_equal(actions,np.asarray([x['action'] for x in trace],np.float32)):raise ValueError('Teacher commands disagree')
        prefix=point['prefix_actions'];n=len(actions)
        if result['action_count']!=prefix+n or prefix+n>result['native_limit']:raise ValueError('Wrong native budget')
        sources.extend([source/'trace.jsonl',source/'fork.json',source/'native_actions.npz',source/'history_context.json'])
        context=read(source/'history_context.json')
        if context['past_observation']!=point['past_observation'] or context['past_observation_sha256']!=point['past_observation_sha256']:
            raise ValueError('Fork history changed')
        group=f"{point['task']}:{point['seed']}:p{point['index']}";scene=f"{point['task']}:{point['seed']}"
        def feature_record(path,count):
            sig,_=signature_file(path)
            row=dict(observation=str(path),observation_sha256=digest(path),observation_signature=sig,
                     instruction=point['instruction'],native_count=count,group=group,scene_group=scene)
            legal_observation(row);sources.append(path);return row
        past=Path(point['past_observation'])
        if digest(past)!=point['past_observation_sha256']:raise ValueError('Past source changed')
        # Feature-only causal predecessor; never becomes a teacher-labelled training row.
        feature_rows.append(feature_record(past,prefix-32));feature_history.append(-1)
        last_feature=len(feature_rows)-1;last_training=-1
        for offset in range(0,n,32):
            path=source/'fork.npz' if offset==0 else source/f'observation_{offset:04d}.npz'
            row=feature_record(path,prefix+offset);expected=fork['observation'] if offset==0 else trace[offset-1]['after_observation']
            if row['observation_signature']!=expected or trace[offset]['before_observation']!=expected:raise ValueError('Target predecessor mismatch')
            count=min(32,n-offset);target=np.zeros((32,14),np.float32);target[:count]=actions[offset:offset+count];valid=np.arange(32)<count
            plans=[]
            for j in range(count):
                tr=trace[offset+j];planfile=source/f"plan_{tr['plan']:03d}.npz"
                with np.load(planfile) as z:
                    if tr['dense_index'] not in z['indices'] or not np.array_equal(target[j],z['dense_commands'][tr['dense_index']]):raise ValueError('Unexecuted plan label')
                sources.append(planfile);plans.append(tr['plan'])
            dest=targets/f"point{point['index']}_n{prefix+offset}.npz"
            np.savez_compressed(dest,target=target,valid=valid,source_native_counts=np.where(valid,prefix+offset+np.arange(32)+1,-1),plan_ids=np.r_[plans,[-1]*(32-count)].astype(np.int64))
            row.update(task=point['task'],scene_seed=point['seed'],source=str(source),source_point=point['index'],
                native_limit=result['native_limit'],role='native_switch_recovery' if point['task']=='turn_switch' else 'verified_held_recovery',pilot_split='recovery_fit',kind='recovery',frame=prefix+offset,
                target=str(dest),target_sha256=digest(dest),valid_actions=count,past_observation=feature_rows[last_feature]['observation'],
                matched_compiled_base=[],scene_role='train_developer',offset=offset)
            current_ids.append(len(feature_rows));feature_history.append(last_feature);feature_rows.append(row.copy())
            last_feature=len(feature_rows)-1;history.append(last_training);last_training=len(rows);rows.append(row);sources.append(dest)
    if not rows:raise ValueError('No qualified full teacher trajectories')
    from .cache_expert import CHECKPOINT,RT
    from .evaluate_b2 import resolve_tokenizer
    tokenizer=resolve_tokenizer();weight=CHECKPOINT/'model.safetensors';norm=CHECKPOINT/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    identity=dict(base_checkpoint_sha256=digest(weight),normalization_sha256=digest(norm),tokenizer_sha256=digest(tokenizer))
    if identity['base_checkpoint_sha256']!=read(FEATURES/'checkpoint_audit.json')['sha256']:raise ValueError('Wrong frozen checkpoint')
    for source in sorted({Path(point['source']) for point,_ in selected}):
        prov=read(source/'provenance.json');sources.append(source/'provenance.json')
        if any(prov[k]!=v for k,v in identity.items()):raise ValueError('Wrong source policy identity')
        for name,h in prov['source_sha256'].items():
            f=RT/'RoboTwin/policy/pi05/src'/name if name.startswith('openpi/') else RT/'pi05_integration/register_motus_config.py' if name=='integration/register_motus_config.py' else None
            if f:
                if digest(f)!=h:raise ValueError('Policy code changed')
                sources.append(f)
    golden=next(r for r in read(FEATURES/'result.json')['records'] if r['source']['kind']=='expert' and r['source']['task']=='place_shoe' and r['source']['pilot_split']=='fit')
    sources.extend([Path(golden['file']),Path(golden['source']['source']),tokenizer,norm])
    sources.extend(ROOT/'cx_tta'/x for x in ('prepare_fixed_cache.py','cache_fixed_recovery.py','audit_fixed_cache.py','fixed_recovery_dataset.py','cache_recovery.py','cache_joint_prefix.py','recovery_context_v2.py','cache_expert.py','evaluate_b2.py','recovery_contract.py'))
    write_json(run/'inventory.json',dict(status='complete',attempts=attempts,qualified_branches=len(selected),
        independent_scenes=sorted({r['scene_group'] for r in rows}),rows=len(rows),valid_teacher_actions=sum(r['valid_actions'] for r in rows),
        excluded_support_assays=True,gate_repeat1_excluded=True))
    sources.append(run/'inventory.json');st=weight.stat()
    write_json(run/'protocol.json',dict(status='frozen_before_inference',rows=rows,feature_rows=feature_rows,feature_history_indices=feature_history,
        feature_current_indices=current_ids,history_indices=history,teacher_protocol=str(protocol),feature_cache=str(FEATURES),
        preprocess=str(FIT/'semantic_preprocess.npz'),coarse_preprocess=str(FIT/'coarse_preprocess.npz'),checkpoint=str(CHECKPOINT),tokenizer=str(tokenizer),
        identity=identity,checkpoint_stat=dict(size=st.st_size,mtime_ns=st.st_mtime_ns),golden=golden,policy_seed=910000,
        source_sha256={str(f):digest(f) for f in sorted(set(sources))},feature_contract=read(FEATURES/'manifest.json')['feature'],
        noise='Official compiled internal-noise policy; warm each branch before resetting chronological branch stream. Repeat from identical full RNG; retain first regardless of target match.',
        history='First row uses saved previous L0 observation only as past feature. Later rows use own teacher observation exactly32 actions earlier. No future or cross-branch observations. Arm state/pi, original fixed PCA/scaler.',
        forward_inputs='Current/past RGB, command state and instruction only. Scene IDs, plan IDs, outcomes and physical GT only metadata/labels.',
        scope='Developer recovery cache; branches grouped by whole source scene. No independent confirmation, original residual-cap or adapter benefit claim. No training/TTA.'))
    print('PREPARED',run,'rows',len(rows),'scenes',len({r['scene_group'] for r in rows}),flush=True)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--protocol',type=Path,required=True);a=p.parse_args();prepare(a.protocol.resolve())

if __name__=='__main__':main()
