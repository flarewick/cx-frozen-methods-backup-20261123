"""CPU readback of actual prefix actions, first pairing and complete outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import write_json
from .native_history_cache import read,arrays,verify,require
from .native_entry_probe import CONDITIONS,decision
from .eval_diagnostics import array_hash


def audit(run):
    run=Path(run).resolve();manifest=read(run/'manifest.json');protocol=Path(manifest['protocol'])
    require(digest(protocol)==manifest['protocol_sha256'],'Probe protocol changed')
    p=read(protocol);verify(p['source_sha256']);r=read(run/'result.json');episodes=read(run/'episodes.json')
    require(r['status']=='complete' and r['episodes']==episodes,'Complete saved branch results required')
    repetitions=1 if manifest['smoke'] else 2
    require(len(episodes)==repetitions*len(CONDITIONS),'Missing selected condition/repeat')
    point=manifest['point'];target=arrays(point['target']);valid=target['valid'];caps=np.asarray(p['caps'],np.float32)
    first=None;sources={};snapshots={};checked=0
    for episode in episodes:
        repeat,condition=episode['repeat'],episode['condition']
        stem=f'r{repeat}_{condition}';path=run/(stem+'.jsonl');values=arrays(run/(stem+'.npz'))
        lines=path.read_text().splitlines();records=[json.loads(x) for x in lines]
        require(len(records)==episode['queries']==len(values['base'])==len(values['actions']),'Query arrays/trace length')
        require(values['base'].shape==values['actions'].shape==(len(records),32,14),'Saved native arrays shape')
        trajectory=hashlib.sha256();native=point['native_count']
        for i,(row,base,action) in enumerate(zip(records,values['base'],values['actions'])):
            require(i==row['query'] and row['base_sha256']==array_hash(base) and row['actions_sha256']==array_hash(action),
                'Numeric action provenance mismatch')
            require(row['native_before']==native and 0<row['executed']<=32,'Native query budget mismatch')
            native+=row['executed'];require(row['native_after']==native,'Native increment mismatch')
            if i==0:
                # Independent NumPy formula, ignoring invalid teacher padding.
                delta=np.zeros_like(base);delta[valid]=target['target'][valid]-base[valid]
                if condition=='L0':expected=base.copy()
                elif condition=='teacher_prefix_raw':
                    expected=base.copy();expected[valid]=target['target'][valid]
                elif condition=='teacher_prefix_clip':expected=base+np.maximum(np.minimum(delta,caps),-caps)
                else:
                    applied=np.maximum(np.minimum(delta,caps),-caps)
                    for start in (0,7):
                        ratios=np.abs(delta[:,start:start+6])/caps[start:start+6]
                        divisor=np.maximum(1.,ratios.max(-1))
                        applied[:,start:start+6]=delta[:,start:start+6]/divisor[:,None]
                    expected=base+applied
                expected[~valid]=base[~valid]
                require(expected.tobytes()==action.tobytes(),'Applied prefix differs from declared formula')
                identity={key:row[key] for key in ('observation_signature','before_physical_sha256','base_sha256','rng_before','rng_after')}
                require(identity['observation_signature']==point['source_signature'],'Fork input source mismatch')
                if first is None:first=identity
                else:require(identity==first,'Actual first-query pairing mismatch')
            else:require(action.tobytes()==base.tobytes(),'Unexpected correction after teacher prefix')
            if condition!='teacher_prefix_raw':
                require(not np.any(np.abs(action-base)>caps+2e-6),'Original caps violated')
            trajectory.update(json.dumps(row,sort_keys=True,separators=(',',':')).encode())
            checked+=1
        require(native==episode['executed_actions'] and trajectory.hexdigest()==episode['trajectory_sha256'],
            'Complete trajectory identity differs')
        if not manifest['smoke']:
            require(episode['complete'] and isinstance(episode['success'],bool),'Censored branch cannot count')
            if not episode['success']:require(native==(500 if point['task']=='place_shoe' else 400),'Failed branch not exhausted')
        else:require(not episode['complete'] and episode['success'] is None,'Smoke cannot count as task success')
        snapshots[condition,repeat]=(records,values)
        sources[str(path)]=digest(path);sources[str(run/(stem+'.npz'))]=digest(run/(stem+'.npz'))
    if not manifest['smoke']:
        for condition in CONDITIONS:
            left,right=snapshots[condition,0],snapshots[condition,1]
            require(left[0]==right[0] and all(left[1][key].tobytes()==right[1][key].tobytes() for key in left[1]),
                'Measured full repeats differ')
    recomputed=decision(episodes,p['repeats'],manifest['smoke']);require(recomputed==r['decision'],'Saved decision differs')
    live=arrays(run/'first_live_base_target.npz')
    require(array_hash(live['base'])==first['base_sha256'],'Qualification base differs from executed query')
    for key in ('target','valid','source_native_counts'):
        require(live[key].tobytes()==target[key].tobytes(),'Fresh-base certificate teacher source mismatch')
    for key,condition in (('coordinate_clip','teacher_prefix_clip'),('arm_direction','teacher_prefix_arm_direction')):
        require(live[key].tobytes()==snapshots[condition,0][1]['actions'][0].tobytes(),
            'Fresh-base certificate does not match executed bounded prefix')
    require(read(run/'source_runtime_audit.json')['passed'],'Shared runtime source integrity required')
    storage=read(run/'native_storage.json');require(storage['large_static_files_copied']==0,'Unexpected large runtime copy')
    for path,expected in storage['static_source_metadata'].items():
        st=Path(path).stat()
        require(digest(Path(path))==expected['sha256'] and st.st_size==expected['bytes']
            and st.st_mtime_ns==expected['mtime_ns'] and st.st_mode==expected['mode'],
            'Shared static runtime content/metadata changed')
    summary={condition:dict(successes=sum(e['success'] is True for e in episodes if e['condition']==condition),
        completed=sum(e['complete'] for e in episodes if e['condition']==condition),
        censored=sum(not e['complete'] for e in episodes if e['condition']==condition)) for condition in CONDITIONS}
    for name in ('manifest.json','resolved_config.json','episodes.json','result.json','steady_probe.json',
        'first_live_base_target.npz','source_runtime_audit.json','native_storage.json'):
        sources[str(run/name)]=digest(run/name)
    sources[str(Path(__file__))]=digest(Path(__file__))
    report=dict(passed=True,smoke=manifest['smoke'],queries_verified=checked,
        first_pairing_compared_branches=len(episodes),complete_success_counts=summary,
        original_caps_and_prefix_only_intervention_verified=True,full_repeat_arrays_verified=not manifest['smoke'],
        decision=recomputed,source_sha256=sources,gpu_started=False,learned_adapter_used=False,
        scope='Actual native-entry target diagnostic. Smoke has no effect evidence; full certificate is one selected training state only.')
    write_json(run/'CPU_audit.json',report)
    print(json.dumps(dict(passed=True,smoke=manifest['smoke'],summary=summary,decision=recomputed),ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True)
    audit(p.parse_args().run)


if __name__=='__main__':main()
