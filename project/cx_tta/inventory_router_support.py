"""Reconcile existing successful L0 snapshots and executed recovery sources; no collection."""
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .paths import ROOT, new_run, write_json
from .probe_candidates import read
from .audit import digest
from .eval_diagnostics import array_hash

OLD = ROOT/'outputs/reports/audits/recovery_inventory_20261002_063604_876790'
FEATURES = ROOT/'outputs/feature_caches/joint_prefix_20260930_141244_699126'
RECOVERY = ROOT/'outputs/feature_caches/recovery_20261002_072911_916687'
CONTEXT = ROOT/'outputs/feature_caches/recovery_context_20261002_075253_551153'
CROSSFIT = ROOT/'outputs/effect_learning/scene_crossfit_20261002_114334_518712'


def main():
    dest=new_run('router_support',group=('reports','audits'))
    hashes={}
    def tracked(path):
        hashes[str(path)]=digest(path)
        return read(path)
    def check_file(path,sha):
        actual=digest(path)
        if actual!=sha:raise ValueError(f'Changed artifact: {path}')
        hashes[str(path)]=actual
    config=tracked(ROOT/'configs/intervention_batch1_20260927.json')
    scenes={(r['task'],r['seed']):r for r in config['scenes']}
    feature=tracked(FEATURES/'result.json')
    fa=tracked(FEATURES/'feature_audit.json')
    if fa['status']!='passed':raise ValueError('Feature audit required')
    check_file(FEATURES/'result.json',fa['cache_result_sha256'])
    cached={}
    for row in feature['records']:
        s=row['source']
        if s['kind']=='anchor':cached[s['task'],s['seed'],s['chunk']]=row
    cross=tracked(CROSSFIT/'protocol.json');roles=defaultdict(set)
    for part,ids in cross['parts'].items():
        for i in ids:roles[cross['rows'][i]['group']].add(part)
    anchors={};rejected=[];duplicates=0;source_cache={};indexes=[]
    for family in ('cartesian_pulse_batches','short_window_batches','intervention_batches'):
        for ip in sorted((ROOT/'outputs'/family).glob('*/*/index.json')):
            index=tracked(ip);indexes.append(str(ip))
            if index['status']!='complete':raise ValueError('Incomplete canonical index')
            for job in index['records']:
                scene=job['task'],job['seed'];key=(*scene,job['chunk'])
                if scene not in scenes or not scenes[scene]['source_l0_success']:continue
                if key in anchors:duplicates+=1;continue
                run=Path(job['run']);m=tracked(run/'manifest.json');res=tracked(run/'result.json')
                for field,f in (('manifest_sha256','manifest.json'),('result_sha256','result.json')):
                    if field in job:check_file(run/f,job[field])
                if res['status']!='complete' or not res['source_observation_exact_match']:raise ValueError('Replay mismatch')
                source=Path(m['source']);trace=source/'traces'/f'r0_seed{scene[1]}.jsonl'
                check_file(trace,m['source_trace_sha256']);check_file(source/'manifest.json',m['source_manifest_sha256'])
                if (source,scene[1]) not in source_cache:
                    sm=tracked(source/'manifest.json')
                    ep=next(r for r in tracked(source/'episodes.json') if r['seed']==scene[1])
                    if not ep['success'] or not ep['eligible'] or sm['args']['condition']!='L0' or sm['args']['rollout_protocol']!='official':
                        raise ValueError('Not successful official L0')
                    import json
                    source_cache[source,scene[1]]=(ep,[json.loads(x) for x in trace.read_text().splitlines()])
                ep,steps=source_cache[source,scene[1]];step=steps[job['chunk']]
                checkpoint=run/'checkpoint.npz'
                with np.load(checkpoint) as z:
                    base=z['base_actions'];state=z['command_state'];images=[z[k+'_rgb'] for k in ('head','right','left')]
                if step['executed']!=32 or not np.array_equal(base,np.asarray(step['base_executed'],np.float32)) or array_hash(base)!=step['base_chunk_sha256']:
                    raise ValueError('Base target differs from successful source')
                if any(array_hash(im)!=step['observation_components']['images'][cam] for im,cam in zip(images,('head_camera','right_camera','left_camera'))):
                    raise ValueError('Wrong decision image')
                cmp=tracked(run/'observation_comparison.json')
                if not cmp['exact_match'] or cmp['source']!=step['observation_components']:raise ValueError('Wrong observation signature')
                frame=next((f for f in res['branches']['base'].get('frames',[]) if f['offset']==0),None)
                if frame is not None:
                    with np.load(run/frame['observed_file']) as z:
                        if array_hash(z['command_state_raw'])!=step['observation_components']['command_state'] or not np.array_equal(state,z['command_state_raw'].astype(np.float32)):
                            raise ValueError('Wrong raw command state')
                    hashes[str(run/frame['observed_file'])]=digest(run/frame['observed_file'])
                elif not any(array_hash(v)==step['observation_components']['command_state'] for v in (state,state.astype(np.float64))):
                    rejected.append(dict(run=str(run),reason='Raw command state unavailable'));continue
                group=f'{scene[0]}:{scene[1]}'
                existing=cached.get(key)
                if existing:
                    check_file(checkpoint,existing['source']['checkpoint_sha256'])
                    check_file(Path(existing['file']),existing['sha256'])
                anchors[key]=dict(task=scene[0],seed=scene[1],group=group,chunk=job['chunk'],
                    original_split=scenes[scene]['split'],current_s1y_roles=sorted(roles.get(group,())),
                    previous_pilot_role=None if existing is None else existing['source']['pilot_split'],
                    checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),source=str(source),
                    source_trace_sha256=digest(trace),base_chunk_sha256=array_hash(base),
                    feature_cache_record=existing,instruction=ep['instruction'],
                    candidate_use='Existing training/development only' if scenes[scene]['split']=='train' else
                                  'Reserved original intervention validation; inventory only, no fitting or promotion',
                    preservation_target='Exact successful L0 base command, not teacher action',training_approved=False)
                hashes[str(checkpoint)]=digest(checkpoint)
    old=tracked(OLD/'result.json')
    if not tracked(OLD/'audit.json')['passed']:raise ValueError('Recovery inventory unaudited')
    for f,sha in old['source_sha256'].items():check_file(Path(f),sha)
    rec=tracked(RECOVERY/'result.json')
    ca=tracked(CONTEXT/'audit.json')
    if not tracked(RECOVERY/'audit.json')['passed'] or not ca['passed']:
        raise ValueError('Recovery cache/context unaudited')
    check_file(CONTEXT/'data.npz',ca['data_sha256'])
    check_file(RECOVERY/'data.npz',rec['data_sha256']);check_file(RECOVERY/'features.npz',rec['features_sha256'])
    rc={(r['group'],r['native_count']):r for r in rec['rows']}
    recoveries=[]
    for row in old['records']:
        for name in ('observation','target'):check_file(Path(row[name]),row[name+'_sha256'])
        eligible=row['role'] in ('verified_held_recovery','native_switch_recovery')
        cached_row=rc.get((row['group'],row['native_count'])) if eligible else None
        if cached_row and cached_row['observation_signature']!=row['observation_signature']:
            raise ValueError('Recovery identity mismatch')
        recoveries.append(dict(task=row['task'],group=row['group'],native_count=row['native_count'],
            role=row['role'],source=row['source'],target=row['target'],valid_actions=row['valid_actions'],
            usable_executed_recovery=eligible,cached=cached_row is not None,
            independent_unused_scene=False,current_s1y_roles=sorted(roles.get(row['group'],())),
            reason='Already used development recovery; corrected context v2 required' if eligible else
                   'Restart-grasp semantics or successful-source teacher control; not a clean recovery/preservation label'))
    # Index completed continuation sources by scene. Multiple policies/repeats do not create new scenes.
    continuations=[]
    for path in sorted((ROOT/'outputs/recovery_continuations').glob('*/*/result.json')):
        res=tracked(path);m=tracked(path.parent/'manifest.json');group=f"{m['task']}:{m['scene_seed']}"
        continuations.append(dict(run=str(path.parent),group=group,status=res['status'],
            success_by_branch={b:sum(bool(r['success']) for r in res['reports'] if r['branch']==b)
                               for b in sorted({r['branch'] for r in res['reports']})},
            note='Success outcomes from alternative continuations are not semantically audited teacher labels; '
                 'check per-query legal image persistence before any new cache. Existing development scene.'))
    summary={}
    for task in sorted({s[0] for s in scenes}):
        aa=[a for a in anchors.values() if a['task']==task]
        rr=[r for r in recoveries if r['task']==task and r['usable_executed_recovery']]
        summary[task]=dict(success_snapshots=len(aa),success_scenes=len({r['group'] for r in aa}),
            cached_success_snapshots=sum(r['feature_cache_record'] is not None for r in aa),
            reserved_success_scenes=sorted({r['group'] for r in aa if r['original_split']=='validation'}),
            reserved_success_snapshots=sum(r['original_split']=='validation' for r in aa),
            cached_recovery_rows=sum(r['cached'] for r in rr),clean_recovery_scenes=sorted({r['group'] for r in rr}),
            unused_independent_clean_recovery_scenes=0)
    hashes[str(Path(__file__))]=digest(Path(__file__))
    write_json(dest/'result.json',dict(status='complete',summary=summary,anchors=list(anchors.values()),
        recovery_records=recoveries,continuation_sources=continuations,rejected=rejected,
        canonical_indexes=indexes,deduplicated_repeated_jobs=duplicates,source_sha256=hashes,
        scope='Canonical three intervention batch families + prior semantic recovery inventory and completed continuations. '
              'No claim that every unindexed file is training-eligible. Original validation remains reserved; '
              'this inventory does not train, extract features, or relabel successful teacher controls.'))
    write_json(dest/'audit.json',dict(passed=all(digest(Path(f))==sha for f,sha in hashes.items()),
        source_files=len(hashes),exact_success_snapshot_checks=len(anchors),
        recovered_cache_rows=sum(r['cached'] for r in recoveries),training_started=False))
    print('RUN',dest,flush=True);print(summary,flush=True)


if __name__=='__main__':main()
