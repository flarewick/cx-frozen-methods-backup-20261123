"""Audited one-onset decisions; metadata and paired labels never enter forward.

This is an offline data builder, not permission to fit or deploy a router.
Repeated processes share a scene group and cannot repair missing scene support.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .audit import digest
from .native_history_cache import read, arrays, require, verify
from .paths import ROOT, output_path, write_json
from .local_benefit_cache import load_observables
from .onset_observables import at_query, concatenate

CLASSES = ('both_fail', 'rescue', 'harm', 'both_success')
OLD = ROOT/'outputs/reports/diagnostics/onset_attribution_20261007_020824_193778/TRAIN_development'


def outcome(z, f):
    if type(z) is not bool or type(f) is not bool:
        raise ValueError('Only complete Boolean paired outcomes may be labelled')
    return CLASSES[2 * int(z) + int(f)]


def summarize_rows(rows):
    counts = Counter(outcome(r['Z_success'], r['F_success']) for r in rows)
    return dict(pairs=len(rows), scenes=len({r['group'] for r in rows}),
        Z_success=sum(r['Z_success'] for r in rows), F_success=sum(r['F_success'] for r in rows),
        **{k: counts[k] for k in CLASSES}, net=counts['rescue']-counts['harm'])


def support(rows, criteria):
    require(bool(rows), 'Nonempty audited panel')
    seen = set(); groups = defaultdict(set)
    for r in rows:
        require(r['audited'] is True and r['split'] in ('fit', 'calibration'), 'Audited TRAIN only')
        require(r['group'] == f"{r['task']}:{r['seed']}", 'Scene group identity')
        require(r['label'] == outcome(r['Z_success'], r['F_success']), 'Paired label consistency')
        key = r['group'], r['block']
        require(key not in seen, 'Duplicate scene/process decision')
        seen.add(key); groups[r['group']].add(r['split'])
    require(all(len(v) == 1 for v in groups.values()), 'Scene split leakage across repeats')
    result = {}
    for split in ('fit', 'calibration'):
        subset = [r for r in rows if r['split'] == split]
        result[split] = dict(summarize_rows(subset),
            positive_scenes=sorted({r['group'] for r in subset if r['label']=='rescue'}),
            negative_scenes=sorted({r['group'] for r in subset if r['label']=='harm'}),
            base_success_scenes=sorted({r['group'] for r in subset if r['Z_success']}),
            by_task={t: summarize_rows([r for r in subset if r['task']==t])
                     for t in sorted({r['task'] for r in rows})})
    missing = []
    for name in ('positive', 'negative', 'base_success'):
        count = len(result['fit'][name+'_scenes'])
        minimum = criteria['minimum_fit_'+name+'_distinct_scenes']
        if count < minimum:
            missing.append(f'fit {name}: {count} distinct scenes < {minimum}')
        if criteria['calibration_must_contain_positive_negative_and_base_success']:
            if not result['calibration'][name+'_scenes']:
                missing.append(f'calibration {name}: no distinct scene')
    unstable = []
    for g in sorted(groups):
        rs = [r for r in rows if r['group']==g]
        if len({(r['Z_success'], r['F_success']) for r in rs}) > 1:
            unstable.append(dict(group=g, outcomes=[dict(block=r['block'], label=r['label']) for r in rs]))
    return dict(coverage_passed=not missing, missing=missing, splits=result,
        repeat_instability=unstable, training_approved=False,
        reason='Coverage alone never certifies risk calibration or deployment')


def require_fit_support(rows, criteria):
    result = support(rows, criteria)
    if not result['coverage_passed']:
        raise ValueError('Router fitting blocked by frozen support gate: '+ '; '.join(result['missing']))
    return result


def build(review):
    review = output_path(Path(review)); index = read(review/'audit_index.json')
    require(index['stored_results_match'] is True and len(index['runs'])==4, 'Four reaudited runs')
    audits = [read(p) for p in index['audits']]
    require(len(audits)==4 and len({a['process_instance_id'] for a in audits})==4, 'Independent processes')
    require({(a['task'],a['block_id']) for a in audits} ==
            {(t,b) for t in ('place_empty_cup','beat_block_hammer') for b in (0,1)}, 'Complete task/block panel')
    packets=[]; rows=[]; sources={str(review/'audit_index.json'):digest(review/'audit_index.json')}
    protocols=set()
    for run_name in index['runs']:
        run=Path(run_name); m=read(run/'manifest.json'); protocols.add(m['protocol'])
        a=next(a for a in audits if (a['task'],a['block_id'])==(m['task'],m['block_id']))
        require(a['status']=='passed' and a['interface_passed'] and a['censored']==0, 'Accepted complete audit')
        verify(a['source_sha256']); sources.update(a['source_sha256'])
        es=read(run/'episodes.json')
        for d in a['paired_decisions']:
            require(d['passed'] and d['kind']=='onset', 'Actual common onset')
            if not d['decision_reached']:
                continue  # Retained by full audit; no invented decision row.
            pair={e['condition']:e for e in es if e['seed']==d['seed']}
            require(set(pair)=={'Z_zero','F_first_full'} and all(e['complete'] for e in pair.values()), 'Full Z/F pair')
            z,f=pair['Z_zero'],pair['F_first_full']; q=d['query']
            xp,base=load_observables(Path(z['cache'])); raw=arrays(Path(z['cache'])/'queries.npz')['raw_delta']
            packet=at_query(xp,base,raw,q)
            xf,bf=load_observables(Path(f['cache'])); rf=arrays(Path(f['cache'])/'queries.npz')['raw_delta']
            other=at_query(xf,bf,rf,q)
            require(all(packet[k].tobytes()==other[k].tobytes() for k in packet), 'Shared decision packet bytes')
            packets.append(packet)
            rows.append(dict(index=len(rows),task=m['task'],seed=d['seed'],block=m['block_id'],query=q,
                group=f"{m['task']}:{d['seed']}",split=m['scene_split'][str(d['seed'])],
                Z_success=z['success'],F_success=f['success'],label=outcome(z['success'],f['success']),
                audited=True,source_run=str(run),source='new_TRAIN',training_approved=False))
        for name in ('manifest.json','episodes.json','result.json'):
            sources[str(run/name)]=digest(run/name)
    require(len(protocols)==1, 'Same frozen collection protocol')
    protocol=Path(next(iter(protocols))); p=read(protocol)
    sources[str(protocol)]=digest(protocol)
    # Old four TRAIN scenes are fit-only; never load any confirmation directory.
    om=read(OLD/'manifest.json'); require(om['role']=='TRAIN_development' and digest(OLD/'observables.npz')==om['data_sha256'], 'Old TRAIN packet identity')
    old_packets=arrays(OLD/'observables.npz'); old_rows=read(OLD/'offline_index.json')['rows']
    require(len(old_rows)==8 and {r['group'] for r in old_rows}=={
        'place_empty_cup:501101','place_empty_cup:501102','beat_block_hammer:501100','beat_block_hammer:501103'}, 'Fixed fit-only parent scenes')
    from .arm_prescription_analysis_v2 import summarize as audit_old
    for run_name in sorted({r['source_run'] for r in old_rows}):
        a=audit_old(Path(run_name)); require(a['status']=='passed' and a['interface_passed'], 'Old real Z/F audit')
        sources.update(a['source_sha256'])
        es=read(Path(run_name)/'episodes.json')
        for r in [r for r in old_rows if r['source_run']==run_name]:
            d=next(d for d in a['paired_decisions'] if d['seed']==r['seed'] and d['a']=='Z_zero' and d['b']=='F_first_full' and d['kind']=='onset')
            require(d['passed'] and d['decision_reached'] and d['query']==r['query'], 'Old actual onset')
            for c,k in [('Z_zero','Z_success'),('F_first_full','F_success')]:
                e=next(e for e in es if e['seed']==r['seed'] and e['condition']==c)
                require(e['complete'] and e['success']==r[k], 'Old full outcome identity')
            xp,base=load_observables(Path(r['source_cache'])); raw=arrays(Path(r['source_cache'])/'queries.npz')['raw_delta']
            packet=at_query(xp,base,raw,r['query'])
            require(all(packet[k].tobytes()==old_packets[k][r['index']:r['index']+1].tobytes() for k in packet), 'Parent packet unchanged')
            packets.append(packet)
            rows.append(dict(r,index=len(rows),split='fit',audited=True,source='old_TRAIN',
                label=outcome(r['Z_success'],r['F_success'])))
    for n in ('manifest.json','offline_index.json','observables.npz'):
        sources[str(OLD/n)]=digest(OLD/n)
    decision_dir=review/'decisions'; decision_dir.mkdir(exist_ok=False)
    np.savez_compressed(decision_dir/'observables.npz',**concatenate(packets))
    counts=Counter(r['group'] for r in rows)
    for r in rows:r['scene_weight']=1.0/counts[r['group']]
    write_json(decision_dir/'offline_index.json',dict(rows=rows,scope='Labels/grouping only; not forward input'))
    result=support(rows,p['support_gate'])
    result.update(new_TRAIN=summarize_rows([r for r in rows if r['source']=='new_TRAIN']),
        by_task={t:summarize_rows([r for r in rows if r['source']=='new_TRAIN' and r['task']==t]) for t in p['tasks']},
        input_rows=len(rows), source_sha256=sources,criteria=p['support_gate'],
        off_onset_supervision=False,model_trained=False)
    write_json(review/'decision_support.json',result)
    write_json(decision_dir/'manifest.json',dict(rows=len(rows),fields=sorted(packets[0]),
        observables_sha256=digest(decision_dir/'observables.npz'),offline_index_sha256=digest(decision_dir/'offline_index.json'),
        training_approved=False,coverage_passed=result['coverage_passed'],source_sha256=sources))
    print({k:result[k] for k in ('new_TRAIN','by_task','coverage_passed','missing')},flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--review',type=Path,required=True)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(2)
    build(args.review)


if __name__=='__main__':main()
