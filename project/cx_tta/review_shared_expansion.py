"""Aggregate explicit audited runs without fitting or threshold selection (CPU only)."""
import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CONDITIONS = ('Z_zero', 'F_first_full', 'G_candidate_cost2', 'G_observation_cost2')
GATES = CONDITIONS[2:]


def read(path):
    return json.loads(path.read_text())


def stats(rows, condition):
    n = len(rows)
    base = sum(r['Z_zero'] for r in rows)
    successes = sum(r[condition] for r in rows)
    rescue = sum(not r['Z_zero'] and r[condition] for r in rows)
    harm = sum(r['Z_zero'] and not r[condition] for r in rows)
    result = dict(n=n, successes=successes, rate=successes/n, rescued=rescue,
                  harmed=harm, net=rescue-harm, gain_pp=100*(rescue-harm)/n,
                  base_successes=base, base_preserved=base-harm,
                  preservation=(base-harm)/base if base else None)
    if condition in GATES:
        result.update(scored=sum(r[condition+'_scored'] for r in rows),
                      accepted=sum(r[condition+'_accepted'] for r in rows),
                      rejected_harms=sum(r['Z_zero'] and not r['F_first_full']
                                         and not r[condition+'_accepted'] for r in rows),
                      missed_rescues=sum(not r['Z_zero'] and r['F_first_full']
                                         and not r[condition+'_accepted'] for r in rows))
    return result


def intervals(rows, draws=20000):
    """Descriptive percentile intervals, resample whole scenes within each task."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['task']][row['seed']].append(row)
    panels = [list(group.values()) for _, group in sorted(groups.items())]
    rng = random.Random(20261007)
    comparisons = [(c, 'Z_zero') for c in CONDITIONS[1:]] + [(GATES[0], c) for c in (CONDITIONS[1], GATES[1])]
    samples = {a+'__minus__'+b: [] for a,b in comparisons}
    for _ in range(draws):
        selected = [r for panel in panels for _ in panel for r in panel[rng.randrange(len(panel))]]
        for a,b in comparisons:
            samples[a+'__minus__'+b].append(100*sum(r[a]-r[b] for r in selected)/len(selected))
    return {key: [sorted(values)[int(.025*draws)], sorted(values)[int(.975*draws)]]
            for key,values in samples.items()}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--runs', nargs=4, type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    assert out.is_relative_to(root/'outputs/reports') and not out.exists()
    rows, runs, sources, identities = [], [], {}, set()
    protocol_hashes, registry_hashes, process_ids = set(), set(), set()
    for run in args.runs:
        run = run.resolve()
        assert run.is_relative_to(root/'outputs/shared_expansion')
        m, result, audit, branches = [read(run/n) for n in (
            'manifest.json', 'result.json', 'CPU_audit.json', 'independent_branch_check.json')]
        assert m['stage']=='shared_onset_expansion' and m['mode']=='full'
        assert result['status']==audit['status']=='passed' and audit['interface_passed']
        assert branches['passed'] and audit['censored']==0
        assert read(run/'storage_cleanup.json')['removed']
        for key in ('outcomes','changes','records','actual_queries','actual_actions','branch_equivalence'):
            assert result[key]==audit[key], key
        identity = (m['task'], m['block_id'])
        assert identity not in identities
        identities.add(identity)
        protocol_hashes.add(m['protocol_sha256']); registry_hashes.add(m['registry_records_sha256'])
        process_ids.add(m['process_instance_id'])
        ep = {(e['seed'],e['condition']):e for e in read(run/'episodes.json')}
        assert len(ep)==len(m['seeds'])*4 and all(e['complete'] for e in ep.values())
        branch = {(b['seed'],b['gate']):b for b in branches['branches']}
        for seed in m['seeds']:
            row = dict(task=m['task'], seed=seed, block=m['block_id'])
            row.update({c:int(ep[seed,c]['success']) for c in CONDITIONS})
            for c in GATES:
                b = branch[seed,c]
                assert row[c]==row[b['reference']]
                row.update({c+'_scored': b['scored_query'] is not None,
                            c+'_accepted': b['accepted'], c+'_margin': b['margin']})
                with gzip.open(run/'traces'/(ep[seed,c]['trace_stem']+'.jsonl.gz'),'rt') as stream:
                    scored = [x['empirical_gate'] for line in stream
                              if (x:=json.loads(line))['empirical_gate']['scored']]
                assert len(scored)==int(b['scored_query'] is not None)
                if scored:
                    row.update({c+'_score_'+k:v for k,v in scored[0]['scores'].items()})
            rows.append(row)
        runs.append(dict(path=str(run),task=m['task'],block=m['block_id'],
                         complete=audit['complete_rollouts'],queries=audit['actual_queries'],
                         actions=audit['actual_actions'],outcomes=audit['outcomes']))
        for name in ('manifest.json','episodes.json','result.json','CPU_audit.json','independent_branch_check.json'):
            sources[str(run/name)]=hashlib.sha256((run/name).read_bytes()).hexdigest()
    assert len(protocol_hashes)==len(registry_hashes)==1 and len(process_ids)==4
    assert identities=={(t,b) for t in ('place_empty_cup','beat_block_hammer') for b in (0,1)}
    scenes=defaultdict(list)
    for r in rows: scenes[r['task'],r['seed']].append(r)
    assert all(sorted(r['block'] for r in group)==[0,1] for group in scenes.values())
    summaries={}
    for task in ('all','beat_block_hammer','place_empty_cup'):
        subset=[r for r in rows if task=='all' or r['task']==task]
        summaries[task]=dict(distinct_scenes=len({(r['task'],r['seed']) for r in subset}),
                            conditions={c:stats(subset,c) for c in CONDITIONS},
                            oracle_successes=sum(max(r['Z_zero'],r['F_first_full']) for r in subset),
                            both_fail=sum(not r['Z_zero'] and not r['F_first_full'] for r in subset),
                            descriptive_scene_bootstrap_gain_pp_95=intervals(subset),
                            blocks={b:{c:stats([r for r in subset if r['block']==b],c)
                                       for c in CONDITIONS} for b in (0,1)})
    stability={c:dict(scenes=len(scenes),outcome_flips=sum(v[0][c]!=v[1][c] for v in scenes.values())) for c in CONDITIONS}
    for c in GATES:
        stability[c]['decision_flips']=sum(v[0][c+'_accepted']!=v[1][c+'_accepted'] for v in scenes.values())
    main_gate=summaries['all']['conditions']['G_candidate_cost2']
    progression=dict(
        each_task_positive=all(summaries[t]['conditions']['G_candidate_cost2']['net']>0 for t in ('beat_block_hammer','place_empty_cup')),
        each_process_pooled_positive=all(summaries['all']['blocks'][b]['G_candidate_cost2']['net']>0 for b in (0,1)),
        less_harm_than_F=main_gate['harmed']<summaries['all']['conditions']['F_first_full']['harmed'],
        positive_cost2_utility=main_gate['rescued']-2*main_gate['harmed']>0)
    result=dict(status='passed',scope='Frozen finite expansion; no fitting or threshold search',
                prospective_progression_checks=progression,progression_passed=all(progression.values()),
                complete_rollouts=len(rows)*4,distinct_scenes=len(scenes),runs=runs,summary=summaries,
                repeat_stability=stability,bootstrap=dict(draws=20000,seed=20261007,
                unit='Whole task/scene with both processes, within-task resampling; descriptive, not a safety bound'),
                source_sha256=sources,training_approved=False)
    sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.mkdir(parents=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'per_scene.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader();writer.writerows(sorted(rows,key=lambda r:(r['task'],r['seed'],r['block'])))
    print(out,flush=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','source_sha256')},indent=2),flush=True)


if __name__=='__main__':
    main()
