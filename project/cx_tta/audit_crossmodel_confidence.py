"""Independently verify the post-hoc held-model rankings on real saved predictions."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .audit import digest
from .paths import write_json
from .probe_candidates import read, verify, load_candidate


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run.resolve()
    threadpool_limits(2)
    p = read(run / 'protocol.json')
    cp = read(run / 'crossmodel_protocol.json')
    result = read(run / 'crossmodel_result.json')
    verify(p)
    verify(cp)
    with np.load(p['data']) as z:
        data = {k: z[k].copy() for k in z.files}
    with np.load(run / 'observable_scores.npz') as z:
        ids, weights = z['held_indices'].copy(), z['held_weights'].copy()
    # Reconstruct weighting from complete scene membership, rather than trusting saved weights.
    tasks = np.array([p['rows'][i]['task'] for i in ids]).repeat(2)
    groups = np.array([p['rows'][i]['group'] for i in ids]).repeat(2)
    expected = np.zeros(len(weights))
    for task in set(tasks):
        gs = set(groups[tasks == task])
        for group in gs:
            mask = groups == group
            expected[mask] = 1 / (len(set(tasks)) * len(gs) * mask.sum())
    np.testing.assert_array_equal(weights, expected)
    context = data['semantic'].astype(float)
    ref = context[p['parts']['expert_fit']]
    near = -np.array([np.linalg.norm(ref - context[i], axis=1).min() / np.sqrt(context.shape[1]) for i in ids])
    near = near.repeat(2)
    maximum = 0.
    checked = 0
    original = read(run / 'result.json')['reports']
    for condition in sorted({x['condition'] for x in result['reports']}):
        models = {x['seed']: x for x in p['models'] if x['condition'] == condition}
        candidates = {s: load_candidate(m, p, data)[ids].astype(float) for s, m in models.items()}
        panels = {}
        for report in original:
            if report['condition'] == condition:
                with np.load(report['panel']) as z:
                    panels[report['seed']] = {k: z[k].copy() for k in z.files if k.endswith('_benefit')}
        for report in [x for x in result['reports'] if x['condition'] == condition]:
            held = report['held_seed']
            others = report['scoring_seeds']
            if len(others) != 2 or set(others) != set(models) - {held}:
                raise ValueError('Held model leaked into confidence score')
            component = report['component']
            channel = slice(0, 6) if component == 'arm' else slice(6, 7)
            cap = .1 if component == 'arm' else .2
            # For two models the population standard deviation is half their difference.
            residuals = [(candidates[s] - data['base'][ids]).reshape(len(ids), 32, 2, 7)[..., channel] / cap for s in others]
            spread = np.sqrt(np.mean(((residuals[0] - residuals[1]) / 2) ** 2, axis=(1, 3))).ravel()
            amplitude = np.sqrt(np.mean(((residuals[0] + residuals[1]) / 2) ** 2, axis=(1, 3))).ravel()
            score = dict(relative_agreement=-spread / (amplitude + 1e-6), low_disagreement=-spread,
                         high_correction_magnitude=amplitude, near_fit_context=near)[report['score']]
            flags = panels[held][report['support'] + '_' + component + '_benefit']
            label = np.logical_or.reduce(flags[1:], axis=0)[ids].ravel()
            selections = [('inner_held', np.ones(len(label), bool), report['inner_held'])]
            selections.extend((task, tasks == task, value) for task, value in report['per_task'].items())
            for _, mask, saved in selections:
                s, y, w = score[mask], label[mask], weights[mask]
                if y.any() and (~y).any():
                    difference = s[y, None] - s[None, ~y]
                    mass = w[y, None] * w[None, ~y]
                    auc = float((mass * ((difference > 0) + .5 * (difference == 0))).sum() / mass.sum())
                    maximum = max(maximum, abs(auc - saved['auc']))
                    if abs(auc - saved['auc']) > 1e-10:
                        raise ValueError('Independent AUC mismatch')
                elif saved['auc'] is not None:
                    raise ValueError('Single-class AUC must be null')
                if int(y.sum()) != saved['positive_units']:
                    raise ValueError('Held-model labels mismatch')
                selected = s >= np.quantile(s, .9)
                for key, value in [('label_prevalence', np.average(y, weights=w)),
                                   ('actual_weighted_coverage', w[selected].sum() / w.sum()),
                                   ('top10_weighted_precision', np.average(y[selected], weights=w[selected]))]:
                    if abs(float(value) - saved[key]) > 1e-10:
                        raise ValueError('Independent ranking statistic mismatch: ' + key)
            checked += 1
    if checked != 192:
        raise ValueError('Incomplete condition/seed/support/component/score matrix')
    verify(p)
    verify(cp)
    write_json(run / 'crossmodel_audit.json', dict(passed=True, report_checks=checked,
        held_model_excluded_from_scores=True, independent_pairwise_auc_max_error=maximum,
        scene_weights_recomputed=True, source_sha256={str(f): digest(f) for f in
            [run / 'crossmodel_protocol.json', run / 'crossmodel_result.json', Path(__file__)]},
        scope='Numerical/input separation audit only; reused developer scenes, post-hoc check, no independent recovery evidence.'))
    print('PASS', checked, maximum)


if __name__ == '__main__':
    main()
