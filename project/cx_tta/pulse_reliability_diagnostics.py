"""Frozen-data temporal effects and label-free train-support reliability proxies."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from .audit import digest
from .effect_learning_probe import weights
from .paths import new_run, write_json
from .shared_effect_model import fold_indices

MAIN = {'beat_block_hammer': 'hammer', 'place_empty_cup': 'cup', 'place_shoe': 'shoe', 'turn_switch': 'switch'}
KINDS = [('translation_m', np.arange(3), .005), ('rotation_rad', np.arange(3, 6), .02), ('joint_fraction', np.array([6]), .005)]


def nearest_other_unit(features, units):
    """Each reference excludes ALL rows from its own scene/task, including duplicates."""
    distance = cdist(features, features, metric='sqeuclidean') / features.shape[1]
    distance[np.asarray(units)[:, None] == np.asarray(units)[None, :]] = np.inf
    return np.sqrt(distance.min(axis=1))


def support_scores(context, records, train, test, unit):
    unique = {}
    for i in train:
        unique.setdefault((records[i]['job'], records[i]['entity'], records[i]['horizon']), i)
    refs = np.array(list(unique.values()))
    calibration = nearest_other_unit(context[refs], [records[i][unit] for i in refs])
    if not np.isfinite(calibration).all():
        raise ValueError('Insufficient independent reference groups')
    distance = np.sqrt(cdist(context[test], context[refs], metric='sqeuclidean').min(axis=1) / context.shape[1])
    thresholds = np.quantile(calibration, [.25, .5, .75, 1.])
    return distance, thresholds, refs


def temporal(dataset):
    index = json.loads((dataset / 'index.json').read_text())
    labels_path = dataset / 'labels_offline.npz'
    if digest(labels_path) != index['labels_sha256']:
        raise ValueError('Labels changed')
    records = index['records']
    with np.load(labels_path) as z:
        y, mask = z['target'] * z['scale'], z['mask']
    lookup = {(r['job'], r['entity'], r['branch'], r['horizon']): i for i, r in enumerate(records)}
    reports = []
    for task, entity in MAIN.items():
        ids = np.array([i for i, r in enumerate(records) if r['task'] == task and r['entity'] == entity and r['horizon'] == 24])
        early = np.array([lookup[records[i]['job'], entity, records[i]['branch'], 8] for i in ids])
        for kind, columns, threshold in KINDS:
            valid = mask[np.ix_(ids, columns)].all(1) & mask[np.ix_(early, columns)].all(1)
            a, b = ids[valid], early[valid]
            if not len(a):
                continue
            final, pulse_end = y[np.ix_(a, columns)], y[np.ix_(b, columns)]
            norms = np.linalg.norm(final, axis=1)
            initial = np.linalg.norm(pulse_end, axis=1)
            event = norms > threshold
            delayed = event & (initial <= threshold)
            # Report a low-amplitude cosine separately; never filter training by this.
            floor = .0005 if kind == 'translation_m' else threshold / 10
            directional = (norms > floor) & (initial > floor)
            cos = (final[directional] * pulse_end[directional]).sum(1) / (norms[directional] * initial[directional])
            correlation = spearmanr(initial, norms).statistic if np.ptp(initial) > 0 and np.ptp(norms) > 0 else None
            reports.append(dict(task=task, entity=entity, kind=kind, rows=len(a), scenes=len({records[i]['group'] for i in a}), pulse_end_events=int((initial > threshold).sum()), final_events=int(event.sum()), delayed_events=int(delayed.sum()), delayed_event_scenes=len({records[i]['group'] for i in a[delayed]}), vanished_events=int(((initial > threshold) & ~event).sum()), spearman_magnitude_h8_h24=None if correlation is None else float(correlation), direction_pairs=len(cos), direction_positive=int((cos > 0).sum()), direction_median_cosine=float(np.median(cos)) if len(cos) else None, directional_floor=floor, h8_max=float(initial.max()), h24_max=float(norms.max())))
    return dict(reports=reports, scope='Pulse origin offset8; pulse ends offset16 (h8); last frame offset32 (h24). Thresholds are descriptive only; branch/horizon rows are correlated, not independent scenes. No repeated identical interventions or amplitude sweep: temporal persistence is NOT repeatability, causal noise floor or proof of local linearity.')


def evaluate(run, dataset):
    manifest = json.loads((run / 'manifest.json').read_text())
    index = json.loads((dataset / 'index.json').read_text())
    if Path(manifest['dataset']).resolve() != dataset.resolve() or digest(dataset / 'index.json') != manifest['dataset_index_sha256']:
        raise ValueError('Model/dataset mismatch')
    if json.loads((run / 'result.json').read_text())['status'] != 'complete':
        raise ValueError('Incomplete training')
    records = index['records']
    with np.load(dataset / 'inputs_oracle.npz') as z:
        vision, geometry = z['vision'], z['geometry']
    with np.load(dataset / 'labels_offline.npz') as z:
        target, mask, scale = z['target'], z['mask'], z['scale']
    reports, audit = [], []
    raw_scores = {}
    for unit in ('group', 'task'):
        with np.load(run / (unit + '_predictions.npz')) as z:
            pred, supported = z['conditioned'], z['supported']
            if not np.array_equal(z['target'], target) or not np.array_equal(z['mask'], mask):
                raise ValueError('Prediction alignment changed')
        distance = np.full(len(records), np.nan)
        accepted = np.zeros((len(records), 4), bool)
        for fold, held in enumerate(sorted({r[unit] for r in records})):
            train, test = fold_indices(records, held, unit)
            if {records[i]['group'] for i in train} & {records[i]['group'] for i in test}:
                raise ValueError('Scene leakage')
            with np.load(run / f'{unit}_{fold:02d}_preprocess.npz') as z:
                pre = dict(z)
            if not np.array_equal(pre['fit_rows'], train):
                raise ValueError('Normalizer leakage')
            v = (vision - pre['pca_mean']) @ pre['pca_components'].T
            context = ((np.c_[v, geometry] - pre['context_mean']) / pre['context_std']).astype(np.float32)
            score, thresholds, refs = support_scores(context, records, train, test, unit)
            distance[test] = score
            accepted[test] = score[:, None] <= thresholds
            audit.append(dict(unit=unit, held=held, reference_rows=len(refs), reference_scenes=len({records[i]['group'] for i in refs}), thresholds=thresholds.tolist(), excluded_calibration_unit=unit, held_labels_used_for_threshold=False))
        raw_scores[unit + '_distance'] = distance
        raw_scores[unit + '_accepted'] = accepted
        for task, entity in MAIN.items():
            for horizon in (4, 8, 12, 16, 20, 24):
                ids = np.array([i for i, r in enumerate(records) if r['task'] == task and r['entity'] == entity and r['horizon'] == horizon])
                for kind, columns, threshold in KINDS:
                    ids_valid = ids[mask[np.ix_(ids, columns)].all(1)]
                    if not len(ids_valid):
                        continue
                    base = dict(unit=unit, task=task, entity=entity, horizon=horizon, kind=kind, rows=len(ids_valid))
                    if not supported[np.ix_(ids_valid, columns)].all():
                        reports.append(dict(base, status='unsupported_output_in_training'))
                        continue
                    ids_use = ids_valid
                    actual = target[np.ix_(ids_use, columns)] * scale[columns]
                    prediction = pred[np.ix_(ids_use, columns)] * scale[columns]
                    zeroerr = np.mean(actual ** 2, axis=1)
                    error = np.mean((prediction - actual) ** 2, axis=1)
                    excess = error - zeroerr
                    norm = np.linalg.norm(actual, axis=1)
                    events = norm > threshold
                    w = weights([records[i] for i in ids_use])
                    curves = []
                    for col, quantile in enumerate((.25, .5, .75, 1.)):
                        keep = accepted[ids_use, col]
                        retained_mass = float(w @ keep)
                        # SAME retained rows and original scene weights for both risks.
                        zerr = float(w[keep] @ zeroerr[keep] / retained_mass) if retained_mass else None
                        perr = float(w[keep] @ error[keep] / retained_mass) if retained_mass else None
                        evmass = float(w @ events)
                        curves.append(dict(train_reference_quantile=quantile, accepted_rows=int(keep.sum()), accepted_scenes=len({records[i]['group'] for i in ids_use[keep]}), coverage=retained_mass, event_rows=int(events.sum()), retained_event_rows=int((events & keep).sum()), event_retention=float(w @ (events & keep) / evmass) if evmass else None, accepted_mse=perr, same_rows_zero_mse=zerr, accepted_improvement_vs_zero=1 - perr / zerr if zerr is not None and zerr > 1e-16 else None, all_rows_zero_fallback_improvement=1 - float(w @ np.where(keep, error, zeroerr)) / float(w @ zeroerr) if w @ zeroerr > 1e-16 else None))
                    corr = spearmanr(distance[ids_use], excess).statistic if np.ptp(excess) > 0 and np.ptp(distance[ids_use]) > 0 else None
                    reports.append(dict(base, status='complete', distance_excess_error_spearman=None if corr is None else float(corr), all_rows_improvement_vs_zero=1 - float(w @ error) / float(w @ zeroerr) if w @ zeroerr > 1e-16 else None, curves=curves))
    return dict(model=str(run), dataset=str(dataset), folds=audit, reports=reports, scope='Label-free RMS nearest conditional-context distance, not a calibrated probability or ensemble uncertainty. Reference and normalization use training folds only; threshold references exclude their own entire held-unit (scene or task). Threshold quantiles pre-fixed .25/.50/.75/1.0, no held-error threshold selection. Same state score for all directions at a horizon; no action-specific confidence claim. Primary results are scene-weighted and compare to zero on identical accepted rows. Zero fallback is offline effect prediction, NOT a policy or demonstrated safe action.'), raw_scores


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--model', type=Path, action='append', default=[])
    a = p.parse_args()
    run = new_run('pulse_reliability', group=('reports', 'audits'))
    write_json(run / 'temporal.json', temporal(a.dataset))
    for n, model in enumerate(a.model):
        manifest = json.loads((model / 'manifest.json').read_text())
        report, scores = evaluate(model, Path(manifest['dataset']))
        write_json(run / f'reliability_{n}.json', report)
        np.savez_compressed(run / f'reliability_{n}_scores.npz', **scores)
        print('CHECKED ' + str(model), flush=True)
    write_json(run / 'manifest.json', dict(status='complete', dataset=str(a.dataset), dataset_index_sha256=digest(a.dataset / 'index.json'), models=list(map(str, a.model)), code_sha256=digest(Path(__file__)), scope='Existing original-train data only; no new collection, validation tuning or TTA.'))
    print(run, flush=True)

if __name__ == '__main__':
    main()
