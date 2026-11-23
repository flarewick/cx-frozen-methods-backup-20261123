"""CPU learning-interface probe: predict intervention-induced RGB feature changes.

This is not the M1 object-relation model, residual controller, or TTA. Future
images are supervision only. All normalizers/PCA are fitted on training scenes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .audit import digest
from .audit_interventions import audit_run
from .dataset import context_features
from .intervention_batch import DEFAULT_PLAN, validate_plan
from .paths import ROOT, new_run, write_json


def endpoint_rgb_features(data):
    images = [data[name + '_rgb_after'] for name in ('head', 'right', 'left')]
    return context_features(images, np.zeros(14, np.float32))[14:]


def load_data(plan_path):
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    planned = {j['id']: j for j in plan['jobs']}
    seen, scene_splits, records, excluded, provenance = set(), {}, [], [], []
    contexts, residuals, targets = [], [], []
    for task in sorted({j['task'] for j in planned.values()}):
        index_path = ROOT / 'outputs/intervention_batches' / plan['batch_id'] / task / 'index.json'
        index = json.loads(index_path.read_text())
        if index['status'] != 'complete' or index['plan_sha256'] != digest(plan_path):
            raise ValueError('Incomplete or changed batch index')
        provenance.append({'index': str(index_path), 'sha256': digest(index_path)})
        for row in index['records']:
            job = planned[row['id']]
            if row['id'] in seen or any(row[k] != job[k] for k in job if k not in ('reuse_run', 'reuse_manifest_sha256')):
                raise ValueError('Duplicate or mismatching job')
            seen.add(row['id'])
            group = f"{row['task']}:{row['seed']}"
            if scene_splits.setdefault(group, row['split']) != row['split']:
                raise ValueError('Scene leakage')
            run = Path(row['run'])
            audit = audit_run(run)
            manifest = json.loads((run / 'manifest.json').read_text())
            if (manifest['source'] != job['source'] or manifest['source_seed'] != job['seed']
                    or manifest['chunk'] != job['chunk']):
                raise ValueError('Index/run identity mismatch')
            provenance.append({'run': str(run), 'manifest_sha256': digest(run / 'manifest.json'),
                               'result_sha256': digest(run / 'result.json')})
            with np.load(run / 'checkpoint.npz', allow_pickle=False) as data:
                start = data['context'].copy()
                base = data['base_actions'].copy()
            with np.load(run / 'transition_base.npz', allow_pickle=False) as data:
                base_visual = endpoint_rgb_features(data)
            caps = np.asarray(json.loads((run / 'proposal_spec.json').read_text())['caps'])
            action_scale = np.ones(14)
            action_scale[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]] = np.pi
            condition = np.r_[start, (base / action_scale).reshape(4, 8, 14).mean(axis=1).ravel()]
            for branch, summary in audit['branches'].items():
                if branch == 'base':
                    continue  # Avoid inflating metrics with trivially zero labels.
                if not summary['positive_time'] or not summary['full_horizon']:
                    excluded.append({'job': row['id'], 'branch': branch, 'reason': 'partial_or_zero_time'})
                    continue
                with np.load(run / f'transition_{branch}.npz', allow_pickle=False) as data:
                    target = endpoint_rgb_features(data) - base_visual
                    residual = (data['residual_actions'] / caps).reshape(4, 8, 14).mean(axis=1).ravel()
                contexts.append(condition)
                residuals.append(residual)
                targets.append(target)
                records.append({'task': task, 'group': group, 'job': row['id'],
                                'branch': branch, 'split': row['split']})
            print('AUDIT ' + row['id'], flush=True)
    if seen != set(planned):
        raise ValueError('Missing planned jobs')
    return (np.asarray(contexts), np.asarray(residuals), np.asarray(targets), records,
            {'jobs': len(seen), 'scene_splits': scene_splits, 'excluded': excluded, 'sources': provenance})


def weights(records):
    """Equal tasks, scenes within tasks, checkpoints within scenes, then branches."""
    tasks = {r['task'] for r in records}
    groups = {t: {r['group'] for r in records if r['task'] == t} for t in tasks}
    jobs = {g: {r['job'] for r in records if r['group'] == g} for gs in groups.values() for g in gs}
    count = Counter(r['job'] for r in records)
    return np.array([1 / (len(tasks) * len(groups[r['task']]) * len(jobs[r['group']]) * count[r['job']])
                     for r in records])


def shuffle_within_checkpoint(residual, records, seed):
    rng = np.random.default_rng(seed)
    shuffled = residual.copy()
    for job in sorted({r['job'] for r in records}):
        ids = np.array([i for i, r in enumerate(records) if r['job'] == job])
        if len(ids) > 1:
            perm = rng.permutation(len(ids))
            while np.any(perm == np.arange(len(ids))):
                perm = rng.permutation(len(ids))
            shuffled[ids] = residual[ids[perm]]
    return shuffled


def features(condition, residual, action=True):
    if not action:
        return condition
    r = np.c_[residual, residual ** 2]
    return np.c_[condition, r, (condition[:, :, None] * r[:, None, :]).reshape(len(r), -1)]


def fit_ridge(x, y, sample_weights, penalty):
    mean = (sample_weights[:, None] * x).sum(axis=0)
    std = np.maximum(np.sqrt((sample_weights[:, None] * (x - mean) ** 2).sum(axis=0)), .05)
    design = np.c_[np.ones(len(x)), (x - mean) / std]
    weighted = design * np.sqrt(sample_weights[:, None])
    coef = weighted.T @ np.linalg.solve(weighted @ weighted.T + penalty * np.eye(len(x)),
                                       y * np.sqrt(sample_weights[:, None]))
    return mean, std, coef


def predict(x, model):
    mean, std, coef = model
    return np.c_[np.ones(len(x)), (x - mean) / std] @ coef


def metrics(predicted, target, records):
    error = np.mean((predicted - target) ** 2, axis=1)
    result = {'macro_mse': float(weights(records) @ error), 'by_task': {}, 'by_scene': {}}
    for task in sorted({r['task'] for r in records}):
        ids = [i for i, r in enumerate(records) if r['task'] == task]
        result['by_task'][task] = float(weights([records[i] for i in ids]) @ error[ids])
    for group in sorted({r['group'] for r in records}):
        ids = [i for i, r in enumerate(records) if r['group'] == group]
        result['by_scene'][group] = float(weights([records[i] for i in ids]) @ error[ids])
    return result


def run_probe(context, residual, target, records, run, seeds=(0, 1, 2), penalty=.01):
    train = np.array([r['split'] == 'train' for r in records])
    valid = ~train
    tr = [r for r in records if r['split'] == 'train']
    va = [r for r in records if r['split'] == 'validation']
    if not train.any() or not valid.any() or not np.isfinite(np.c_[context, residual, target]).all():
        raise ValueError('Invalid learning data')
    w = weights(tr)
    cmean = (w[:, None] * context[train]).sum(axis=0)
    cstd = np.maximum(np.sqrt((w[:, None] * (context[train] - cmean) ** 2).sum(axis=0)), .05)
    ymean = (w[:, None] * target[train]).sum(axis=0)
    _, sv, vt = np.linalg.svd((target[train] - ymean) * np.sqrt(w[:, None]), full_matrices=False)
    components = vt[:16]
    y = (target[train] - ymean) @ components.T
    report = {'scope': 'developmental visual-effect learning probe; not object-direction M1 or controller',
              'target': 'endpoint RGB grid8(branch) minus RGB grid8(base), 576 values',
              'feature_inputs': 'current RGB/proprioception + frozen base actions + proposed residual; no task ID, GT, endpoint or realized duration',
              'train_rows': len(tr), 'validation_rows': len(va), 'ridge_penalty': penalty,
              'pca_training_variance_retained': float((sv[:16] ** 2).sum() / max((sv ** 2).sum(), 1e-30)),
              'zero': metrics(np.zeros_like(target[valid]), target[valid], va), 'seeds': []}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        projection = rng.normal(size=(context.shape[1], 8)) / np.sqrt(context.shape[1])
        condition = np.clip((context - cmean) / cstd, -5, 5) @ projection
        shuffled_train = shuffle_within_checkpoint(residual[train], tr, seed + 101)
        shuffled_valid = shuffle_within_checkpoint(residual[valid], va, seed + 202)
        row = {'seed': seed}
        for mode in ('action', 'no_action', 'shuffled_action'):
            rt = shuffled_train if mode == 'shuffled_action' else residual[train]
            x = features(condition[train], rt, mode != 'no_action')
            model = fit_ridge(x, y, w, penalty)
            xv = features(condition[valid], residual[valid], mode != 'no_action')
            prediction = predict(xv, model) @ components + ymean
            row[mode] = metrics(prediction, target[valid], va)
            row[mode]['train_mse'] = metrics(predict(x, model) @ components + ymean, target[train], tr)['macro_mse']
            if mode == 'action':
                row['action_shuffled_at_validation'] = metrics(
                    predict(features(condition[valid], shuffled_valid), model) @ components + ymean, target[valid], va)
            np.savez_compressed(run / f'{mode}_seed{seed}.npz',
                                context_mean=cmean, context_std=cstd, projection=projection,
                                target_mean=ymean, target_components=components,
                                feature_mean=model[0], feature_std=model[1], coefficient=model[2])
        report['seeds'].append(row)
        print(json.dumps(row), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    run = new_run('visual_probe', group=('effect_learning',))
    print(run, flush=True)
    write_json(run / 'manifest.json', {'plan': str(args.plan), 'plan_sha256': digest(args.plan),
                                      'code_sha256': digest(Path(__file__)), 'device': 'cpu',
                                      'seed_list': [0, 1, 2], 'penalty': .01,
                                      'selection': 'fixed settings, no validation tuning'})
    try:
        context, residual, target, records, audit = load_data(args.plan)
        write_json(run / 'data_audit.json', audit)
        write_json(run / 'rows.json', records)
        np.savez_compressed(run / 'features.npz', context=context, residual=residual, target=target)
        report = run_probe(context, residual, target, records, run)
        write_json(run / 'result.json', dict(status='complete', **report))
    except BaseException:
        import traceback
        write_json(run / 'failure.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
