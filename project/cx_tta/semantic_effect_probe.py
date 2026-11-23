"""Oracle entity-associated readouts: diagnose features, NOT a deployable router.

Entity/task keys select supervised readouts only. They are not numeric model
inputs, but that association is privileged and must not be advertised as shared
entity discovery or unseen-task control. All hyperparameters use train scenes.
"""
import argparse
import json
from pathlib import Path
import traceback

import numpy as np

from .audit import digest
from .effect_learning_probe import weights, shuffle_within_checkpoint
from .paths import ROOT, new_run, write_json

ALPHAS = (1., .1, .01, .001, .0001)  # Prefer more regularization on exact ties.


def squared_distances(x):
    gram = x @ x.T
    return np.maximum(np.diag(gram)[:, None] + np.diag(gram)[None] - 2 * gram, 0)


def rbf(distances, train):
    d = distances[np.ix_(train, train)]
    positive = d[d > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.
    return np.exp(-distances / max(scale, 1e-12)), scale


def kernels(vdist, pdist, action, train, mode):
    kv, sv = rbf(vdist, train)
    kp, sp = rbf(pdist, train)
    kc = (kv + kp) / 2
    if mode == 'vision':
        return kv, (sv, sp, 1.)
    if mode == 'proprio':
        return kp, (sv, sp, 1.)
    if mode in ('no_action', 'combined'):
        return kc, (sv, sp, 1.)
    a = np.c_[action, action ** 2]
    ka = a @ a.T
    scale = max(float(np.diag(ka)[train].mean()), 1e-12)
    ka /= scale
    return (ka if mode == 'action_only' else kc * ka), (sv, sp, scale)


def fit_predict(kernel, y, records, train, test, alpha, centered):
    w = weights([records[i] for i in train])
    mean = w @ y[train] if centered else np.zeros(y.shape[1])
    sw = np.sqrt(w)
    k = kernel[np.ix_(train, train)] * sw[:, None] * sw[None, :]
    dual = np.linalg.solve(k + alpha * np.eye(len(train)), (y[train] - mean) * sw[:, None])
    return kernel[np.ix_(test, train)] @ (sw[:, None] * dual) + mean, (dual, w, mean)


def scene_mse(pred, y, records):
    return float(weights(records) @ np.mean((pred - y) ** 2, axis=1))


def select_alpha(vdist, pdist, action, y, records, train, mode, centered):
    scores = {a: [] for a in ALPHAS}
    groups = sorted({records[i]['group'] for i in train})
    if len(groups) < 2:
        raise ValueError('At least two training scenes required for CV')
    for group in groups:
        fit = np.array([i for i in train if records[i]['group'] != group])
        held = np.array([i for i in train if records[i]['group'] == group])
        kernel, _ = kernels(vdist, pdist, action, fit, mode)
        for alpha in ALPHAS:
            prediction, _ = fit_predict(kernel, y, records, fit, held, alpha, centered)
            scores[alpha].append(scene_mse(prediction, y[held], [records[i] for i in held]))
    means = {a: float(np.mean(value)) for a, value in scores.items()}
    best = min(ALPHAS, key=lambda a: means[a])
    return best, means


def load_entries(cache):
    result = json.loads((cache / 'result.json').read_text())
    if result['status'] != 'complete' or result['limited_probe']:
        raise ValueError('Full feature cache required')
    entries, seen, groups = [], set(), {}
    for row in result['files']:
        job = row['job']
        if job['id'] in seen:
            raise ValueError('Duplicate job')
        seen.add(job['id'])
        group = f"{job['task']}:{job['seed']}"
        if groups.setdefault(group, job['split']) != job['split']:
            raise ValueError('Scene leakage')
        for key in ('features', 'offline_labels'):
            if digest(Path(row[key])) != row[key + '_sha256']:
                raise ValueError('Cache checksum mismatch')
        with np.load(row['features'], allow_pickle=False) as data:
            vision = data['pooled'].astype(np.float64).reshape(3, -1)
            vision /= np.maximum(np.linalg.norm(vision, axis=1, keepdims=True), 1e-12)
            command = data['command_state'].astype(float)
            base = data['base_actions'].astype(float)
        scale = np.ones(14)
        scale[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]] = np.pi
        labels = json.loads(Path(row['offline_labels']).read_text())
        source = Path(job['run'])
        caps = np.array(json.loads((source / 'proposal_spec.json').read_text())['caps'])
        residuals = {}
        for branch in labels['branches']:
            if branch == 'base':
                continue
            with np.load(source / f'transition_{branch}.npz', allow_pickle=False) as data:
                residuals[branch] = (data['residual_actions'] / caps).reshape(4, 8, 14).mean(axis=1).ravel()
        entries.append(dict(job=job, group=group, vision=vision.ravel(), state=command / scale,
                            condition=np.r_[command / scale, (base / scale).reshape(4, 8, 14).mean(axis=1).ravel()],
                            labels=labels, residuals=residuals))
    return entries


def make_groups(entries, phase):
    """Separate entity-associated outputs, never concatenate GT into features."""
    groups = {}
    for entry in entries:
        job, labels = entry['job'], entry['labels']
        rows = [('before', labels['before_offline_only'])] if phase == 'state' else list(labels['branches'].items())
        for branch, value in rows:
            if phase == 'effect' and (branch == 'base' or not value['full_horizon'] or value['duration_s'] <= 0):
                continue
            if phase == 'state':
                targets = [('position_m', name, np.asarray(pose[:3])) for name, pose in value['object_poses'].items()]
                targets += [('joint_native', name, np.asarray(q)) for name, q in value['articulation_qpos'].items()]
            else:
                value = value['versus_base']
                targets = [(kind, name, np.asarray(x[key])) for name, x in value['rigid'].items()
                           for kind, key in [('translation_m', 'translation_world_m'), ('rotation_rad', 'rotation_world_rad')]]
                targets += [('joint_fraction', name, np.asarray(x['delta_range_fraction'])[x['range_valid_mask']])
                            for name, x in value['articulated'].items() if any(x['range_valid_mask'])]
            for kind, entity, target in targets:
                key = (job['task'], entity, kind)
                groups.setdefault(key, []).append(dict(
                    vision=entry['vision'], condition=entry['state'] if phase == 'state' else entry['condition'],
                    action=np.zeros(56) if phase == 'state' else entry['residuals'][branch], target=target,
                    record={'task': job['task'], 'group': entry['group'], 'job': job['id'],
                            'split': job['split'], 'branch': branch}))
    return groups


def evaluate_group(key, rows, phase, folder):
    records = [r['record'] for r in rows]
    y = np.stack([r['target'] for r in rows])
    vision = np.stack([r['vision'] for r in rows])
    condition = np.stack([r['condition'] for r in rows])
    action = np.stack([r['action'] for r in rows])
    train = np.array([i for i, r in enumerate(records) if r['split'] == 'train'])
    valid = np.array([i for i, r in enumerate(records) if r['split'] == 'validation'])
    vdist, pdist = squared_distances(vision), squared_distances(condition)
    va = [records[i] for i in valid]
    centered = phase == 'state'
    mean = weights([records[i] for i in train]) @ y[train] if centered else np.zeros(y.shape[1])
    baseline = np.broadcast_to(mean, y[valid].shape)
    report = {'task': key[0], 'entity': key[1], 'target': key[2], 'train_rows': len(train),
              'validation_rows': len(valid), 'baseline': 'train_mean' if centered else 'zero_effect',
              'baseline_mse': scene_mse(baseline, y[valid], va),
              'training_target_rms': float(np.sqrt(np.mean((y[train] - mean) ** 2))), 'models': {}}
    modes = ('vision', 'proprio', 'combined') if centered else ('no_action', 'action_only', 'conditioned', 'shuffled')
    if np.max(np.abs(y[train] - mean)) < 1e-10:
        report['constant_training_target'] = True
        return report
    for mode in modes:
        a = action.copy()
        if mode == 'shuffled':
            a[train] = shuffle_within_checkpoint(a[train], [records[i] for i in train], 1201)
        alpha, scores = select_alpha(vdist, pdist, a, y, records, train, mode, centered)
        kernel, scales = kernels(vdist, pdist, a, train, mode)
        prediction, fitted = fit_predict(kernel, y, records, train, valid, alpha, centered)
        model_report = {'alpha': alpha, 'train_scene_cv_mse': scores,
                        'validation_mse': scene_mse(prediction, y[valid], va),
                        'per_scene_mse': {g: scene_mse(prediction[[i for i, r in enumerate(va) if r['group'] == g]],
                                                     y[valid][[i for i, r in enumerate(va) if r['group'] == g]],
                                                     [r for r in va if r['group'] == g]) for g in sorted({r['group'] for r in va})}}
        report['models'][mode] = model_report
        if mode == 'conditioned':
            permuted = action.copy()
            permuted[valid] = shuffle_within_checkpoint(action[valid], va, 2201)
            permuted_kernel, _ = kernels(vdist, pdist, permuted, train, mode)
            alternate, _ = fit_predict(permuted_kernel, y, records, train, valid, alpha, centered)
            model_report['validation_action_shuffled_mse'] = scene_mse(alternate, y[valid], va)
        np.savez_compressed(folder / (mode + '.npz'), dual=fitted[0], weights=fitted[1], mean=fitted[2],
                            train_row_indices=train, train_condition=condition[train],
                            train_action=a[train], kernel_scales=scales, alpha=alpha,
                            prediction=prediction, target=y[valid])
    write_json(folder / 'rows.json', records)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'outputs/feature_caches/siglip_initial_v1_gpu')
    args = parser.parse_args()
    run = new_run('entity_probe', group=('effect_learning',))
    print(run, flush=True)
    write_json(run / 'manifest.json', {'cache': str(args.cache), 'cache_index_sha256': digest(args.cache / 'result.json'),
                                      'code_sha256': digest(Path(__file__)), 'alphas': ALPHAS,
                                      'validation_tuning': False, 'scope': 'oracle entity-associated readouts, not deployable shared M1'})
    try:
        entries = load_entries(args.cache)
        result = {'status': 'complete', 'state': [], 'effect': []}
        for phase in ('state', 'effect'):
            for number, (key, rows) in enumerate(sorted(make_groups(entries, phase).items())):
                folder = run / f'{phase}_{number:03d}'
                folder.mkdir()
                report = evaluate_group(key, rows, phase, folder)
                report['artifact'] = str(folder)
                result[phase].append(report)
                print(phase, key, 'baseline', report['baseline_mse'],
                      {k: v['validation_mse'] for k, v in report['models'].items()}, flush=True)
        write_json(run / 'result.json', result)
    except BaseException:
        write_json(run / 'failure.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
