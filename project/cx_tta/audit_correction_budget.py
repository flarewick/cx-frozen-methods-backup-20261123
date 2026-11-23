"""Offline box-oracle diagnostic; expert labels are never a deployable controller."""
import argparse
import json
from pathlib import Path

import numpy as np

from .audit import digest
from .paths import new_run, write_json

CAPS = np.array([.1] * 6 + [.2] + [.1] * 6 + [.2], dtype=np.float64)


def oracle_and_errors(base, target, predicted, valid, caps=CAPS):
    base, target, predicted = [np.asarray(x, dtype=np.float64) for x in (base, target, predicted)]
    valid = np.asarray(valid)
    if base.shape != target.shape or base.shape != predicted.shape or base.ndim != 3 or base.shape[-1] != 14:
        raise ValueError('Expected matching [N,H,14] actions')
    if valid.shape != base.shape[:2] or valid.dtype != bool or not valid.any(1).all():
        raise ValueError('Every query needs a nonempty boolean validity mask')
    if not all(np.isfinite(x[valid]).all() for x in (base, target, predicted)):
        raise ValueError('Nonfinite valid action')
    # Tanh reaches the open box: projection onto its closure is an optimistic infimum.
    oracle = base + np.clip(target - base, -caps, caps)
    def error(action):
        squared = np.where(valid[..., None], ((action - target) / caps) ** 2, 0)
        return squared.sum((1, 2)) / (14 * valid.sum(1))
    return oracle, dict(base=error(base), oracle=error(oracle), model=error(predicted))


def balanced(values, rows):
    groups = sorted({r['group'] for r in rows})
    return float(np.mean([np.mean(values[[i for i, r in enumerate(rows) if r['group'] == g]]) for g in groups]))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    source = args.run.resolve()
    protocol = json.loads((source / 'protocol.json').read_text())
    result = json.loads((source / 'result.json').read_text())
    if result['status'] != 'complete':
        raise ValueError('Incomplete pilot')
    out = new_run('correction_budget', group=('reports', 'audits'))
    write_json(out / 'protocol.json', dict(source=str(source), source_protocol_sha256=digest(source/'protocol.json'),
        source_result_sha256=digest(source/'result.json'), caps=CAPS.tolist(), code_sha256=digest(Path(__file__)),
        scope='Previously consumed inner expert holdout only; no original validation, fitting, cap search or rollout. Box projection minimizes separable normalized action MSE, not physical/task error. No temporal, IK or dynamics constraints: optimistic action-space bound.'))
    rows = [protocol['rows'][i] for i in protocol['partitions']['inner_held']]
    if any(r['kind'] != 'expert' or r['pilot_split'] != 'inner_held' for r in rows):
        raise ValueError('Unexpected partition')
    reports, reference = [], None
    for report in result['reports']:
        variant = report['variant']
        path = source / f'{variant}_inner_held.npz'
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        if data['indices'].tolist() != protocol['partitions']['inner_held']:
            raise ValueError('Row alignment mismatch')
        if reference is None:
            reference = data
        elif any(not np.array_equal(data[k], reference[k]) for k in ('base', 'target', 'valid', 'indices')):
            raise ValueError('Unpaired model comparison')
        base, target, pred, valid = (data[k] for k in ('base', 'target', 'predicted', 'valid'))
        delta = pred.astype(float) - base
        if np.any(np.abs(delta[valid]) > CAPS + 1e-6):
            raise ValueError('Model exceeds declared cap')
        _, errors = oracle_and_errors(base, target, pred, valid)
        by_task = {}
        for task in sorted({r['task'] for r in rows}):
            ids = [i for i, r in enumerate(rows) if r['task'] == task]
            tr = [rows[i] for i in ids]
            e = {k: balanced(v[ids], tr) for k, v in errors.items()}
            if not np.isclose(e['model'], report['metrics']['inner_held'][task], rtol=2e-5, atol=1e-8):
                raise ValueError('Saved metric mismatch')
            if e['oracle'] > min(e['model'], e['base']) + 1e-8:
                raise ValueError('Oracle lower-bound violation')
            room = e['base'] - e['oracle']
            available = np.abs(target[ids].astype(float) - base[ids])
            mask = valid[ids][..., None]
            over_cap = ((available > CAPS) & mask).sum((1, 2)) / (valid[ids].sum(1) * 14)
            harmful = (((delta[ids] * (target[ids] - base[ids])) < 0) & mask).sum((1, 2)) / (valid[ids].sum(1) * 14)
            by_task[task] = dict(episodes=len({r['group'] for r in tr}), queries=len(ids), normalized_mse=e,
                oracle_improvement_fraction=room/e['base'] if e['base'] else None,
                model_improvement_fraction=(e['base']-e['model'])/e['base'] if e['base'] else None,
                fraction_of_oracle_gain_realized=(e['base']-e['model'])/room if room > 1e-12 else None,
                target_coordinates_beyond_cap_fraction=balanced(over_cap, tr),
                opposite_expert_correction_sign_fraction=balanced(harmful, tr))
        reports.append(dict(variant=variant, predictions_sha256=digest(path), tasks=by_task))
    write_json(out/'result.json', dict(status='complete', reports=reports, all_models_paired=True,
        metric_recomputed=True, oracle_bound_checked=True,
        limits='Expert-state imitation only. Unused original validation stays unopened. The oracle uses future expert labels and is not deployable; same-state recovery targets in failed policy states remain missing. No cap change is selected from this diagnostic.'))
    print(json.dumps(dict(output=str(out), reports=reports), indent=2), flush=True)


if __name__ == '__main__':
    main()
