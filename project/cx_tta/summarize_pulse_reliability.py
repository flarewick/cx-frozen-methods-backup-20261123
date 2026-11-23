"""Consolidate all fixed-horizon diagnostics without selecting a winning model."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import new_run, write_json
from .pulse_reliability_diagnostics import MAIN
from .effect_learning_probe import weights


def strata(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    dataset = Path(manifest['dataset'])
    records = json.loads((dataset / 'index.json').read_text())['records']
    with np.load(run / 'task_predictions.npz') as z:
        actual = z['target'][:, :3] * .01
        predicted = z['conditioned'][:, :3] * .01
    out = []
    for task, entity in MAIN.items():
        if task == 'turn_switch':
            continue
        for horizon in (4, 8, 12, 16, 20, 24):
            ids = np.array([i for i, r in enumerate(records) if r['task'] == task and r['entity'] == entity and r['horizon'] == horizon])
            y, p = actual[ids], predicted[ids]
            norms = np.linalg.norm(y, axis=1)
            w = weights([records[i] for i in ids])
            subgroups = []
            groups = [('under_0.5mm', norms < .0005), ('0.5_to_5mm', (norms >= .0005) & (norms <= .005)), ('over_5mm', norms > .005)]
            groups += [('scene_' + group, np.array([records[i]['group'] == group for i in ids])) for group in sorted({records[i]['group'] for i in ids})]
            for name, take in groups:
                mass = float(w @ take)
                if not mass:
                    subgroups.append(dict(name=name, rows=0))
                    continue
                zero = float(w[take] @ np.mean(y[take] ** 2, axis=1) / mass)
                err = float(w[take] @ np.mean((p[take] - y[take]) ** 2, axis=1) / mass)
                directional = take & (norms > .0005)
                prednorm = np.linalg.norm(p[directional], axis=1)
                cos = (p[directional] * y[directional]).sum(1) / np.maximum(prednorm * norms[directional], 1e-15)
                subgroups.append(dict(name=name, rows=int(take.sum()), scenes=len({records[i]['group'] for i in ids[take]}), zero_mse=zero, model_mse=err, improvement=1 - err / zero if zero > 1e-16 else None, direction_rows=len(cos), zero_predictions=int((prednorm < 1e-12).sum()), positive_direction_rows=int((cos > 0).sum()), median_cosine=float(np.median(cos)) if len(cos) else None))
            out.append(dict(task=task, horizon=horizon, rows=len(ids), effect_max_mm=float(norms.max() * 1000), strata=subgroups))
    return out


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', action='append', required=True, type=Path)
    p.add_argument('--evidence', action='append', required=True, type=Path)
    a = p.parse_args()
    out = new_run('pulse_reliability_summary', group=('reports', 'audits'))
    models = []
    original = None
    original_model_hash = None
    controls_identical = {}
    for model in a.model:
        result = json.loads((model / 'result.json').read_text())
        manifest = json.loads((model / 'manifest.json').read_text())
        selected = [r for r in result['reports'] if r['entity'] == MAIN.get(r['task'])]
        models.append(dict(run=str(model), result_sha256=digest(model / 'result.json'), manifest=manifest, reports=selected, translation_strata=strata(model)))
        if original is None:
            original = model
            original_model_hash = manifest['model_sha256']
        if manifest['model_sha256'] == original_model_hash:
            same = {}
            for unit in ('group', 'task'):
                with np.load(original / f'{unit}_predictions.npz') as x, np.load(model / f'{unit}_predictions.npz') as y:
                    same[unit] = bool(np.array_equal(x['action_only'], y['action_only'], equal_nan=True))
            controls_identical[str(model)] = same
    write_json(out / 'result.json', dict(status='complete', models=models, action_only_control_identity=controls_identical, evidence=[dict(path=str(path), sha256=digest(path)) for path in a.evidence], findings=dict(generalization_gate='not_passed', local_translation='Original model h4 positive relative to zero/action-only in three task holdouts; h8-h24 and rotation are not stable. Post-hoc developmental hypothesis, not new primary benchmark.', geometry='Oracle current/history 3D reduces h24 task-held error but still worse than zero, and worsens several scene-held results. No promotion.', confidence='Train-support distance is an uncalibrated proxy; fixed-quantile rejection does not recover robust h24 benefit and can drop strong effects.', next='Existing-data short-time translation vs delayed-effect factorization and grouped confidence validation; no collection, policy evaluation or TTA.'), scope='Scene-grouped development analysis on 16 original train scenes. All six horizons retained; threshold strata descriptive only. Not success rates, statistical significance or deployment evidence.'))
    print(out / 'result.json', flush=True)

if __name__ == '__main__':
    main()
