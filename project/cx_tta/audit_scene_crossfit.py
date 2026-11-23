"""Actual-source scene/preprocessing/weight and saved prediction audit for S1y."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from .paths import write_json
from .audit import digest
from .probe_candidates import read, verify
from .scene_crossfit_data import load_raw
from .time_encoding import TimeEncodedAdapter


def preflight(run):
    p = read(run / 'protocol.json')
    verify(p)
    parent, rows, data, features, state, history, pairs, files = load_raw(Path(p['source']))
    if digest(Path(p['source']) / 'data.npz') != parent['data_sha256']:
        raise ValueError('Source data differs from original training protocol')
    if rows != p['rows'] or pairs != p['seam_pairs']:
        raise ValueError('Raw source row/seam mismatch')
    old_held = set(parent['parts']['inner_held'] + parent['parts']['anchor_held'])
    assert not old_held & {r['source_index'] for r in rows}
    with np.load(run / 'data.npz') as z:
        for k, v in dict(**data, state=state, history=history).items():
            np.testing.assert_array_equal(z[k], v)
    counts = Counter()
    maxima = dict(context=0., components=0., weights=0.)
    audits = []
    for job in p['jobs']:
        ti, pi = set(job['train']), set(job['predict'])
        tg, pg = ({rows[i]['group'] for i in ids} for ids in (ti, pi))
        if tg & pg:
            raise ValueError('Full-scene training/prediction overlap')
        for i in ti | pi:
            if history[i] >= 0 and ((i in ti) != (history[i] in ti)):
                raise ValueError('History crosses candidate fold')
        expected_fit = np.array(sorted(i for i in ti if rows[i]['kind'] == 'expert'))
        with np.load(job['preprocess']) as z:
            pre = {k: z[k].copy() for k in z.files}
        np.testing.assert_array_equal(pre['fit_indices'], expected_fit)
        independent = PCA(n_components=64, svd_solver='randomized', random_state=0).fit(features[expected_fit])
        maximum = float(np.max(np.abs(independent.components_ - pre['pca_components'])))
        maxima['components'] = max(maxima['components'], maximum)
        if maximum > 1e-6:
            raise ValueError('PCA was not fitted on declared fold-only raw features')
        visual = independent.transform(features).astype(np.float32)
        current = np.c_[visual, state]
        past = np.zeros_like(current)
        past[history >= 0] = current[history[history >= 0]]
        unscaled = np.c_[current, past, (history >= 0).astype(np.float32)]
        mean = unscaled[expected_fit].mean(0)
        std = np.maximum(unscaled[expected_fit].std(0), .05)
        np.testing.assert_array_equal(pre['context_mean'], mean)
        np.testing.assert_array_equal(pre['context_std'], std)
        expected_context = ((unscaled - mean) / std).astype(np.float32)
        weights = np.zeros(len(rows), np.float32)
        for kind in ('expert', 'anchor', 'recovery'):
            ki = [i for i in ti if rows[i]['kind'] == kind]
            tasks = {rows[i]['task'] for i in ki}
            gc = Counter(rows[i]['group'] for i in ki)
            for i in ki:
                r = rows[i]
                ng = len({rows[j]['group'] for j in ki if rows[j]['task'] == r['task']})
                weights[i] = 1 / (len(tasks) * ng * gc[r['group']])
        with np.load(job['context']) as z:
            error = float(np.abs(z['context'] - expected_context).max())
            maxima['context'] = max(maxima['context'], error)
            maxima['weights'] = max(maxima['weights'], float(np.abs(z['weights'] - weights).max()))
            if error > 2e-4:
                raise ValueError('Fold context mismatch')
            np.testing.assert_array_equal(z['weights'], weights)
        counts.update(pi)
        audits.append(dict(job=job['name'], train_rows=len(ti), predict_rows=len(pi),
                           train_scenes=len(tg), predict_scenes=len(pg), preprocess_expert_rows=len(expected_fit)))
        print('PREFLIGHT', job['name'], flush=True)
    if set(counts) != set(range(len(rows))) or set(counts.values()) != {1}:
        raise ValueError('Predicted rows must form an exact partition, including outer held')
    verify(p)
    write_json(run / 'preflight_audit.json', dict(passed=True, rows=len(rows), jobs=audits, maxima=maxima,
        old_developer_rows_excluded=True, fold_pca_refits=5, source_sha256={str(run/'protocol.json'):digest(run/'protocol.json'),
        str(Path(__file__)):digest(Path(__file__))}))


def predictions(run):
    p, result = read(run/'protocol.json'), read(run/'result.json')
    verify(p)
    if not read(run/'preflight_audit.json')['passed'] or not read(run/'interface_checks.json')['passed']:
        raise ValueError('Preflight and actual interface checks required')
    with np.load(run/'data.npz') as z:
        base = z['base'].copy()
    contexts = {}
    for job in p['jobs']:
        with np.load(job['context']) as z:
            contexts[job['name']] = z['context'].copy()
    seen, maximum = set(), 0.
    for report in result['reports']:
        key = report['job'], report['encoding'], report['seed']
        if key in seen:
            raise ValueError('Duplicate model')
        seen.add(key)
        job = next(j for j in p['jobs'] if j['name'] == report['job'])
        for key, hashkey in [('file','sha256'), ('checkpoint','checkpoint_sha256')]:
            if digest(Path(report[key])) != report[hashkey]:
                raise ValueError('Changed saved model/prediction')
        with np.load(report['file']) as z:
            pred, ids = z['predicted'].copy(), z['indices'].copy()
        np.testing.assert_array_equal(ids, job['predict'])
        model = TimeEncodedAdapter(157, encoding=report['encoding'], envelope=True)
        model.load_state_dict(torch.load(report['checkpoint'], map_location='cpu', weights_only=True))
        model.eval()
        with torch.no_grad():
            values = np.concatenate([model(torch.from_numpy(contexts[job['name']][ii]), torch.from_numpy(base[ii])).numpy()
                for start in range(0,len(ids),64) for ii in [ids[start:start+64]]])
        difference = float(np.abs(values-pred).max())
        maximum = max(maximum, difference)
        if difference > 2e-5:
            raise ValueError('CPU checkpoint prediction mismatch')
        lo, hi = np.minimum(base[ids][...,[6,13]],0), np.maximum(base[ids][...,[6,13]],1)
        if np.any(pred[...,[6,13]] < lo-1e-6) or np.any(pred[...,[6,13]] > hi+1e-6):
            raise ValueError('Gripper envelope mismatch')
    if len(seen) != 30:
        raise ValueError('Incomplete30model matrix')
    verify(p)
    write_json(run/'candidate_audit.json',dict(passed=True, models=30,cpu_prediction_max_error=maximum,
        source_sha256={**{str(run/f):digest(run/f) for f in ['protocol.json','preflight_audit.json','interface_checks.json','result.json']},
                       str(Path(__file__)):digest(Path(__file__))},
        scope='Input/numerical audit only; no independent recovery or task-success claim.'))
    print('CANDIDATE_AUDIT_PASS',maximum,flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    threadpool_limits(2)
    if args.preflight_only:
        preflight(args.run.resolve())
    else:
        predictions(args.run.resolve())


if __name__=='__main__':main()
