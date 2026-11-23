"""Fit-only whole-scene splits and fold-local observable preprocessing for S1y."""
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from .audit import digest
from .paths import ROOT, new_run, write_json
from .probe_candidates import read, verify
from .train_semantic_pilot import past_indices, balanced_weights

SOURCE = ROOT / 'outputs/effect_learning/coverage_20261002_092902_991797'
ARMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def ordered(groups):
    return sorted(groups, key=lambda g: hashlib.sha256(('s1y_scene_split_v1:' + g).encode()).hexdigest())


def load_raw(source=SOURCE):
    p = read(source / 'protocol.json')
    verify(p)
    for name in ('audit.json', 'coverage_audit.json'):
        if not read(source / name)['passed']:
            raise ValueError('Audited coverage source required')
    ids = sorted(set(p['parts']['expert_fit'] + p['parts']['anchor_fit'] + p['parts']['recovery']))
    rows = [dict(p['rows'][i], source_index=i) for i in ids]
    with np.load(source / 'data.npz') as z:
        arrays = {k: z[k][ids].copy() for k in ('base', 'target', 'valid')}
    lookup = {(r['group'], r['frame']): i for i, r in enumerate(rows)}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate group/frame')
    features = np.zeros((len(rows), 28672), np.float32)
    state = np.zeros((len(rows), 14), np.float32)
    covered = set()
    files = [source / f for f in ('protocol.json', 'data.npz', 'audit.json', 'coverage_audit.json')]
    ep = Path(p['expert_source']) / 'protocol.json'
    cache = Path(read(ep)['cache'])
    files += [ep, cache / 'manifest.json', cache / 'result.json']
    for record in read(cache / 'result.json')['records']:
        src = record['source']
        if src['pilot_split'] not in ('fit', 'anchor_fit'):
            continue
        f = Path(record['file'])
        if digest(f) != record['sha256']:
            raise ValueError('Changed feature file')
        with np.load(f) as z:
            joint, frames = z['joint'].astype(np.float32), z['frames']
        files.append(f)
        if src['kind'] == 'expert':
            f = Path(src['cache'])
            if digest(f) != src['cache_sha256']:
                raise ValueError('Changed expert command cache')
            with np.load(f) as z:
                if not np.array_equal(frames, z['frame_index']):
                    raise ValueError('Misaligned expert features')
                states = z['context'][:, :14].copy()
                base, target, valid = z['base_action'], z['target'], z['valid']
        else:
            f = Path(src['checkpoint'])
            if digest(f) != src['checkpoint_sha256']:
                raise ValueError('Changed anchor checkpoint')
            with np.load(f) as z:
                states = z['command_state'][None].astype(np.float32)
                states[:, ARMS] /= np.pi
                base = z['base_actions'][None].copy()
            target, valid = base.copy(), np.ones((1, 32), bool)
        files.append(f)
        for j, frame in enumerate(frames):
            i = lookup[(src['group'], int(frame))]
            if i in covered:
                raise ValueError('Duplicate raw feature source')
            features[i], state[i] = joint[j], states[j]
            for key, value in [('base', base), ('target', target), ('valid', valid)]:
                np.testing.assert_array_equal(arrays[key][i], value[j])
            covered.add(i)
    ca = Path(p['context_source']) / 'audit.json'
    recovery = Path(read(ca)['source'])
    rp = read(recovery / 'protocol.json')
    files += [ca, recovery / 'protocol.json', recovery / 'features.npz', recovery / 'data.npz']
    with np.load(recovery / 'features.npz') as z:
        joint, states = z['joint'].astype(np.float32), z['state'].astype(np.float32)
    states[:, ARMS] /= np.pi
    for j, row in enumerate(rp['rows']):
        i = lookup[(row['group'], row['native_count'])]
        if i in covered:
            raise ValueError('Recovery feature overlap')
        features[i], state[i] = joint[j], states[j]
        covered.add(i)
    if len(covered) != len(rows) or len(rows) != 880:
        raise ValueError('Incomplete fit-only coverage')
    history = past_indices(rows)
    old_to_new = {v: i for i, v in enumerate(ids)}
    pairs = [dict(e, before=old_to_new[e['before']], after=old_to_new[e['after']]) for e in p['seam_pairs']]
    return p, rows, arrays, features, state, history, pairs, files


def layout(rows):
    roles, folds = {}, {}
    experts = defaultdict(set)
    anchors, recovery = set(), set()
    for r in rows:
        if r['kind'] == 'expert':
            experts[r['task']].add(r['group'])
        elif r['kind'] == 'anchor':
            anchors.add(r['group'])
        else:
            recovery.add(r['group'])
    for task, groups in sorted(experts.items()):
        if len(groups) != 12:
            raise ValueError('Expected twelve original fit scenes per task')
        for j, group in enumerate(ordered(groups)):
            roles[group] = 'gate_fit' if j < 8 else ('calibration' if j < 10 else 'validation')
            if j < 8:
                folds[group] = j % 4
    # Only one task has >=2 fit anchors. Reserve one each for calibration and validation.
    at = defaultdict(set)
    for g in anchors:
        at[g.split(':')[0]].add(g)
    eligible = sorted(t for t, gs in at.items() if len(gs) >= 2)
    if len(eligible) != 1:
        raise ValueError('Anchor allocation contract changed')
    held_anchors = ordered(at[eligible[0]])[:2]
    for j, group in enumerate(held_anchors):
        roles[group] = ('calibration', 'validation')[j]
    for j, group in enumerate(ordered(anchors - set(held_anchors))):
        roles[group], folds[group] = 'gate_fit', j % 4
    for j, group in enumerate(ordered(recovery)):
        roles[group], folds[group] = 'gate_fit', j % 4
    parts = {role: [i for i, r in enumerate(rows) if roles[r['group']] == role]
             for role in ('gate_fit', 'calibration', 'validation')}
    jobs = []
    for f in range(4):
        train = [i for i in parts['gate_fit'] if folds[rows[i]['group']] != f]
        predict = [i for i in parts['gate_fit'] if folds[rows[i]['group']] == f]
        jobs.append(dict(name=f'inner{f}', train=train, predict=predict))
    jobs.append(dict(name='outer', train=parts['gate_fit'], predict=parts['calibration'] + parts['validation']))
    return parts, roles, folds, jobs


def fit_preprocess(features, state, history, fit):
    fit = np.asarray(fit, int)
    pca = PCA(n_components=64, svd_solver='randomized', random_state=0).fit(features[fit])
    visual = pca.transform(features).astype(np.float32)
    current = np.c_[visual, state]
    past = np.zeros_like(current)
    has = history >= 0
    past[has] = current[history[has]]
    combined = np.c_[current, past, has.astype(np.float32)]
    mean, std = combined[fit].mean(0), np.maximum(combined[fit].std(0), .05)
    context = ((combined - mean) / std).astype(np.float32)
    pre = dict(pca_mean=pca.mean_, pca_components=pca.components_, context_mean=mean, context_std=std,
               fit_indices=fit, history_indices=history)
    return context, pre


def component_weights(rows, indices):
    out = np.zeros(len(rows), np.float32)
    for kind in ('expert', 'anchor', 'recovery'):
        ix = [i for i in indices if rows[i]['kind'] == kind]
        if not ix:
            raise ValueError('A candidate training fold lost an entire component')
        out[ix] = balanced_weights([rows[i] for i in ix]).numpy().astype(np.float32)
    return out


def prepare():
    parent, rows, arrays, features, state, history, pairs, files = load_raw()
    parts, roles, folds, jobs = layout(rows)
    run = new_run('scene_crossfit', group=('effect_learning',))
    print(run, flush=True)
    np.savez_compressed(run / 'data.npz', **arrays, state=state, history=history)
    for job in jobs:
        fit = [i for i in job['train'] if rows[i]['kind'] == 'expert']
        context, pre = fit_preprocess(features, state, history, fit)
        weights = component_weights(rows, job['train'])
        f = run / (job['name'] + '_preprocess.npz')
        np.savez_compressed(f, **pre)
        c = run / (job['name'] + '_context.npz')
        np.savez_compressed(c, context=context, weights=weights)
        job.update(preprocess=str(f), context=str(c), preprocess_fit=fit)
        files.extend([f, c])
        print('PREPROCESS', job['name'], 'train', len(job['train']), 'predict', len(job['predict']), flush=True)
    files += [run / 'data.npz', Path(__file__), ROOT / 'cx_tta/train_scene_crossfit.py',
              ROOT / 'scripts/train_scene_crossfit.sh']
    protocol = dict(status='frozen_before_training', source=str(SOURCE), rows=rows, parts=parts,
        group_roles=roles, group_folds=folds, jobs=jobs, seam_pairs=pairs, config=parent['config'],
        source_sha256={**parent['source_sha256'], **{str(f): digest(f) for f in files}},
        encodings=['repeat', 'fourier'], seeds=[0, 1, 2], updates=1400, lr=.0003, parameters=41742,
        context_dim=157, rotation_scale_rad=.1, position_scale_m=.02,
        design='Four inner candidate folds on gate-fit scenes plus outer fit predicting calibration/validation. '
               'Each task:8 expert gate-fit/2 calibration/2 validation.3 anchor gate-fit,1 cup calibration,1 cup validation. '
               'Both recovery scenes gate-fit, assigned different inner folds, so all gate supervision is candidate-out-of-scene.',
        preprocessing='Each candidate PCA/scaler fits its training expert rows only. All current/past observations move '
                      'with complete scenes. Router uses ONE separate coordinate system: outer preprocessing fit on '
                      'gate-fit expert observations only; inner PCA coordinates never pooled as router features.',
        training='Same S1v recovery loss: separate task/scene-balanced expert,anchor,recovery components, native BC+SE3 '
                 '+within/boundary increments, gripper envelope; final1400 only, no checkpoint selection. '
                 'Each surviving recovery boundary is balanced within surviving recovery tasks/scenes. No held targets in loss.',
        probe_design=dict(models=['linear', 'mlp32'], seeds=[0, 1, 2], updates=600, lr=.001, batch_size=512,
            labels='Per executed single candidate seed, component, alpha .25/.5/1: strict5%benefit; '
                   'continuous normalized gain (base-new)/(1+base) clipped[-5,1]; risk exceeds max(base,existing1deg/1cm/.02) or worsens range. '
                   'No3seed consensus-label target. Raw and original-cap projections separate.',
            inputs='Outer-coordinate157D legal context; side/component/alpha; full32step base and candidate per-arm '
                   'command summaries; other-two-model spread and relative agreement, amplitude separately. No labels/masks/taskID.',
            fitting='One task-shared probe per encoding/support/architecture/init, pooled executed candidate seeds with '
                    'equal mass, independent gain/benefit/risk heads. Train only inner OOF gate-fit examples. Fixed budget, no tuning.',
            calibration='Choose gain-threshold among0/.01/.05/.1/.2 and risk-cutoff.05/.1/.2 using calibration only, '
                        'maximize balanced selected normalized gain subject to harm<=.05 over all expert arm/component units '
                        'and zero tolerant harm on calibration anchors. Include always-off fallback; never choose on validation.',
            evaluation='Frozen validation expert8scenes and cup anchor1scene; per candidate seed and task. Compare base, '
                       'fixed.25/.5/1, teacher-known potential, and fixed observable magnitude/agreement ranking. '
                       'Report weighted AUC, actual selected gain/harm/coverage and command/pose metrics. '
                       'Three probe/candidate seeds not independent scenes; no formal promotion from this small diagnostic.'),
        scope='S1y nested FIT-pool diagnostic, not a pristine final test: source scenes were used by past S1v models, '
              'but new candidate/probe pipelines exclude their declared held scenes. Old12expert/2anchor developer '
              'sets are not loaded or evaluated. Only2 recovery scenes; no independent held recovery or task-success claims. '
              '30 small candidate models, no VLA inference/simulation/new collection/TTA. Unbounded candidate versus explicit cap projection.')
    write_json(run / 'protocol.json', protocol)
    print('PREREGISTERED', run, flush=True)
    return run
