"""Task-shared, component-specific S1y probe inputs and offline labels.

Only observable_features is a forward-input builder. No target, tail mask,
source role, task name or scene identity is accepted by that function.
"""
from collections import Counter
import numpy as np
import torch
from torch import nn
from .candidate_diagnostic import CAPS, ALPHAS, SUPPORTS, fixed_actions, row_metrics
from .component_selector import component_grid, benefit_flags
from .recovery_fit_metrics import endpoints


def command_stats(x):
    return np.concatenate([x[:, 0], x[:, 15], x[:, -1], x.mean(1), x.std(1)], axis=-1)


def observable_features(context, base, proposals, support):
    if context.shape != (len(base), 157) or proposals.shape != (3, *base.shape):
        raise ValueError('Expected current/causal157D context and three full32step candidates')
    if support not in ('raw', 'capped'):
        raise ValueError('Unknown support')
    n = len(base)
    delta = proposals.astype(float) - base[None]
    if support == 'capped':
        delta = np.clip(delta, -CAPS, CAPS)
    native = base.reshape(n, 32, 2, 7) / CAPS.reshape(2, 7)
    norm = delta.reshape(3, n, 32, 2, 7) / CAPS.reshape(2, 7)
    features, magnitude, agreement = [], [], []
    for seed in range(3):
        others = [i for i in range(3) if i != seed]
        fs = np.empty((n, 2, 2, 3, 234), np.float32)
        mags, ags = np.empty((n, 2, 2, 3)), np.empty((n, 2, 2, 3))
        for side in range(2):
            bs = command_stats(native[:, :, side])
            ds = command_stats(norm[seed, :, :, side])
            for component, channel in enumerate((slice(0, 6), slice(6, 7))):
                z = norm[others, :, :, side, channel]
                spread = np.sqrt(z.var(0).mean((1, 2)))
                other_amplitude = np.sqrt((z.mean(0)**2).mean((1, 2)))
                own_amplitude = np.sqrt((norm[seed, :, :, side, channel]**2).mean((1, 2)))
                relative = -spread / (other_amplitude + 1e-6)
                extra = np.c_[spread, other_amplitude, own_amplitude, relative]
                for ai, alpha in enumerate(ALPHAS[1:]):
                    fs[:, side, component, ai] = np.c_[context, bs, ds * alpha, extra,
                        np.full((n, 1), alpha), np.full((n, 1), side), np.full((n, 1), component)]
                    mags[:, side, component, ai] = other_amplitude * alpha
                    ags[:, side, component, ai] = relative
        features.append(fs)
        magnitude.append(mags)
        agreement.append(ags)
    return np.stack(features), dict(magnitude=np.stack(magnitude), relative_agreement=np.stack(agreement))


def labels(base, target, valid, proposals, support, robots):
    target_pose = endpoints(target.astype(float), robots)
    output = {key: [] for key in ('gain', 'benefit', 'harm')}
    for seed in range(3):
        fixed = fixed_actions(base, proposals[seed])
        metric = [row_metrics(x, target, valid, robots, target_pose) for x in fixed]
        grid = component_grid(fixed, metric, target, valid, support)
        flags = benefit_flags(grid)
        gain, positive, harm = [], [], []
        for component in ('arm', 'grip'):
            indices = [a * 4 for a in range(4)] if component == 'arm' else list(range(4))
            loss = grid['arm_score'][indices] if component == 'arm' else grid['grip_mse'][indices] / .2**2
            advantage = (loss[0] - loss) / (1 + loss[0])
            if component == 'arm':
                bad = ((grid['angle_deg'][indices] > np.maximum(grid['angle_deg'][0], 1.) + 1e-6) |
                       (grid['position_m'][indices] > np.maximum(grid['position_m'][0], .01) + 1e-9) |
                       (grid['joint_limit_violations'][indices] > grid['joint_limit_violations'][0]))
            else:
                bad = ((grid['gripper_mae'][indices] > np.maximum(grid['gripper_mae'][0], .02) + 1e-8) |
                       (grid['gripper_outside_margin'][indices] > grid['gripper_outside_margin'][0]))
            gain.append(advantage.transpose(1, 2, 0))
            positive.append(flags[component].transpose(1, 2, 0))
            harm.append(bad.transpose(1, 2, 0))
        output['gain'].append(np.stack(gain, axis=2))
        output['benefit'].append(np.stack(positive, axis=2))
        output['harm'].append(np.stack(harm, axis=2))
    return {k: np.stack(v) for k, v in output.items()}


def row_weights(rows, ids, kind=None):
    w = np.zeros(len(rows), float)
    kinds = [kind] if kind is not None else ['expert', 'anchor', 'recovery']
    present = 0
    for k in kinds:
        ix = [i for i in ids if rows[i]['kind'] == k]
        if not ix:
            continue
        present += 1
        tasks = {rows[i]['task'] for i in ix}
        counts = Counter(rows[i]['group'] for i in ix)
        for i in ix:
            r = rows[i]
            groups = {rows[j]['group'] for j in ix if rows[j]['task'] == r['task']}
            w[i] = 1 / (len(tasks) * len(groups) * counts[r['group']])
    return w / max(1, present)


class GainProbe(nn.Module):
    def __init__(self, dim, architecture):
        super().__init__()
        if architecture == 'linear':
            self.net = nn.Linear(dim, 3)
        elif architecture == 'mlp32':
            self.net = nn.Sequential(nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 3))
        else:
            raise ValueError('Unknown probe architecture')

    def forward(self, x):
        return self.net(x)  # normalized gain, strict-benefit logit, tolerant-harm logit


def select(prediction, gain_threshold, risk_threshold):
    gain, risk = prediction[..., 0], prediction[..., 2]
    acceptable = (gain > gain_threshold) & (risk <= risk_threshold)
    values = np.where(acceptable, gain, -np.inf)
    choices = values.argmax(-1) + 1
    choices[~acceptable.any(-1)] = 0
    return choices.astype(np.int16)


def selected_metrics(panel, choices, weights):
    mass = np.broadcast_to(weights[None, :, None, None], choices.shape)
    if not mass.sum():
        return None
    out = {}
    for key, values in panel.items():
        chosen = np.take_along_axis(values, choices[..., None], axis=-1)[..., 0]
        out[key] = float(np.average(chosen, weights=mass))
    active = choices > 0
    out['coverage'] = float(np.average(active, weights=mass))
    out['benefit_precision'] = None if not mass[active].sum() else out['benefit'] / out['coverage']
    out['harm_given_active'] = None if not mass[active].sum() else out['harm'] / out['coverage']
    out['selected_units'] = int(np.sum(active & (mass > 0)))
    return out


def calibrate(prediction, panel, rows, ids):
    expert = row_weights(rows, ids, 'expert')
    anchor = row_weights(rows, ids, 'anchor')
    off = np.zeros(prediction.shape[:-2], np.int16)
    best = dict(gain_threshold=None, risk_threshold=None, expert=selected_metrics(panel, off, expert),
                anchor=selected_metrics(panel, off, anchor))
    best_key = (0., 0., 0., 0.)
    records = []
    for threshold in (0., .01, .05, .1, .2):
        for risk in (.05, .1, .2):
            choices = select(prediction, threshold, risk)
            e, a = selected_metrics(panel, choices, expert), selected_metrics(panel, choices, anchor)
            # Require zero harm on every calibration anchor, not just a scene average.
            harm = np.take_along_axis(panel['harm'], choices[..., None], -1)[..., 0]
            anchor_bad = bool(harm[:, anchor > 0].any())
            eligible = e['harm'] <= .05 and not anchor_bad
            entry = dict(gain_threshold=threshold, risk_threshold=risk, expert=e, anchor=a, eligible=eligible)
            records.append(entry)
            key = (e['gain'], -e['coverage'], threshold, -risk)
            if eligible and key > best_key:
                best, best_key = entry, key
    return best, records


def chosen_actions(base, proposal, choices, support):
    delta = proposal.astype(float) - base
    if support == 'capped':
        delta = np.clip(delta, -CAPS, CAPS)
    alpha = np.asarray(ALPHAS)[choices]
    out = base.astype(float).copy()
    for side, start in enumerate((0, 7)):
        out[:, :, start:start+6] += delta[:, :, start:start+6] * alpha[:, side, 0, None, None]
        out[:, :, start+6] += delta[:, :, start+6] * alpha[:, side, 1, None]
    return out
