"""Compact frozen-component measurements, using only saved numeric diagnostics.

These are development ablations, not independent generalization estimates. No
physics measurements or outcome labels are fed into a deployed model here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audit import digest
from .structured_attribution_analysis import _comparison, _load_trace, _trace_errors

ARMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPERS = (6, 13)


def channel_metrics(values):
    """Every executed step has equal mass, without padding or terminal tails."""
    values = np.asarray(values, np.float32)
    if values.ndim != 2 or values.shape[1] != 14 or not np.isfinite(values).all():
        raise ValueError('Expected finite executed [N,14] measurements')
    result = dict(steps=len(values))
    for name, columns in (('arms', ARMS), ('grippers', GRIPPERS)):
        selected = values[:, columns].astype(np.float64)
        result[name] = dict(rms=float(np.sqrt(np.mean(selected ** 2))) if len(values) else None,
            mean_abs=float(np.mean(np.abs(selected))) if len(values) else None,
            max_abs=float(np.max(np.abs(selected))) if len(values) else None,
            positive_fraction=float(np.mean(selected > 0)) if len(values) else None,
            negative_fraction=float(np.mean(selected < 0)) if len(values) else None)
    return result


def trajectory_measurements(trace):
    """Summarize raw/applied size and observed correction block discontinuities."""
    rows, arrays = trace['rows'], trace['arrays']
    errors = _trace_errors(trace, sham=trace['episode']['condition'] == 'L0')
    if errors:
        raise ValueError('Invalid component numeric trace: ' + '; '.join(errors))
    result = dict(queries=len(rows), executed_steps=sum(r['executed'] for r in rows),
                  queries_with_model_history=sum(r['has_history'] for r in rows))
    for key in ('raw_delta', 'applied_delta'):
        values = [arrays[key][i, :row['executed']] for i, row in enumerate(rows)]
        all_values = np.concatenate(values)
        first = values[0]
        later = np.concatenate(values[1:]) if len(values) > 1 else np.empty((0, 14), np.float32)
        jumps = [arrays[key][i + 1, 0] - arrays[key][i, 31]
                 for i in range(len(rows) - 1) if rows[i]['executed'] == 32]
        boundary = np.stack(jumps) if jumps else np.empty((0, 14), np.float32)
        result[key] = dict(all_executed=channel_metrics(all_values),
            query0=channel_metrics(first), later_queries=channel_metrics(later),
            block_boundary_change=channel_metrics(boundary))
    result['history_contract'] = dict(
        observed_flags=[row['has_history'] for row in rows],
        history_policy=rows[0].get('history_policy', 'real'),
        real_history_available=[row.get('real_history_available') for row in rows])
    return result


def summarize_components(run):
    """Report all conditions; never choose a winning task-specific strength."""
    run = Path(run).resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    episodes = json.loads((run / 'episodes.json').read_text())
    report = dict(task=manifest['task'], stage='components', checkpoint_sha256=manifest['checkpoint_sha256'],
        scope='Fixed selected development scenes, one repeat per component; diagnostic only',
        errors=[], observed_outcomes={}, pairs=[], trajectories=[], source_sha256={})
    traces = {}
    for episode in episodes:
        key = (episode['seed'], episode['repeat'], episode['condition'])
        try:
            trace = _load_trace(run, episode)
            measurements = trajectory_measurements(trace)
            if key in traces:
                raise ValueError('Duplicate component record')
            traces[key] = trace
            report['trajectories'].append(dict(seed=key[0], repeat=key[1], condition=key[2], **measurements))
            for suffix in ('.npz', '.jsonl.gz'):
                path = run / 'traces' / (episode['trace_stem'] + suffix)
                report['source_sha256'][str(path)] = digest(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            report['errors'].append(dict(seed=key[0], repeat=key[1], condition=key[2], error=str(error)))
    for condition in manifest['conditions']:
        rows = [r for r in episodes if r['condition'] == condition]
        completed = [r for r in rows if r.get('complete') is True and type(r.get('success')) is bool]
        report['observed_outcomes'][condition] = dict(successes=sum(r['success'] for r in completed),
            completed=len(completed), attempted=len(rows),
            censored=len(rows) - len(completed),
            ratio=sum(r['success'] for r in completed) / len(completed) if completed else None)
    for seed in manifest['seeds']:
        for repeat in range(manifest['repeats']):
            baseline = traces.get((seed, repeat, 'L0'))
            for condition in manifest['conditions']:
                if condition == 'L0':
                    continue
                candidate = traces.get((seed, repeat, condition))
                if baseline is None or candidate is None:
                    continue
                pair = _comparison(baseline, candidate, 'component_vs_same_runtime_L0')
                first = pair['first_divergence_query']
                initial_paired = (pair['initial_observation_equal'] and pair['initial_native_equal']
                    and pair['initial_rng_equal'] and first['rng_after'] != 0
                    and first['base_numbers'] != 0 and first['structured_observables'] != 0
                    and first['command_state'] != 0)
                complete = all(t['episode'].get('complete') is True and type(t['episode'].get('success')) is bool
                               for t in (baseline, candidate))
                if not initial_paired:
                    label = 'invalid_initial_pairing'
                elif not complete:
                    label = 'censored'
                elif baseline['episode']['success'] == candidate['episode']['success']:
                    label = 'tie'
                else:
                    label = 'rescued' if candidate['episode']['success'] else 'harmed'
                report['pairs'].append(dict(seed=seed, repeat=repeat, condition=condition,
                    initial_pairing=bool(initial_paired), outcome_change=label,
                    base_success=baseline['episode'].get('success'), candidate_success=candidate['episode'].get('success'),
                    numerical_difference_same_observation_rng=pair['base_difference_same_observation_rng'],
                    first_divergence_query=first))
    report['changes'] = {condition: {
        name: sum(p['condition'] == condition and p['outcome_change'] == name for p in report['pairs'])
        for name in ('rescued', 'harmed', 'tie', 'censored', 'invalid_initial_pairing')}
        for condition in manifest['conditions'] if condition != 'L0'}
    for name in ('manifest.json', 'episodes.json'):
        report['source_sha256'][str(run / name)] = digest(run / name)
    report['limitations'] = [
        'Two reused selected scenes per task; no representative success rate or independent confirmation.',
        'One repeat per component; repeated component efficacy is not yet measured.',
        'All variants are retained. No per-task coefficient selection or new learned router.',
        'Later trajectories intentionally differ; base differences use only equal observation/RNG queries.',
        'Raw/applied magnitude and block changes diagnose execution; they do not prove teacher-direction correctness.',
    ]
    return report
