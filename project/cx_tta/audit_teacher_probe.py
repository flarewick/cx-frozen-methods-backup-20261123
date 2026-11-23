"""Independently check a native teacher probe's saved command/budget/observation chain."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .eval_diagnostics import array_hash
from .paths import write_json


def audit(run):
    m = json.loads((run/'manifest.json').read_text())
    r = json.loads((run/'result.json').read_text())
    fork = json.loads((run/'fork.json').read_text())
    traces = [json.loads(line) for line in (run/'trace.jsonl').read_text().splitlines()]
    with np.load(run/'native_actions.npz', allow_pickle=False) as saved:
        actions = saved['actions']
    checks = dict(code_unchanged=all(digest(Path(p)) == h for p,h in m['code_sha256'].items()),
        trace_unchanged=digest(run/'trace.jsonl') == r['trace_sha256'],
        fork_matched=fork['physical_match']['matched'] and r['fork_verified'],
        budget=r['action_count'] == m['prefix_actions']+len(traces) <= r['native_limit'] == fork['native_limit'],
        counts=len(traces) == r['teacher_actions'] == r['observation_updates'] == len(actions),
        exported_actions=actions.shape == (len(traces),14) and np.isfinite(actions).all(),
        commands=True, chain=True, saved_observations=True,
        physics=sum(x['physics_steps'] for x in traces) == r['physics_steps'],
        terminal_success=(r['terminal_success'] == r['teacher_success'] if r['native_terminal'] else r['terminal_success'] is None))
    expected = fork['observation']
    for i,row in enumerate(traces):
        with np.load(run/f"plan_{row['plan']:03d}.npz", allow_pickle=False) as p:
            checks['commands'] &= bool(row['dense_index'] in p['indices'] and
                np.array_equal(actions[i], p['dense_commands'][row['dense_index']]) and
                np.array_equal(actions[i], np.asarray(row['action'], np.float32)))
        checks['chain'] &= bool(row['before_observation'] == expected and
            row['action_count'] == m['prefix_actions']+i+1 and row['physics_steps'] > 0 and
            (not row['success'] or i == len(traces)-1))
        expected = row['after_observation']
    if traces:
        checks['chain'] &= bool(traces[-1]['success'] == r['teacher_success'])
    count = 0
    for path in [run/'fork.npz', *sorted(run.glob('observation_*.npz')), run/'terminal.npz']:
        with np.load(path, allow_pickle=False) as obs:
            observed = dict(images={key:array_hash(obs[name]) for key,name in
                (('head_camera','head_rgb'), ('right_camera','right_rgb'), ('left_camera','left_rgb'))},
                command_state=array_hash(obs['command_state_raw']))
        if path.stem == 'fork': expected = fork['observation']
        elif path.stem == 'terminal': expected = traces[-1]['after_observation'] if traces else fork['observation']
        else: expected = traces[int(path.stem.split('_')[1])-1]['after_observation']
        checks['saved_observations'] &= observed == expected
        count += 1
    checks = {k:bool(v) for k,v in checks.items()}
    report = dict(passed=all(checks.values()), checks=checks, actions=len(traces),
                  saved_observations=count, result_sha256=digest(run/'result.json'),
                  auditor_sha256=digest(Path(__file__)),
                  scope='Saved interface accounting only, not teacher generality or residual reachability')
    write_json(run/'audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    report = audit(args.run.resolve())
    print(json.dumps(report))
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
