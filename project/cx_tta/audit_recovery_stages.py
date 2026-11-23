"""Offline stage diagnostics on the frozen 80-branch recovery pilot (no new labels)."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .audit import digest
from .paths import ROOT, new_run, write_json

COMPLETION = ROOT / 'outputs/reports/audits/recovery_completion_20261001_072637_817377/result.json'


def metrics(states, task):
    objects = states[0]['offline_only']['object_poses']
    answer = {}
    for name in objects:
        if name in ('wall', 'cluttered_obj'):
            continue
        poses = np.array([s['offline_only']['object_poses'][name] for s in states])
        answer[name] = dict(start=poses[0].tolist(), first_chunk=poses[1].tolist(),
                            end=poses[-1].tolist(),
                            max_translation_from_fork=float(np.linalg.norm(poses[:, :3]-poses[0, :3], axis=1).max()),
                            max_height_rise_from_fork=float((poses[:, 2]-poses[0, 2]).max()))
        if task == 'place_shoe' and name == 'shoe':
            # Exact geometric predicates from the frozen task; gripper/contact not inferred.
            q = poses[:, 3:] * np.where(poses[:, 3:4] < 0, -1, 1)
            xy = np.abs(poses[:, :2] - [0, -.08])
            qe = np.abs(q - [.5, .5, -.5, -.5]).max(axis=1)
            pos_ok = (xy < [.05, .02]).all(axis=1)
            answer[name].update(end_xy_abs_error=xy[-1].tolist(),
                end_quaternion_max_abs_error=float(qe[-1]),
                any_sample_xy_pass=bool(pos_ok.any()),
                any_sample_xy_and_orientation_pass=bool((pos_ok & (qe < .07)).any()),
                end_geometry_pass=bool(pos_ok[-1] and qe[-1] < .07))
    for name in states[0]['offline_only']['articulation_qpos']:
        q = np.array([s['offline_only']['articulation_qpos'][name] for s in states])
        answer['joint/' + name] = dict(start=q[0].tolist(), end=q[-1].tolist(),
                                      min=q.min(axis=0).tolist(), max=q.max(axis=0).tolist())
    return answer


def montage(run, rows, out):
    # Preselected seed and branches, including failures; no best-trajectory selection.
    selected = [0, 1, len(rows)//2, len(rows)]
    names = ['fork', 'first chunk', 'middle', 'terminal']
    canvas = Image.new('RGB', (4*320, 3*266), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, (index, label) in enumerate(zip(selected, names)):
        path = run / ('terminal.npz' if index == len(rows) else f'observation_{index:04d}.npz')
        with np.load(path, allow_pickle=False) as data:
            for camera, name in enumerate(('head_rgb', 'right_rgb', 'left_rgb')):
                im = Image.fromarray(data[name])
                canvas.paste(im, (col*320, camera*266+26))
                draw.text((col*320+4, camera*266+4), f'{label}: {name}', fill='black')
    canvas.save(out)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--completion', type=Path, default=COMPLETION)
    args = parser.parse_args()
    completion = json.loads(args.completion.read_text())
    if completion['branches'] != 80:
        raise ValueError('Expected the complete frozen pilot')
    for name, sha in completion['source_sha256'].items():
        if digest(Path(name)) != sha:
            raise ValueError(f'Changed source: {name}')
    out = new_run('recovery_stages', group=('reports', 'audits'))
    branches = []
    source_code = {}
    protocol = json.loads((ROOT/'outputs/reports/protocols/recovery_protocol_20260930_154709_684366/protocol.json').read_text())
    for point in completion['points']:
        pid, task = point['point'], point['task']
        code = Path(protocol['points'][pid]['source'])/'robotwin/envs'/f'{task}.py'
        source_code[str(code)] = digest(code)
        for result in point['rows']:
            branch = Path(point['run'])/f"policy_seed_{result['policy_seed']}"/result['branch']
            rows = [json.loads(line) for line in (branch/'trace.jsonl').read_text().splitlines()]
            actual = json.loads((branch/'result.json').read_text())
            if digest(branch/'trace.jsonl') != actual['trace_sha256']:
                raise ValueError(f'Changed trace {branch}')
            with np.load(branch/'terminal.npz', allow_pickle=False) as obs:
                command = obs['command_state_raw'].tolist()
            record = dict(point=pid, task=task, scene_seed=point['scene_seed'], **result,
                          run=str(branch), trace_sha256=actual['trace_sha256'],
                          terminal_gripper_commands=[command[6], command[13]],
                          sampled_geometry=metrics([rows[0]['before']]+[r['after'] for r in rows], task))
            branches.append(record)
            if pid in (4, 5, 6, 7) and result['policy_seed'] == 910000 and result['branch'] == 'base':
                montage(branch, rows, out/f'point_{pid}_base_910000.png')
    report = dict(status='complete', branches=branches,
        source_completion=str(args.completion), source_completion_sha256=digest(args.completion),
        task_source_sha256=source_code, code_sha256=digest(Path(__file__)),
        limitations=['Geometry sampled at chunk boundaries, not all physics steps.',
            'Contact, grasp ownership and functional point poses absent from old snapshots.',
            'Shoe geometry predicates exclude gripper predicate; do not replace official outcome.',
            'Switch joint limits absent from old snapshots; no inferred success threshold.',
            'Developer checkpoints; no held-out evaluation or recovery training labels.'])
    write_json(out/'result.json', report)
    print(out)
    for pid in (4, 5, 6, 7):
        records = [r for r in branches if r['point'] == pid and r['branch'] == 'base']
        print(json.dumps(records, ensure_ascii=False))


if __name__ == '__main__':
    main()
