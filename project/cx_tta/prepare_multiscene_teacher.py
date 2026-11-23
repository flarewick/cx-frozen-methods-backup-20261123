"""Audit new sources and freeze all developer forks before teacher outcomes."""
import argparse
import json
from pathlib import Path
from .audit import digest
from .audit_recovery_sources import audit
from .paths import ROOT, new_run, write_json

SOURCE_PROTOCOL = ROOT/'outputs/reports/protocols/multiscene_recovery_20261002_145008_830306/protocol.json'
SOURCE_RUNS = (
    ROOT/'outputs/diagnostics/recovery_sources/place_shoe/source_20261002_153938_124375',
    ROOT/'outputs/diagnostics/recovery_sources/turn_switch/source_20261002_153403_771011',
)


def main():
    argparse.ArgumentParser(__doc__).parse_args()
    out = new_run('multiscene_teacher', group=('reports','protocols'))
    print(out, flush=True)
    points=[]; inventories=[]; hashes={str(SOURCE_PROTOCOL):digest(SOURCE_PROTOCOL)}
    for source in SOURCE_RUNS:
        checked=audit(source)
        m=json.loads((source/'manifest.json').read_text())
        if m['role']!='train' or m['protocol_sha256']!=digest(SOURCE_PROTOCOL):
            raise ValueError('Wrong source protocol/role')
        write_json(out/f"{m['task']}_source_audit.json",checked)
        inventories.append(dict(task=m['task'],source=str(source),scenes=checked['scenes'],
                                result=json.loads((source/'result.json').read_text())))
        hashes.update(checked['evidence_sha256'])
        episodes=json.loads((source/'episodes.json').read_text())
        for e in episodes:
            if not e['eligible'] or e['success']:continue
            rows=[json.loads(x) for x in (source/'traces'/f"r0_seed{e['seed']}.jsonl").read_text().splitlines()]
            entries=json.loads((source/'observations'/f"seed_{e['seed']}"/'index.json').read_text())
            count=0
            for i,row in enumerate(rows):
                if count in (32,64,96,128):
                    prefix=source/'observations'/f"seed_{e['seed']}"/f'chunk_{i:03d}'
                    past=prefix.with_name(f'chunk_{i-1:03d}.npz') if i else None
                    points.append(dict(index=len(points),task=m['task'],seed=e['seed'],source=str(source),
                        chunk=i,prefix_actions=count,instruction=e['instruction'],
                        observation_signature=row['observation_components'],
                        observation=str(prefix.with_suffix('.npz')),
                        observation_sha256=digest(prefix.with_suffix('.npz')),
                        past_observation=str(past) if past else None,
                        past_observation_sha256=digest(past) if past else None,
                        entry='native_switch' if m['task']=='turn_switch' else 'pending_shoe_support',
                        runnable=m['task']=='turn_switch'))
                count+=row['executed']
    # Freeze code before any teacher query. Old sources and model files stay unchanged.
    files=[Path(__file__),ROOT/'cx_tta/probe_multiscene_teacher.py',ROOT/'scripts/probe_multiscene_teacher.sh',
           ROOT/'cx_tta/native_teacher.py',ROOT/'cx_tta/audit_teacher_probe.py',
           ROOT/'cx_tta/collect_recovery.py',ROOT/'cx_tta/probe_action_replay.py',
           ROOT/'cx_tta/probe_recovery_teacher.py',ROOT/'cx_tta/evaluate_b2.py',ROOT/'cx_tta/recovery_contract.py']
    for source in SOURCE_RUNS:
        m=json.loads((source/'manifest.json').read_text())
        files.extend([source/'robotwin/envs/_base_task.py',source/'robotwin/envs'/f"{m['task']}.py",
                      source/'robotwin/envs/robot/robot.py',source/'robotwin/envs/robot/planner.py'])
    from .cache_expert import RT
    files.append(RT/'RoboTwin/envs/curobo/src/curobo/wrap/reacher/motion_gen.py')
    gate=next(p['index'] for p in points if p['runnable'])
    protocol=dict(version='S2c-teacher-v1',status='frozen_before_teacher_outcomes',
        source_protocol=str(SOURCE_PROTOCOL),inventories=inventories,points=points,
        gate_point=gate,gate_repetitions=[0,1],teacher_seed=930000,
        teacher='Official task play_once through unchanged NativeTeacher, source save_freq stride, '
                'actual native actions and original remaining budget; no VLA or adapter query.',
        entry='Only turn_switch enabled: non-successful fork, native task-semantic switch press. '
              'place_shoe requires measured support/owner qualification first; no unconditional regrasp.',
        qualification='Exact source RGB/command and physical prefix, fixed native limit, actual teacher '
                      'command accounting, original success predicate false at fork and true at terminal, '
                      'switch qpos threshold audited. Unfinished/failed planning is censored, never positive.',
        selection='Every failed source scene retained; four fixed forks each. First runnable point and '
                  'two fresh-process repeats form local interface/planner check. No best-of-N. '
                  'No calibration/confirmation or reserves opened.',
        history='Fork may reference previous L0 observation; subsequent teacher queries reference '
                'only previous teacher observation at 32 actual teacher actions. Do not reuse future L0 actions.',
        learning='Teacher outputs are privileged, unbounded action-center candidates, not deployable '
                 'router inputs, residual-cap feasibility, or adapter success improvement. Frozen '
                 'features and same-state base queries on teacher observations are still required.',
        no_video=True,source_sha256=hashes,code_sha256={str(f):digest(f) for f in files})
    write_json(out/'protocol.json',protocol)
    print(json.dumps(dict(points=len(points),independent_failed_scenes=len({(x['task'],x['seed']) for x in points}),
                          runnable_points=sum(x['runnable'] for x in points),gate_point=gate)),flush=True)


if __name__=='__main__':main()
