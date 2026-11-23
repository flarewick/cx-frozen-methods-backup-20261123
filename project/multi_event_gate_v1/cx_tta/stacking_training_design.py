"""Preregister the stacking data split before any new training or selection."""
import argparse
import json
from pathlib import Path

from .audit import digest
from .paths import ROOT, WORKSPACE_ROOT, new_run, write_json

TASKS = ('stack_bowls_two', 'stack_blocks_two')
SPLITS = {
    'smoke': dict(start=7099000, stop=7099020, episodes=1, role='interface_only'),
    'adapter_fit': dict(start=7100000, stop=7101000, episodes=40, role='independent_train'),
    'adapter_validation': dict(start=7101000, stop=7102000, episodes=10, role='development_holdout'),
    'gate_fit': dict(start=7102000, stop=7103000, role='independent_train'),
    'gate_validation': dict(start=7103000, stop=7104000, role='development_holdout'),
    'gate_calibration': dict(start=7104000, stop=7105000, role='independent_calibration'),
    'final_eval': dict(start=7105000, stop=7106000, episodes=100, role='final_eval'),
    'confirmation_eval': dict(start=7106000, stop=7107000, episodes=100, role='final_eval'),
}
SCHEMA = 'stacking_two_task_training_v1'


def validate_design(p):
    if p.get('schema') != SCHEMA or p.get('tasks') != list(TASKS):
        raise ValueError('Wrong shared two-task protocol')
    if p.get('splits') != SPLITS:
        raise ValueError('Registered seed windows/counts changed; create a new version')
    for name, s in p['splits'].items():
        if not (0 <= s['start'] < s['stop']):
            raise ValueError('Invalid seed window')
        for other, t in p['splits'].items():
            if other != name and max(s['start'], t['start']) < min(s['stop'], t['stop']):
                raise ValueError('Scene split overlap')
    for name, sha in p['source_sha256'].items():
        if digest(Path(name)) != sha:
            raise ValueError('Frozen dependency changed: ' + name)
    return p


def load_design(path):
    return validate_design(json.loads(Path(path).read_text()))


def prepare():
    reference = ROOT/'outputs/reports/protocols/shared_onset_expansion_20261007_145526_085093/protocol.json'
    ref = json.loads(reference.read_text())
    old = Path(ref['training_run'])
    train = json.loads((old/'training_protocol.json').read_text())
    parent = Path(train['parent_run'])
    rows = json.loads((parent/'protocol.json').read_text())['rows']
    from collections import Counter
    sources = [reference, old/'training_protocol.json', old/'A_bounded_clipped_seed0.pt',
               parent/'protocol.json', Path(train['projection']), Path(train['robot_config']),
               WORKSPACE_ROOT/'configs/events.json']
    sources += [WORKSPACE_ROOT/p for p in (
        'cx_tta/stacking_training_design.py', 'cx_tta/collect_stacking_experts.py',
        'cx_tta/recording.py', 'cx_tta/capture.py', 'cx_tta/effect_contract.py',
        'cx_tta/collect_smoke.py', 'cx_tta/expert_screening.py', 'cx_tta/stacking_training_data.py',
        'cx_tta/stacking_support.py', 'cx_tta/native_entry_storage.py',
        'cx_tta/evaluate_structured_low_storage.py', 'scripts/collect_stacking_experts.sh')]
    p = dict(schema=SCHEMA, tasks=list(TASKS), splits=SPLITS, task_config='demo_randomized',
        instruction_split='seen', evaluation_instruction_split='unseen',
        old_training_run=str(old), old_gate_run=ref['gate_run'],
        old_adapter_training_rows_by_task=dict(Counter(r['task'] for r in rows)),
        adapter=dict(shared=True, initialize_from=str(old/'A_bounded_clipped_seed0.pt'),
            projection=train['projection'], projection_fit='frozen old projection; no refit on holdouts',
            arm_cap_rad=.1, gripper_residual=0., horizon=32, auxiliary_geometry_weight=0.,
            source_tasks=list(TASKS), weighting='equal task, then equal scene, then equal row',
            proposed_updates=1400, proposed_lr=.0003, seed=0,
            qualification='Hyperparameters frozen at training-cache audit, before GPU training; no claim of trained model yet',
            teacher='expert initialization first; native recovery targets added only after qualified replay',
            target_alignment='Saved expert future commands are demonstration targets, NOT certified32 native executor actions',
            history='No invented32-action history from expert frame indices; separately qualify native history',
            success_preservation='Collect independent successful base TRAIN prefixes; zero ARM target only on qualified preservation rows'),
        gate=dict(shared=True, train_after='new adapter frozen and checkpoint hashed',
            input='legal current/history observation, command state/FK, base, specific candidate, event history',
            label='same-prefix Z/C full terminal outcomes; one candidate ARM chunk then both return to base',
            labels_depend_on_candidate_checkpoint=True, reuse_old_candidate_labels=False,
            split_by='whole scene including all branches, instructions and events',
            histories='base-only and previous accept/reject histories; first and later open/close events',
            calibration='separate gate_calibration scenes; thresholds never tuned on EVAL'),
        evaluation=dict(episodes_per_task_condition=100,
            stage1='Z / old candidate / new candidate, same first-event detector and max1; candidate attribution',
            stage2='same new adapter: learned gate max1 versus same learned gate max3, plus Z and ungated controls',
            claims='multitask held-out-scene efficacy, not zero-shot task generalization; no TTA yet'),
        collection=dict(schema=3, video=False, images='three JPEG streams in HDF5 required for VLA encoding',
            retained='commands, actual arm state, physics ticks/dt/dense command trace, grounded language, offline geometry, source hashes',
            raw_cache='delete only this attempt temporary frame pickles after validated export; keep trajectory and HDF5'),
        current_status='ready_for_external_collection_interface_check_not_training',
        source_sha256={str(f): digest(f) for f in sources})
    validate_design(p)
    run = new_run('stacking_training', group=('reports', 'protocols'))
    write_json(run/'protocol.json', p)
    print(run/'protocol.json', flush=True)
    return run/'protocol.json'


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--check', type=Path)
    args = ap.parse_args()
    if args.check:
        p = load_design(args.check)
        print(json.dumps(dict(valid=True, tasks=p['tasks'], splits=p['splits'])))
    else:
        prepare()


if __name__ == '__main__':
    main()
