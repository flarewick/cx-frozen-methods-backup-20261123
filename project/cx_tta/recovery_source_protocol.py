"""Preregister independent recovery scenes before obtaining their outcomes."""
import argparse
import json
from pathlib import Path

from .audit import digest
from .paths import ROOT, new_run, write_json

TASKS = ('beat_block_hammer', 'place_shoe', 'turn_switch', 'place_empty_cup')
SOURCES = (
    'cx_tta/recovery_source_protocol.py', 'cx_tta/collect_recovery_sources.py',
    'cx_tta/audit_recovery_sources.py', 'cx_tta/verify_recovery_source_replay.py',
    'scripts/collect_recovery_sources.sh', 'cx_tta/evaluate_b2.py',
    'cx_tta/eval_diagnostics.py', 'cx_tta/deploy.py', 'cx_tta/cache_expert.py',
    'cx_tta/collect_smoke.py', 'cx_tta/recovery_contract.py', 'cx_tta/collect_recovery.py',
    'cx_tta/probe_action_replay.py', 'cx_tta/capture.py', 'cx_tta/recording.py',
    'cx_tta/audit_recovery_cache.py',
)


def allocation():
    return {task: dict(gate=[420000], train=list(range(421000, 421008)),
                      train_reserve=list(range(421008, 421012)),
                      calibration=list(range(422000, 422006)),
                      confirmation=list(range(423000, 423006))) for task in TASKS}


def validate(protocol):
    if protocol['version'] != 'S2b-source-v1' or protocol['scenes'] != allocation():
        raise ValueError('Unexpected scene allocation')
    for file, sha in protocol['source_sha256'].items():
        if digest(Path(file)) != sha:
            raise ValueError('Frozen source changed: ' + file)
    return protocol


def main():
    argparse.ArgumentParser(__doc__).parse_args()
    run = new_run('multiscene_recovery', group=('reports', 'protocols'))
    protocol = dict(
        version='S2b-source-v1', status='frozen_before_new_scene_outcomes', scenes=allocation(),
        open_roles=['gate', 'train'], policy_init_seed=0, instruction_type='unseen',
        rollout='Official compiled Pi0.5, internal noise, get_obs after each of 32 callbacks; '
                'native task budget. Expert-valid scene screening and seed reset as evaluate_b2.run_episode.',
        randomness='Seed Python/NumPy/Torch with scene seed before screening and reset; no extra policy '
                   'warmup or query. Save before/after RNG for every actual policy query.',
        scene_unit='Task plus scene seed. All frames, branches, policy repeats and recovery targets '
                   'of that scene remain in one role; gate scene is development only.',
        source_selection='Attempt all eight train seeds in order, once. Do not replace successes, '
                         'expert-invalid scenes, teacher failures or runtime failures with easier seeds. '
                         'No train_reserve/calibration/confirmation collection in this version.',
        fork_candidates=dict(native_counts=[32, 64, 96, 128],
            rule='All existing nonterminal checkpoints at these counts in failed L0 scenes are candidates; '
                 'max four per scene. A later frozen teacher-entry protocol must qualify every attempted '
                 'branch. Multiple checkpoints never count as independent scenes.'),
        input_schema='Lossless uint8 3RGB + raw and float32 command_state + grounded instruction; '
                     'full same-state base 32x14 and internal sampler RNG. Strictly previous query '
                     'observation within the same scene only. Privileged physical trace stored separately.',
        label_policy='L0 success may supply success-preservation targets. L0 failures are source states, '
                     'NOT recovery action targets. Teacher recovery must save actual native commands, '
                     'causal observations every 32 teacher actions, valid tail and full outcome. '
                     'Require exact fork replay, native budget, semantic entry and completed success audit '
                     'before admitting recovery targets. Never label failed base commands as expert targets.',
        qualification='First establish at least four clean recovery scenes/task in train and later two each '
                      'in calibration/confirmation for shoe and switch. Report all denominators. '
                      'These are feasibility minima, not powered success-rate evidence. Freeze a new '
                      'protocol before activating reserves or held-out labels. Hammer/cup teacher entry '
                      'is unvalidated and must be qualified separately.',
        training_next='Fixed expert/PCA/preservation/init/budget matched comparisons with same-task '
                      'recovery scene exclusion; task-held-out separate. No router/TTA training from sources.',
        storage=dict(video=False, lossless_chunk_observations=True, duplicate_expert_dataset=False,
                     maximum_run_gib=1.5, minimum_reported_free_gib=1,
                     note='Run plus private compiler scratch checked between chunks/scenes; shared df '
                          'does not measure user quota. OS write failures remain explicit incomplete runs.'),
        source_sha256={str(ROOT / rel): digest(ROOT / rel) for rel in SOURCES},
    )
    validate(protocol)
    write_json(run / 'protocol.json', protocol)
    print(run / 'protocol.json')


if __name__ == '__main__':
    main()
