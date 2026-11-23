"""Fail closed on candidate changes, replacement scenes, or confirmation registries."""
from pathlib import Path
import numpy as np
from .audit import digest
from .native_history_cache import read, require, verify

CONDITIONS = ('Z_zero', 'F_first_full')
TASKS = ('place_empty_cup', 'beat_block_hammer')
SEEDS = list(range(501300, 501308))
STAGE = 'onset_paired_TRAIN'
CHECKPOINT = '92b1a2e487ebb5f9ecad6bc519577962b69971a4bf8717ed14b7d0c735ddbef2'


def validate_protocol(p):
    require(p['stage'] == STAGE and p['audit_version'] == 2, 'Frozen independent TRAIN collection stage')
    require(p['conditions'] == list(CONDITIONS) and p['tasks'] == list(TASKS)
            and p['seeds'] == SEEDS and p['process_blocks'] == [0, 1], 'Fixed panel')
    require(p['candidate'] == dict(ARM_duration_queries=1, ARM_strength=1,
            GRIP_strength=0, default_exit=True, threshold=.1, delay_queries=1,
            one_shot=True), 'Frozen first-full candidate')
    require(p['checkpoint_sha256'] == CHECKPOINT, 'Original frozen A checkpoint')
    require(p['caps'] == np.asarray([.1]*6+[.2]+[.1]*6+[.2], np.float32).tolist(), 'Original native residual bounds')
    require(all(p[k] is True for k in ('compiled', 'internal_noise',
            'per_action_observations', 'original_native_budget',
            'no_new_training_or_router_or_TTA')), 'Official runtime and held-out role')
    require(p['training_approved'] is False and p['confirmation_only'] is False, 'Collection is not automatic training approval')
    require(p['scene_split'] == {str(s): ('fit' if i<4 else 'calibration') for i,s in enumerate(SEEDS)}, 'Pre-outcome whole-scene fit/calibration assignment')


def registry_records(protocol, registry):
    protocol, registry = Path(protocol), Path(registry)
    p = read(protocol); validate_protocol(p)
    m, result = read(registry/'manifest.json'), read(registry/'result.json')
    require(m['stage'] == 'onset_TRAIN_eligibility_only'
            and m['protocol_sha256'] == digest(protocol)
            and m['tasks'] == p['tasks'] and m['seeds'] == p['seeds'], 'Same frozen eligibility protocol')
    require(result['status'] == 'complete' and result['all_fixed_seeds_attempted']
            and result['no_replacement'] and result['policy_model_loaded'] is False
            and result['policy_sampler_calls'] == result['policy_success_outcomes_collected'] == 0,
            'Pre-outcome official eligibility only')
    verify(result['source_sha256'])
    rows = read(registry/'records.json')
    expected = [(t, s, i) for t in p['tasks'] for i, s in enumerate(p['seeds'])]
    require([(r['task'], r['seed'], r['scene_index']) for r in rows] == expected
            and result['attempted'] == len(expected), 'Every fixed scene, original indices, no replacement')
    for t in p['tasks']:
        subset = [r for r in rows if r['task'] == t]
        require(result['tasks'][t] == dict(attempted=len(subset),
                eligible=sum(r['eligible'] for r in subset),
                invalid=sum(not r['eligible'] for r in subset)), 'Actual eligibility counts')
    for row in rows:
        require(type(row['eligible']) is bool, 'Boolean eligibility')
        if row['eligible']:
            require(row['expert_plan_success'] and row['expert_success']
                    and bool(row['instruction']) and bool(row['initial_physical_sha256'])
                    and bool(row['initial_observation_signature']), 'Official valid scene and reset identity')
    return rows
