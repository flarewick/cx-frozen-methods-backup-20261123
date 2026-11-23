"""Freeze the selected TRAIN prescription before fresh whole-scene outcomes."""
import argparse
import os
from pathlib import Path
from .paths import ROOT, new_run, write_json
from .audit import digest
from .native_history_cache import read, require, verify
from .causal_arm_protocol import identifiers
from .first_chunk_contract import CONDITIONS, TASKS, SEEDS, STAGE, validate_protocol

PARENT = ROOT/'outputs/reports/attribution/arm_prescription_accepted_20261006_145548_103774'
PRIOR = ROOT/'outputs/reports/protocols/arm_prescription_v2_20261006_134344_607783/protocol.json'


def isolation():
    # Include prior eligibility and phase-source metadata omitted by older scans.
    names = {'manifest.json', 'protocol.json', 'training_protocol.json', 'records.json',
             'episodes.json', 'prepared.json', 'preparation.json'}
    skip = {'robotwin', 'runtime', 'cache', 'source_snapshot', 'evaluation_source_snapshot'}
    files = []
    for folder, dirs, entries in os.walk(ROOT/'outputs', followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip and not Path(folder, d).is_symlink()]
        files.extend(Path(folder, n) for n in entries if n in names)
    known = set(); sources = {}
    for file in sorted(files):
        value = read(file)
        known.update(identifiers(value)); sources[str(file)] = digest(file)
        # Conservatively reject interval-based evaluation allocations, too.
        def ranges(x):
            if isinstance(x, dict):
                start, count = x.get('seed_start'), x.get('episodes')
                if type(start) is int and type(count) is int:
                    known.update(('*', n) for n in range(start, start+count))
                for v in x.values(): ranges(v)
            elif isinstance(x, list):
                for v in x: ranges(v)
        ranges(value)
    collisions = [(t,s) for t in TASKS for s in SEEDS if (t,s) in known or ('*',s) in known]
    require(not collisions, 'New scene identifiers already used: '+str(collisions))
    return dict(passed=True, new_scenes={t:SEEDS for t in TASKS}, sources_checked=len(files),
        observed_existing_scene_identifiers=len(known), collisions=collisions, source_sha256=sources,
        scope='All surviving output manifests/protocols/records/episodes, including phase and eligibility registries. '
              'Inherited deleted raw-source gaps remain; this is not proof about unavailable history.')


def prepare():
    parent, design, old = read(PARENT/'result.json'), read(PARENT/'next_experiments.json'), read(PRIOR)
    require(digest(PARENT/'result.json') == design['parent_sha256'] and parent['status']=='passed', 'Accepted selection evidence')
    verify(parent['source_sha256']); verify(old['source_sha256'])
    require(design['conditions']==list(CONDITIONS) and design['proposed_fixed_attempts_per_task']==8, 'Accepted prospective design')
    separated = isolation()
    files = [PARENT/'result.json', PARENT/'next_experiments.json', PARENT/'independent_verification.json', PRIOR]
    files += list((ROOT/'cx_tta').glob('*.py'))
    files += [ROOT/name for name in ('scripts/eval_first_chunk_confirmation.sh',
        'scripts/screen_first_chunk_scenes.sh', 'tests/test_first_chunk_confirmation.py', 'tests/test_arm_prescription.py')]
    sources = dict(old['source_sha256']); sources.update({str(f):digest(f) for f in files})
    out = new_run('first_chunk_confirmation', group=('reports', 'protocols'))
    write_json(out/'scene_isolation.json', separated)
    p = dict(stage=STAGE, status='frozen_before_new_scene_eligibility_and_policy_outcomes', audit_version=2,
        selection_result=str(PARENT/'result.json'), selection_sha256=digest(PARENT/'result.json'),
        selection_role='Four prior TRAIN scenes used to select F; excluded from confirmation',
        tasks=list(TASKS), seeds=SEEDS, process_blocks=[0,1], conditions=list(CONDITIONS),
        candidate=dict(ARM_duration_queries=1, ARM_strength=1, GRIP_strength=0,
            default_exit=True, threshold=.1, delay_queries=1, one_shot=True),
        causal_onset='First current commanded gripper minus min clipped current base forecast >=0.1; next query',
        role='New whole-scene confirmation, never fit or tune thresholds on these scenes',
        registry_binding='Pass completed registry explicitly; it must hash this unchanged protocol',
        maximum_formal_rollouts=64, maximum_distinct_scenes=16,
        compiled=True, internal_noise=True, per_action_observations=True, original_native_budget=True,
        confirmation_only=True, training_approved=False, no_new_training_or_router_or_TTA=True,
        no_RGB_or_video=True, no_GRIP_intervention=True, no_scene_replacement=True,
        ordering='Alternate Z/F by eligible scene index and reverse between blocks; independent compiled processes',
        screening='Official plan_success and check_success before any candidate outcomes; all 16 fixed attempts kept',
        smoke='First eligible scene, at most five queries per condition, always censored; no tuning after inspection',
        primary_report=['Per task and block rescue/harm/net against same-process Z',
            'Distinct scenes, cross-block replication, base-success preservation denominators',
            'All invalid, untriggered and censored records; original-budget native outcomes'],
        next_decision='Do not expand shoe/switch or fit router yet. Review net effect and harms in both blocks/tasks; '
            'report scene-level uncertainty and preserve small-sample limits. Any future fitted gate needs independent TRAIN '
            'support and another untouched confirmation; never turn these confirmation outcomes into training labels.',
        isolation=str(out/'scene_isolation.json'), isolation_sha256=digest(out/'scene_isolation.json'),
        source_sha256=sources)
    for key in ('reference','training_run','checkpoint_sha256','caps','robot_joint_limits','budget_config_sha256'):
        p[key] = old[key]
    validate_protocol(p)
    write_json(out/'protocol.json', p)
    print('PROTOCOL', out/'protocol.json', flush=True)
    return out/'protocol.json'


def main():
    argparse.ArgumentParser(__doc__).parse_args(); prepare()


if __name__ == '__main__': main()
