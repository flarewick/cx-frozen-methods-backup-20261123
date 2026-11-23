"""Freeze an independent TRAIN onset pilot; no scene chosen from candidate outcomes."""
import argparse
import os
from pathlib import Path
from .paths import ROOT, new_run, write_json
from .audit import digest
from .native_history_cache import read, require, verify
from .causal_arm_protocol import identifiers
from .onset_train_contract import CONDITIONS,TASKS,SEEDS,STAGE,validate_protocol

PARENT=ROOT/'outputs/reports/diagnostics/onset_attribution_20261007_020824_193778'
PRIOR=ROOT/'outputs/reports/protocols/first_chunk_confirmation_20261006_151638_335660/protocol.json'

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
    parent=read(PARENT/'result.json'); verify(parent['source_sha256'])
    require(parent['status']=='passed' and parent['TRAIN_effect_counts']=={'1':4,'0':4}, 'Audited missing harm support')
    design=read(PARENT/'TRAIN_pilot_design.json');require(design['seeds']==SEEDS and not design['training_started'],'Fixed small pilot design')
    old=read(PRIOR);verify(old['source_sha256']);separated=isolation()
    files=[PARENT/'result.json',PARENT/'TRAIN_pilot_design.json',PARENT/'findings.md',PRIOR]
    files+=list((ROOT/'cx_tta').glob('*.py'))
    files += [ROOT/n for n in ('scripts/collect_onset_training.sh','scripts/screen_onset_training.sh','tests/test_onset_training.py','tests/test_onset_observables.py')]
    sources=dict(old['source_sha256']);sources.update({str(f):digest(f) for f in files})
    out=new_run('onset_TRAIN',group=('reports','protocols'));write_json(out/'scene_isolation.json',separated)
    p=dict(stage=STAGE,audit_version=2,status='frozen_before_official_eligibility_and_candidate_outcomes',
        parent_result=str(PARENT/'result.json'),tasks=list(TASKS),seeds=SEEDS,conditions=list(CONDITIONS),process_blocks=[0,1],
        candidate=old['candidate'],scene_split={str(s):('fit' if i<4 else 'calibration') for i,s in enumerate(SEEDS)},
        confirmation_only=False,training_approved=False,no_new_training_or_router_or_TTA=True,
        compiled=True,internal_noise=True,per_action_observations=True,original_native_budget=True,
        no_RGB_or_video=True,no_scene_replacement=True,maximum_formal_rollouts=64,
        labels='At the exact common onset only: paired complete Z/F outcome; keep both-success and both-failure separate. No dense query labels.',
        role='Independent TRAIN candidate pool with fixed whole-scene fit/calibration; all current S2r/S2v confirmation scenes excluded',
        calibration='Held from gradient fitting; never move calibration scenes to fit to repair label coverage. Existing four S2u TRAIN scenes may enter fit only.',
        support_gate=design['support_gate'],do_not_train_automatically=True,
        no_trigger='Keep all records; no decision label if shared candidate onset not reached',
        partial='Keep failure/censored; never classify incomplete outcomes as failure',
        blocks='Separate processes and private compilation; same sampler and seeds for paired Z/F inside a block; reverse condition order between blocks',
        isolation=str(out/'scene_isolation.json'),isolation_sha256=digest(out/'scene_isolation.json'),source_sha256=sources)
    for k in ('reference','training_run','checkpoint_sha256','caps','robot_joint_limits','budget_config_sha256'):p[k]=old[k]
    validate_protocol(p);write_json(out/'protocol.json',p);print('PROTOCOL',out/'protocol.json',flush=True)


def main():argparse.ArgumentParser(__doc__).parse_args();prepare()
if __name__=='__main__':main()
