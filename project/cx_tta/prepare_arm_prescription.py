"""Freeze finite ARM contrasts on existing TRAIN scenes; no new scene screening."""
import argparse
from pathlib import Path
import numpy as np
from .paths import ROOT, new_run, write_json
from .audit import digest
from .native_history_cache import read, verify, require
from .arm_prescription_policy import compose, CONDITIONS, PRESCRIPTIONS
from .causal_arm_gate import OneShotClosingGate
from .structured_attribution_analysis import _load_trace

PARENT=ROOT/'outputs/reports/attribution/phase_targets_accepted_20261006_124642_562354'
PHASE=ROOT/'outputs/reports/protocols/phase_targets_20261006_112047_095878/protocol.json'
REGISTRY=ROOT/'outputs/reports/phase_registries/registry_20261006_112203_442094'


def prepare():
    parent=read(PARENT/'result.json'); verify(parent['source_sha256'])
    design=read(PARENT/'next_experiments.json')
    require(digest(PARENT/'result.json')==design['parent_sha256'] and parent['qualified_teacher_recovery_certificates']==0,'Accepted preceding result')
    phase=read(PHASE); verify(phase['source_sha256'])
    require(read(REGISTRY/'result.json')['status']=='complete','Existing fixed registry')
    checks=[]
    for block in parent['blocks']:
        run=Path(block['run'])
        for e in read(run/'episodes.json'):
            if e['condition'] not in ('Z_zero','A_causal_arm'): continue
            t=_load_trace(run,e); gate=OneShotClosingGate(); z=t['arrays']
            saved=np.load(Path(e['cache'])/'queries.npz',allow_pickle=False)
            c='Z_zero' if e['condition']=='Z_zero' else 'B_two_full'
            for q in range(len(t['rows'])):
                w=gate.step(q,z['state'][q],z['base'][q])['window']
                actions,delta,_=compose(z['base'][q],z['raw_delta'][q],c,q,w)
                require(actions.tobytes()==saved['actions'][q].tobytes() and delta.tobytes()==z['applied_delta'][q].tobytes(),'New control exactly reproduces prior actual commands')
                # At the first block the one/full and two/full prescriptions agree.
                if w is not None and q==w[0]:
                    a=compose(z['base'][q],z['raw_delta'][q],'F_first_full',q,w)[0]
                    b=compose(z['base'][q],z['raw_delta'][q],'B_two_full',q,w)[0]
                    require(a.tobytes()==b.tobytes(),'Shared first correction')
            saved.close(); checks.append(dict(task=block['task'],seed=e['seed'],condition=c,queries=e['queries'],exact=True))
    files=[PARENT/'result.json',PARENT/'next_experiments.json',PARENT/'independent_verification.json',PHASE]
    files += [REGISTRY/n for n in ('records.json','manifest.json','result.json')]
    # Freeze all implementation modules used by these new entry points; old files remain immutable.
    files += list((ROOT/'cx_tta').glob('*.py'))
    files += [ROOT/'scripts/eval_arm_prescription.sh',ROOT/'tests/test_arm_prescription.py']
    sources=dict(phase['source_sha256']); sources.update({str(f):digest(f) for f in files})
    out=new_run('arm_prescription',group=('reports','protocols'))
    write_json(out/'CPU_existing_controls.json',dict(passed=True,checks=checks,actual_queries=sum(c['queries'] for c in checks),
        GPU_started=False,old_A_two_full_and_zero_exact=True,source_sha256=parent['source_sha256']))
    protocol=dict(stage='local_arm_prescription_TRAIN',status='frozen_before_new_duration_strength_outcomes',
        tasks=phase['tasks'],seeds=phase['seeds'],registry=str(REGISTRY),registry_protocol=str(PHASE),
        process_blocks=[0,1],conditions=list(CONDITIONS),prescriptions={k:dict(duration_queries=v[0],arm_strength=v[1]) for k,v in PRESCRIPTIONS.items()},
        reference=phase['reference'],training_run=phase['training_run'],checkpoint_sha256=phase['checkpoint_sha256'],
        caps=phase['caps'],robot_joint_limits=phase['robot_joint_limits'],budget_config_sha256=phase['budget_config_sha256'],
        causal_onset='Original first current state/base gripper-decrease >=0.1; next query; no task/seed/GT/outcome in actor',
        gripper_strength=0,compiled=True,internal_noise=True,per_action_observations=True,original_native_budget=True,
        role='Existing four independent TRAIN scenes, previous source outcomes inspected; development not held-out confirmation',
        same_process_controls=True,blocks_independent_OS_process=True,
        ordering='Alternate condition order by scene and reverse across blocks; common sampler within each block',
        comparisons=['zero versus each local prescription at shared onset','first-full versus two-full at exact shared state after first full chunk'],
        labels='Only reached common decision states and complete within-block outcomes; finite prescription benefit, not dense per-query reward',
        smoke='First registered eligible scene; four conditions; maximum five queries each, all censored including native success',
        success_criterion='Report per-scene rescued/harmed and each block. Shorter/weaker must preserve recovery and reduce harm before promotion; all failures do not uniquely prove incorrect direction.',
        maximum_formal_rollouts=32,distinct_TRAIN_scenes=4,no_RGB_or_video=True,
        no_new_GRIP_training_or_router_or_TTA=True,source_sha256=sources)
    write_json(out/'protocol.json',protocol)
    print('PROTOCOL',out/'protocol.json',flush=True)


def main():
    argparse.ArgumentParser(__doc__).parse_args(); prepare()


if __name__=='__main__': main()
