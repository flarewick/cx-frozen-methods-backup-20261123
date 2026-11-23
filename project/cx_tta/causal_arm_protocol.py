"""Freeze causal proxy, scene isolation and independent process-block design."""
import argparse
import json
import os
from pathlib import Path
import re
from .audit import digest
from .paths import ROOT,new_run,write_json
from .native_history_cache import read,verify,require
from .evaluate_structured_attribution import pilot_reference
from .train_output_support import BASE

DESIGN=ROOT/'outputs/reports/attribution/local_components_accepted_20261006_042530_435693/next_experiments.json'
TRAIN=ROOT/'outputs/effect_learning/output_support_20261005_082307_793921'
TASKS=('place_empty_cup','beat_block_hammer')
SEEDS=[501000,501001,501002,501003]
SOURCE_FILES=('cx_tta/causal_arm_gate.py','cx_tta/causal_arm_protocol.py','cx_tta/causal_arm_analysis.py',
    'cx_tta/evaluate_causal_arm.py','cx_tta/screen_causal_scenes.py',
    'scripts/eval_causal_arm.sh','scripts/screen_causal_scenes.sh','tests/test_causal_arm_gate.py')


def identifiers(value,task=None):
    found=set()
    if isinstance(value,dict):
        task=value.get('task',value.get('task_name',task))
        for k,v in value.items():
            if k in ('seed','scene_seed') and type(v) is int:found.add((task or '*',v))
            if k=='seeds' and isinstance(v,list):found.update((task or '*',s) for s in v if type(s) is int)
            found.update(identifiers(v,task))
    elif isinstance(value,list):
        for x in value:found.update(identifiers(x,task))
    elif isinstance(value,str):
        m=re.match(r'^(beat_block_hammer|place_empty_cup|place_shoe|turn_switch):(\d+)(?:\D|$)',value)
        if m:found.add((m[1],int(m[2])))
    return found


def scene_isolation():
    protected=[BASE/'protocol.json',BASE/'training_protocol.json',
        ROOT/'outputs/feature_caches/structured_readout_20261003_065755_140725/protocol.json',
        ROOT/'outputs/feature_caches/native_history_20261005_030329_014198/protocol.json']
    first=read(BASE/'protocol.json');projection=read(protected[2])
    # Include surviving JSON provenance of fixed background and PCA sources.
    for parent in (first,projection):
        for name in parent['source_sha256']:
            file=Path(name)
            if file.is_file() and file.name in ('protocol.json','manifest.json','prepared.json','preparation.json'):
                protected.append(file)
    for branch in ('collections','collections_effect','diagnostics','evaluations','feature_caches','effect_learning'):
        for folder,dirs,files in os.walk(ROOT/'outputs'/branch,followlinks=False):
            dirs[:]=[d for d in dirs if d not in ('robotwin','runtime','cache','source_snapshot','evaluation_source_snapshot') and not Path(folder,d).is_symlink()]
            if 'manifest.json' in files:protected.append(Path(folder)/'manifest.json')
    protected=sorted(set(protected));known=set();sources={}
    for file in protected:
        try:x=read(file)
        except (ValueError,OSError):raise ValueError('Unreadable existing provenance '+str(file))
        known.update(identifiers(x));sources[str(file)]=digest(file)
    collisions=[(t,s) for t in TASKS for s in SEEDS if (t,s) in known or ('*',s) in known]
    require(not collisions,'New scene seeds already present in inspected support/manifests')
    return dict(passed=True,new_scenes={t:SEEDS for t in TASKS},sources_checked=len(protected),
        observed_existing_scene_identifiers=len(known),collisions=collisions,source_sha256=sources,
        scope='Surviving background/PCA/native metadata and current source/evaluation/collection manifests;33 inherited raw-source gaps retained, not reconstructed')


def prepare():
    d=read(DESIGN);verify(d['source_sha256']);require(d['status']=='design_prepared_not_executed','Sealed prospective design')
    require(d['proposed_next_scene_screen']['scene_seeds']==SEEDS,'Fixed new scene allocation')
    isolation=scene_isolation();sources=dict(isolation['source_sha256']);references={}
    for task in TASKS:
        _,_,_,evidence=pilot_reference(task,BASE);references[task]=evidence
    q=read(TRAIN/'training_protocol.json');sources.update(q['source_sha256'])
    paths=[DESIGN,TRAIN/'A_bounded_clipped_seed0.pt',TRAIN/'training_audit.json']+[ROOT/name for name in SOURCE_FILES]
    for task,evidence in references.items():
        paths += [Path(evidence['run'])/'robotwin/task_config/_eval_step_limit.yml']
    sources.update({str(f):digest(f) for f in paths})
    out=new_run('causal_arm',group=('reports','protocols'))
    write_json(out/'scene_isolation.json',isolation)
    p=dict(stage='causal_arm_confirmation',status='frozen_before_new_scene_outcomes',design=str(DESIGN),
        tasks=list(TASKS),seeds=SEEDS,process_blocks=[0,1],conditions=['Z_zero','A_causal_arm'],
        training_run=str(TRAIN),checkpoint_sha256=digest(TRAIN/'A_bounded_clipped_seed0.pt'),
        trigger=dict(threshold=.1,source='Current commanded gripper state minus min(clipped current32-step VLA base forecast)',
            latch_first=True,delay_queries=1,duration_queries=2,no_future_observation=True,action_clipping_changed=False),
        smoke_fixed_windows=dict(place_empty_cup=[1,3],beat_block_hammer=[2,4]),
        reference=references,scales=q['reference_scales_native_units'],robot_joint_limits=q['robot_joint_limits'],
        budget_config_sha256={t:digest(Path(e['run'])/'robotwin/task_config/_eval_step_limit.yml') for t,e in references.items()},
        isolation=str(out/'scene_isolation.json'),isolation_sha256=digest(out/'scene_isolation.json'),
        compiled=True,internal_noise=True,per_action_observations=True,original_native_budget=True,
        window_absent='Keep current base throughout; report no trigger or termination before window',
        gripper_correction=0,arm_strength_when_active=1,arm_only_all_tasks=True,
        screening='Official expert-valid plan+check_success, deterministic unseen grounded instruction; all fixed4 seeds attempted before candidate outcomes, invalid attempts kept and not replaced',
        process_design='Each block separate OS process/fresh compiled sampler/private cache. In-block pairedzero/candidate same sampler.1 repeat per block; reverse order across blocks; same scene and sampler seeds, no cross-block outcome pairing.',
        maximum_full_rollouts=32,no_new_training_or_router_or_TTA=True,no_RGB_or_video=True,
        scientific_scope='New4 scenes/task small development confirmation. Handcrafted causal proxy proposed after old data; not learned gate or majority-task paper benchmark.',
        source_sha256=sources)
    write_json(out/'protocol.json',p);print('PROTOCOL',out/'protocol.json',flush=True);return out/'protocol.json'


def main():
    argparse.ArgumentParser(__doc__).parse_args();prepare()

if __name__=='__main__':main()
