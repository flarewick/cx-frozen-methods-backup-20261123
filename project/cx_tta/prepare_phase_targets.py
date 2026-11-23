"""Freeze independent TRAIN phase targets before eligibility/source outcomes."""
from pathlib import Path
import argparse
from .paths import ROOT,new_run,write_json
from .audit import digest
from .native_history_cache import read,verify,require
from .causal_arm_protocol import TRAIN
from .evaluate_structured_attribution import pilot_reference
from .train_output_support import BASE

def prepare():
    parent=ROOT/'outputs/reports/diagnostics/causal_mechanisms_20261006_084505_064975'
    design=read(parent/'candidate_completion_design.json');require(design['status']=='design_prepared_not_executed','Prepared target design')
    isolation=read(parent/'next_TRAIN_scene_isolation.json');require(isolation['passed'],'Independent TRAIN scene allocation')
    # Check current metadata again without changing frozen old code or seed files.
    import cx_tta.causal_arm_protocol as old
    previous=old.SEEDS
    try:old.SEEDS=design['seeds'];current=old.scene_isolation()
    finally:old.SEEDS=previous
    files=[parent/'candidate_completion_design.json',parent/'result.json',parent/'feature_recovery_audit.json',
        TRAIN/'A_bounded_clipped_seed0.pt',ROOT/'cx_tta/prepare_phase_targets.py',ROOT/'cx_tta/phase_target_policy.py',
        ROOT/'cx_tta/collect_phase_targets.py',ROOT/'cx_tta/screen_phase_targets.py',ROOT/'cx_tta/local_benefit_cache.py',
        ROOT/'tests/test_phase_target_policy.py',ROOT/'scripts/qualify_phase_targets.sh',ROOT/'scripts/screen_phase_targets.sh']
    references={}
    for task in design['independent_TRAIN_tasks']:
        _,_,_,ref=pilot_reference(task,BASE);references[task]=ref;files.append(Path(ref['run'])/'robotwin/task_config/_eval_step_limit.yml')
    q=read(TRAIN/'training_protocol.json');verify(q['source_sha256'])
    sources=dict(q['source_sha256']);sources.update({str(f):digest(f) for f in files})
    run=new_run('phase_targets',group=('reports','protocols'));write_json(run/'scene_isolation.json',current)
    p=dict(stage='phase_target_qualification',status='frozen_before_independentTRAINeligibility_and_policyoutcomes',tasks=design['independent_TRAIN_tasks'],
        seeds=design['seeds'],reference=references,training_run=str(TRAIN),checkpoint_sha256=digest(TRAIN/'A_bounded_clipped_seed0.pt'),
        caps=q['reference_scales_native_units'],robot_joint_limits=q['robot_joint_limits'],no_RGB_or_video=True,
        budget_config_sha256={t:digest(Path(ref['run'])/'robotwin/task_config/_eval_step_limit.yml') for t,ref in references.items()},
        source_modes=['Z_zero','A_causal_arm'],probe_cases=['C_base','A_grip','T_stage_grip'],
        selection='First current functional geometry satisfied but native task incomplete,after prescribedsource early armwindow; EACH source has at most1point. No finaloutcome/candidate success inselection.',
        canonical_teacher_goal='Same-scene official successful expert terminal left/right command values,screened before policyoutcomes. Not fixedcup/openhardcode.',
        teacher_primitive='Current base arms unchanged; GRIP clamp(expert_goal-currentbase,+/-0.2) at one query. Then official realtime base continuation under remaining original400/500budget.',
        source_Cbase_reference='Completed source outcome is samepostpoint base continuation; prove all sourceapplieddelta afterpoint zero.',
        pairing='Exact initialreg,prefixphysics/RGB,previousrawjoint/command andCPU/CUDA/Python/NumPy RNG restored; firstfork all9/base/raw/rng hashesexact. Shared compiledsampler throughouttask process.',
        max_full_rollouts_if_all8eligible_and16points=48,model_fields='Exactly nine legalfields pluscurrentbase; GT/task/seed/teachergoal excludedfromNNforward.',
        qualification='Teacherfullsuccess with sourceCbase fullfailure; multi-source points counted per distinctTRAIN scene. Teachersuccess never adaptergain.',
        success_preservation='Keep all actualqueries from successfulsource andqualifiedteachercontinuations,with actualexecutedmask; nevercopyepisode outcome asquerybenefit.',
        no_training_approval_from_collector=True,current6confirmation_scenes_never_fit=True,
        training_after='CPU physical/target/source acceptance then freeze commondata/PCA/actualinit/updatebudget beforepairedsharedadaptertraining;router andTTA aftercandidate/preservation.',
        source_sha256=sources)
    write_json(run/'protocol.json',p)
    write_json(run/'CPU_preparation.json',dict(scene_isolation=True,composition_contracts=4,
        test_invocation_note='tests is not a package; use unittest discover -s tests -p test_phase_target_policy.py; initial import invocation had no tests.',
        teacher_certificates=0,training_updates=0,commands_ready=False))
    print(run/'protocol.json',flush=True);return run/'protocol.json'
def main():argparse.ArgumentParser(__doc__).parse_args();prepare()
if __name__=='__main__':main()

