"""Read-only identities for the newly requested two groups; not a GPU launcher."""
import hashlib,json
from pathlib import Path

T = Path(__file__).resolve().parent.parent
P = T.parents[1]
W = T.parent
O = T/'outputs/native_seven_task_gate_expansion_v1'
PLAIN_TASKS = ('lift_pot','handover_mic','stack_bowls_two')
GATED_TASKS = ('beat_block_hammer','turn_switch','adjust_bottle','open_laptop')
GATE_FREEZE = W/'frozen/all_candidate_gate_v1_20261101'
GATE_ADAPTER = W/'outputs/training_diagnostics/shared6_vs_single_20261025/shared6_lambda0p0/best_scaled.pt'
GATE_MODEL = W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4/fixed_ensemble_TRAIN472_v118/diagnostic_gate.json'
GATE_TRAINING = W/'outputs/task_adapters/shared6_2642e508106d2049/training'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def mapped(p):
    p=Path(p)
    for old in ('/data/qiufw/cx_migration/cx_new_direction_guide_TTA','/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA'):
        if p.is_relative_to(old):return P/p.relative_to(old)
    return p

def plan():
    prior=read(T/'outputs/native_calibrated_five100_v33/official_019_p1/study.json')
    q=mapped(prior['trained_qualification'])
    frozen=read(GATE_FREEZE/'manifest.json');c=frozen['contract']
    assert c['step']==250 and c['alpha']==.5 and c['feature_dimensions']==593
    assert sha(GATE_ADAPTER)==c['adapter_sha256'] and sha(GATE_MODEL)==c['gate_sha256']
    return dict(version=Path(__file__).parent.name,scope='user_requested_new_seven_task_campaign_preparation',official_seed=0,actual_seed_pool_start=100000,episodes_each_task=100,task_config='demo_randomized',
        plain_tasks=list(PLAIN_TASKS),gated_tasks=list(GATED_TASKS),expected_TTA_episodes=700,
        controls={'plain':'same-scene frozen M_event_full','gated':'same-scene frozen step250 alpha0.5 full + all-event v118 gate','zero_shadow':'whole native first-case controls per task and new compiled session'},
        expected_primary_TTA_and_paired_control_episodes=1400,
        current_TTA_qualification=str(q),current_TTA_checkpoint_sha256=read(q)['training_checkpoint_sha256'],
        plain_training=str(mapped(prior['training'])),gate_training=str(GATE_TRAINING),gate_adapter=str(GATE_ADAPTER),gate_model=str(GATE_MODEL),gate_contract=c,
        runtime_source=str(mapped(prior['runtime_source'])),event_config=prior['event_config'],
        new_semantic_interface='native_pot_mic_semantics_v1',original_TTA093_and_completed_official500_unchanged=True,
        gate_response_head_compatibility_must_be_proven=True,gate_rejection_must_not_be_bypassed_by_feedback_application=True,
        all_current_GPU_rules_required=True,performance_goal80_not_yet_achieved=True,
        GPU_launcher_ready=False,official_GPU_started=False,
        asset_sha256={str(p):sha(p) for p in (q,GATE_FREEZE/'manifest.json',GATE_ADAPTER,GATE_MODEL)},
        no_new_official_scene_outcomes_used_for_fitting_or_selection=True)
