"""New independent functional TRAIN batch for the newly requested three tasks."""
import fcntl
from pathlib import Path
from .common import T,P,C,O,COHORT,TASKS,CANON,read,write,sha,identity,mapped,verify

def freeze_stage():
    assert COHORT=='dev_gated'
    O.mkdir(parents=True,exist_ok=True)
    with (O/'study_freeze.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if(O/'study.json').exists():
            v=read(O/'study.json');verify(v);return v
        proof=O.parent/'CPU_contract_checks_v1.json';checks=read(proof);verify(checks);assert checks['passed']
        actual=read(proof.with_name('CPU_actual_host_exit_v1.json'));assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
        from native_seven_task_gate_expansion_v1.contract import plan, O as EXP
        p=plan();assert p['gated_tasks']==list(TASKS)
        canonical=lambda value:str(CANON/Path(value).relative_to(P))
        template=T/'outputs/native_eval_throughput_v4/branch12/study.json';v=dict(read(template));verify(v)
        v.pop('identity');v['source_sha256']=dict(v['source_sha256']);v['input_sha256']=dict(v['input_sha256'])
        v.update(version=C.name,scope='seven_task_gated_TRAIN_interface_validation',stage=COHORT,scene_role='gated_combination_TRAIN_interface_validation_no_fit',dataset_split='TRAIN',tasks=list(TASKS),task_slots=list(TASKS)*2,
            actual_seed_start=7136200,initial_screen_state={t:dict(next_seed=7136200,accepted=0)for t in TASKS},episodes_per_task=2,cases_per_batch=8,simulation_lanes=12,
            expected_primary_episodes=24,expected_zero_shadow_episodes=8,expected_engineering_reference_episodes=4,logical_batch_count=1,
            official_EVAL_used=False,no_official_EVAL_used=True,new_official_EVAL_scenes=0,formal_success_not_used_for_fitting=True,official_EVAL_excluded_from_fitting=True,
            inherited_stage_GPU_seconds=2580.560,stage_total_GPU_budget_seconds=7200,throughput_comparison_total_budget_seconds=None,
            previous_qualified_comparison_root=None,baseline_comparison_root=None,performance_target_utilization=80.,performance_target_window_seconds=180,
            no_matched_speedup_claim_from_previous_v4=True,future_preparation_each_task=0,
            training=canonical(p['gate_training']),trained_qualification=canonical(p['current_TTA_qualification']),runtime_source=canonical(p['runtime_source']),candidate_checkpoint_sha256=p['current_TTA_checkpoint_sha256'],
            original_qualified_TTA_auxiliary_unchanged=True,new_semantic_interface='native_object_association_v1',gate_candidate=p['gate_contract'],combination_identity='shared250_alpha05_all_v118_plus_qualified_auxiliary_fast_increment_ARM_context_v2',semantic_interface_actual_visual_coverage_not_preclaimed=True,
            functional_validation_not_official100_or_TTA_benefit=True,mandatory_first_case_whole_reference=True)
        from .prepared_recovery import cache
        v['cached_preparation']=cache()
        v['input_sha256'].update(v['cached_preparation']['input_sha256'])
        v['expected_engineering_reference_episodes_maximum_with_regeneration']=8
        v['same_eight_accepted_cases_locked_by_preparation']=True
        for f in C.rglob('*'):
            if f.is_file()and f.suffix in('.py','.sh','.md','.json'):v['source_sha256'][str(f.relative_to(P))]=sha(f)
        for f in list((T/'native_pot_mic_semantics_v1').glob('*'))+list((T/'native_seven_task_gate_expansion_v1').glob('*'))+list((T/'gated_native_audit_v1').glob('*')):
            if f.is_file():v['source_sha256'][str(f)]=sha(f)
        for f in(proof,proof.with_name('CPU_actual_host_exit_v1.json'),template,EXP/'campaign_plan_v1.json',EXP/'CPU_compatibility_v1.json',EXP/'language_coverage_v1.json',T/'outputs/native_eval_throughput_diagnosis_v1/diagnosis.json'):
            v['input_sha256'][str(f)]=sha(f)
        for row in read(T.parent/'frozen/all_candidate_gate_v1_20261101/manifest.json')['files']:
            if row['role']in('algorithm','model'):
                file=mapped(row['original_host_path']);assert sha(file)==row['sha256'];v['source_sha256'][str(file)]=sha(file)
        for f in(EXP/'CPU_paired_heads_v1.json',EXP/'CPU_heads_actual_host_exit_v1.json',EXP/'CPU_gate_composition_v2.json',EXP/'CPU_composition_actual_host_exit_v2.json',EXP/'CPU_composite_actor_v2.json',EXP/'CPU_composite_actual_host_exit_v2.json'):
            v['input_sha256'][str(f)]=sha(f)
        for folder in('declared_reference_audit_v1','gated_handoff_compare_v1','resource_repair_audits_v1'):
            for file in(T/folder).glob('*'):
                if file.is_file():v['source_sha256'][str(file)]=sha(file)
        for file in(T/'outputs/resource_repair_audits_v1/CPU_audit_repair_v1.json',T/'outputs/resource_repair_audits_host_v1/actual_exit.json'):
            v['input_sha256'][str(file)]=sha(file)
        from .prerequisite import prior
        v['input_sha256'].update(prior())
        v['initial_ARM_source_context']='five_zero_event_flags_actual_admitted_q0_age_count_gap'
        for f in(T/'gated_arm_context_v1').glob('*'):
            if f.is_file():v['source_sha256'][str(f)]=sha(f)
        v['identity']=identity(v);verify(v);write(O/'study.json',v);return v
