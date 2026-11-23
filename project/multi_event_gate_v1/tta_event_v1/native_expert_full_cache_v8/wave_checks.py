from pathlib import Path
import json,hashlib
from .paths import T,O,C,read,write,sha

def main():
 previous=read(T/'outputs/native_expert_full_failure_v7/report.json');assert previous['passed']and previous['actual_host_exit_code']==1 and previous['remaining_whole_original_cases']==26
 indices=previous['remaining_original96_indices'];assert len(indices)==len(set(indices))==26 and indices==sorted(indices)
 old=read(T/'outputs/native_scene_waves_v1/CPU_checks.json');assert old['passed']and old['weakreference_cycle_release_verified']
 for p,h in old['source_sha256'].items():assert sha(p)==h
 release=previous['completed_scene_free_wave'];assert release['live_tasks_after_gc']==release['live_scenes_after_gc']==0 and release['model_parameter_pointers_and_RNG_unchanged']
 proof=dict(passed=True,waves=[indices],all_remaining_original_cases_once=True,weakreference_cycle_release_verified=True,prior_actual_scene_free_release_verified=True,
 reason='Entireremaining26 in onecontinuouscohort: remove intermediate scene-free/coldsetup interval that produced missingprocessmetrics; terminal release remains. No metricwaiver/reset or shortjob splitting.',
 memory_fit_not_assumed=True,strict24563cap_and34floor_unchanged=True,source_sha256={str(p):sha(p)for p in [Path(__file__),C/'runtime.py',T/'native_scene_waves_v1/lifecycle.py']},
 input_sha256={str(T/'outputs/native_expert_full_failure_v7/report.json'):sha(T/'outputs/native_expert_full_failure_v7/report.json'),str(T/'outputs/native_scene_waves_v1/CPU_checks.json'):sha(T/'outputs/native_scene_waves_v1/CPU_checks.json')})
 write(O/'wave_checks.json',proof);print('ENTIRE_REMAINING26_CONTINUOUS_COHORT_CPU_CHECKS_PASS')
if __name__=='__main__':main()
