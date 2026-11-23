"""Real terminal/exit/release prerequisites, never episode counters alone."""
import json
from pathlib import Path
from .common import T,read,mapped,verify,sha
from .configuration import GROUPS

def qualified_stage(folder):
    folder=Path(folder)
    if folder==(T/'outputs/gated_postexit_recovery_v2/qualified_TRAIN'):
        q=read(folder/'qualification.json');verify(q)
        assert q['passed']and q['complete']and q['actual_original_outer_host_exit_code']==1 and q['original_failure_preserved']
        assert q['actual_GPU_child_exit_code']==q['actual_CPU_recovery_exit_code']==0 and q['whole_native_branches']==36 and q['full_whole_controls_exact']and q['resource180_raw_exact']
        assert q['qualification_kind']=='complete_normal_GPU_child_then_separate_normal_CPU_audit_v1'
        root=mapped(q['root']);spec=read(root/'protocol.json');verify(spec);assert spec['identity']==q['protocol_identity']and spec['candidate_checkpoint_sha256']==q['checkpoint_sha256']and spec['dataset_split']=='TRAIN'
        return root,spec
    a=read(folder/'actual_host_wait_exit.json')
    assert a['exit_code']==0 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
    root=mapped(read(folder/'execution_state.json')['root']);spec=read(root/'protocol.json');verify(spec)
    for file in('terminal_audit.json','CPU_governor180_resource_audit.json','CPU_episode_audit.json','CPU_closed_expert_task_release_audit.json'):
        audit=read(root/file);assert audit['passed'],file;verify(audit)
    complete=read(folder/'pilot_complete.json');assert complete['all_resource_and_native_controls_passed'];verify(complete)
    release=json.loads((root/'attempt/resources.jsonl').read_text().splitlines()[-1]);assert release['own_gpu_released']and not release['remaining_own_gpu_pids']
    assert spec['inherited_stage_GPU_seconds']+release['cleanup_seconds_after_start']<=spec['stage_total_GPU_budget_seconds']
    assert read(root/'attempt/barrier/normal_child_exit.json')['exit_code']==0
    return root,spec

def TRAIN_prerequisites():
    result={}
    for group,cfg in GROUPS.items():
        folder=T/'outputs'/cfg['TRAIN_version']/cfg['TRAIN_stage'];root,spec=qualified_stage(folder)
        assert spec['tasks']==list(cfg['tasks'])and spec['dataset_split']=='TRAIN'
        result[group]=dict(folder=str(folder),root=str(root),protocol_identity=spec['identity'],model_sha256=spec['candidate_checkpoint_sha256'])
    return result
