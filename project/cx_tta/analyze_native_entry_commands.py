"""Read accepted command records, without reconstructing unrecorded physics."""
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .audit import digest
from .native_history_cache import ROOT,arrays,read,require,verify
from .paths import new_run,write_json
from .robot_kinematics import RobotFK

ACCEPTED=ROOT/'outputs/reports/protocols/native_entry_bounded_v2_20261005_045526_497512/accepted_results.json'
CONDITIONS=('L0','teacher_prefix_raw','teacher_prefix_clip','teacher_prefix_arm_direction')
ARM_INDICES=[0,1,2,3,4,5,7,8,9,10,11,12]


def pose_errors(robot,expected,proposed):
    desired=np.stack([robot.pose(x) for x in expected])
    actual=np.stack([robot.pose(x) for x in proposed])
    translation=np.linalg.norm(desired[:,:3,3]-actual[:,:3,3],axis=1)*1000
    relative=desired[:,:3,:3]@actual[:,:3,:3].transpose(0,2,1)
    rotation=Rotation.from_matrix(relative).magnitude()*180/np.pi
    return dict(commanded_tip_translation_error_mean_mm=float(translation.mean()),
        commanded_tip_translation_error_max_mm=float(translation.max()),
        commanded_tip_rotation_error_mean_deg=float(rotation.mean()),
        commanded_tip_rotation_error_max_deg=float(rotation.max()))


def analyze(run,sources):
    manifest=read(run/'manifest.json');protocol=Path(manifest['protocol']);p=read(protocol)
    verify(p['source_sha256']);a=read(run/'CPU_audit.json');require(a['passed'],'Prior independent audit required')
    verify(a['source_sha256']);r=read(run/'result.json');live=arrays(run/'first_live_base_target.npz')
    target=arrays(manifest['point']['target']);valid=live['valid'];base=live['base'];teacher=live['target']
    caps=np.asarray(p['caps'],np.float32);delta=teacher-base
    over=(np.abs(delta)>caps+2e-6)&valid[:,None]
    valid_count=int(valid.sum());require(valid_count>0,'No teacher supervision')
    robots=[RobotFK(read(run/'resolved_config.json'),side) for side in ('left','right')]
    prefix_metrics=[]
    for offset,robot,side in zip((0,7),robots,('left','right')):
        ids=range(offset,offset+6)
        prefix_metrics.append(dict(side=side,raw_arm_residual_max_rad=float(np.abs(delta[valid,offset:offset+6]).max()),
            over_cap_joints=[dict(native_action_column=j,joint=robot.names[j-offset],
                changed_steps=(np.flatnonzero(over[:,j])+1).tolist(),
                max_raw_residual_rad=float(np.abs(delta[valid,j]).max())) for j in ids if over[:,j].any()]))
    condition_metrics={}
    for name in CONDITIONS:
        saved=arrays(run/f'r0_{name}.npz');actions=saved['actions'];bases=saved['base']
        records=[json.loads(x) for x in (run/f'r0_{name}.jsonl').read_text().splitlines()]
        episodes=[e for e in r['episodes'] if e['condition']==name]
        prefix=actions[0];near=np.max(np.abs(prefix-teacher),axis=1)<=2e-6
        observed_padding=max(0,records[0]['executed']-valid_count)
        require(np.array_equal(prefix[~valid],base[~valid]),'Invalid teacher tail was not base')
        metrics=dict(successes=sum(e['success'] for e in episodes),total=len(episodes),
            executed_native_counts=[e['executed_actions'] for e in episodes],
            queries_per_repeat=[e['queries'] for e in episodes],
            terminal_in_first_prefix=[e['prefix_native_success'] for e in episodes],
            first_prefix_executed=records[0]['executed'],
            valid_teacher_commands_preserved_within_2e6=int((near&valid).sum()),
            first_prefix_unmodified_padding_actions_actually_executed=observed_padding,
            gripper_command_error_max=float(np.abs(prefix[valid][:,[6,13]]-teacher[valid][:,[6,13]]).max()),
            per_arm_commanded_tip_error_to_teacher={side:pose_errors(robot,teacher[valid,start:start+6],
                prefix[valid,start:start+6]) for start,robot,side in zip((0,7),robots,('left','right'))})
        if observed_padding:
            jump=prefix[valid_count]-prefix[valid_count-1]
            metrics['valid_to_padding_command_jump']=dict(arm_max_rad=float(np.abs(jump[ARM_INDICES]).max()),
                gripper_max=float(np.abs(jump[[6,13]]).max()))
        if len(bases)>1:
            last=prefix[records[0]['executed']-1];next_base=bases[1,0]
            metrics['next_query_command_jump']=dict(arm_max_rad=float(np.abs((next_base-last)[ARM_INDICES]).max()),
                gripper_max=float(np.abs((next_base-last)[[6,13]]).max()),
                measured_physical_undo=False,
                note='Different reached state and fresh query; numeric command jump alone is not physical undo or same-state benefit.')
        condition_metrics[name]=metrics
        for path in (run/f'r0_{name}.npz',run/f'r0_{name}.jsonl'):sources[str(path)]=digest(path)
    for path in (run/'manifest.json',protocol,run/'result.json',run/'CPU_audit.json',
        run/'first_live_base_target.npz',Path(manifest['point']['target']),run/'resolved_config.json'):
        sources[str(path)]=digest(path)
    for robot in robots:sources[str(robot.urdf)]=digest(robot.urdf)
    return dict(task=manifest['point']['task'],seed=manifest['point']['seed'],
        native_count=manifest['point']['native_count'],run=str(run),valid_teacher_actions=valid_count,
        within_original_caps_actions=valid_count-int(over.any(-1).sum()),
        over_cap_steps_one_based=(np.flatnonzero(over.any(-1))+1).tolist(),
        teacher_plan_ids=target['plan_ids'][valid].tolist(),
        raw_gripper_residual_max=float(np.abs(delta[valid][:,[6,13]]).max()),
        gripper_caps_active=bool(over[:,[6,13]].any()),per_arm_residuals=prefix_metrics,
        conditions=condition_metrics,qualified_bounded_recovery=r['decision']['qualified'],
        scope='Single selected TRAIN state, two deterministic reverse-order repeats. Teacher-prefix target diagnostic, not adapter or population success.')


def main():
    accepted=read(ACCEPTED);require(accepted['status']=='complete_independently_accepted','Completed evidence required')
    verify(accepted['source_sha256']);sources={str(ACCEPTED):digest(ACCEPTED),str(Path(__file__)):digest(Path(__file__))}
    rows=[analyze(Path(accepted[key]),sources) for key in ('local_shoe_run','reused_external_switch_run')]
    out=new_run('native_entry_commands',group=('reports','attribution'))
    report=dict(status='complete_CPU_command_analysis',rows=rows,source_sha256=sources,
        actor_or_sampler_or_simulation_called=False,new_training_started=False,
        corrections_are_joint_space=True,
        FK_scope='URDF move_group endpoint commanded by the joint vectors, not actual motion, object contact, force or physical duration.',
        no_per_action_physics_numeric_record=True,
        interpretation='Limit projection did not preserve successful switch prefix. Last five valid commands exceed arm caps; gripper limiter never activates. '
            'Padding base commands were actually executed only by failing bounded branches. Neither effect has been isolated as the unique physical cause. '
            'Shoe raw-prefix handoff also fails, so cap alone cannot explain both tasks. No router/TTA or added-budget remedy proven.')
    write_json(out/'result.json',report)
    print('REPORT',out/'result.json',flush=True)
    for row in rows:
        print(json.dumps(dict(task=row['task'],over_cap_steps=row['over_cap_steps_one_based'],
            gripper_caps_active=row['gripper_caps_active'],
            condition_results={k:(v['successes'],v['total']) for k,v in row['conditions'].items()}),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
