"""Qualify native-entry target alignment, not bounded physical recovery benefit.

Rebase surviving absolute teacher commands onto actual recorded L0 actions.
Preserve previous feasibility results and avoid rerunning their experiments.
"""
import argparse
from pathlib import Path
import numpy as np

from .audit import digest
from .paths import ROOT,new_run,write_json
from .native_history_cache import read,arrays,verify,require,BASE,ACCEPTED
from .audit_local_geometry import joint_bounds
from .robot_kinematics import RobotFK

ARMS=(0,1,2,3,4,5,7,8,9,10,11,12)
CAPS=np.asarray([.1]*6+[.2]+[.1]*6+[.2],np.float32)


def arm_direction_projection(base,target):
    """Independent arm-vector scaling and gripper clip; diagnostic only."""
    delta=np.asarray(target,np.float32)-np.asarray(base,np.float32)
    out=np.clip(delta,-CAPS,CAPS)
    for start in (0,7):
        d=delta[...,start:start+6]
        scale=np.maximum(1.,np.max(np.abs(d)/CAPS[start:start+6],axis=-1,keepdims=True))
        out[...,start:start+6]=d/scale
    return np.asarray(base,np.float32)+out


def residual_metrics(base,target,valid):
    count=int(valid.sum());require(count>0,'No valid teacher actions')
    delta=target-base;over=(np.abs(delta)>CAPS)&valid[:,None]
    coord=base+np.clip(delta,-CAPS,CAPS);direction=arm_direction_projection(base,target)
    def cosine(a,b):
        a=a[valid][:,ARMS].astype(float).ravel();b=b[valid][:,ARMS].astype(float).ravel()
        norm=np.linalg.norm(a)*np.linalg.norm(b)
        return None if not norm else float(np.dot(a,b)/norm)
    return dict(valid_actions=count,fully_inside_caps_actions=int((~over.any(-1)&valid).sum()),
        over_cap_actions=int((over.any(-1)&valid).sum()),max_normalized_raw_residual=float((np.abs(delta[valid])/CAPS).max()),
        raw_arm_rms=float(np.sqrt(np.mean(delta[valid][:,ARMS]**2))),
        raw_gripper_rms=float(np.sqrt(np.mean(delta[valid][:,[6,13]]**2))),
        coordinate_clip_arm_cosine=cosine(delta,coord-base),
        arm_direction_projection_cosine=cosine(delta,direction-base),
        coordinate_clip_command_rmse_to_teacher=float(np.sqrt(np.mean((coord[valid]-target[valid])**2))),
        arm_direction_projection_command_rmse_to_teacher=float(np.sqrt(np.mean((direction[valid]-target[valid])**2))),
        projection_utility_or_physical_success_claim=False)


def qualify(run):
    run=Path(run).resolve();p=read(run/'protocol.json');verify(p['source_sha256'])
    # This qualification can run while encoding; it needs no GPU cache output.
    parent=read(BASE/'protocol.json');y=arrays(BASE/'labels.npz')
    origins=read(ACCEPTED/'support_audit.json')['first_L0_entry_matches']
    lookup={(x['task'],x['seed'],x['native_count']):x for x in origins}
    rows=[];payloads=[];sources={str(run/'protocol.json'):digest(run/'protocol.json'),
        str(Path(__file__)):digest(Path(__file__)),str(BASE/'protocol.json'):digest(BASE/'protocol.json'),
        str(BASE/'labels.npz'):digest(BASE/'labels.npz')}
    robots=[RobotFK(read(p['robot_config']),side) for side in ('left','right')]
    for index,row in enumerate(p['rows']):
        if row['target_kind']!='unqualified_teacher_entry':continue
        entry=lookup[row['task'],row['seed'],row['native_count']];old=entry['training_row']
        require(row['split']=='train_candidate','Held-out recovery supervision')
        original=parent['rows'][old]
        require(original['task']==row['task'] and original['group']==f"{row['task']}:{row['seed']}"
            and original['frame']==row['native_count'],'Wrong teacher entry mapping')
        path=Path(entry['target']);require(digest(path)==entry['target_sha256'],'Teacher target changed')
        target=arrays(path);actual_base=arrays(row['base'])['base_action']
        valid=target['valid'];command=target['target'];counts=target['source_native_counts']
        require(command.shape==(32,14) and command.dtype==np.float32,'Teacher command shape/coordinates')
        require(valid.shape==(32,) and valid.dtype==bool and counts.shape==(32,),'Teacher valid/count contract')
        n=int(valid.sum());require(np.array_equal(valid,np.arange(32)<n),'Teacher nonprefix mask')
        require(np.array_equal(counts[valid],row['native_count']+np.arange(1,n+1)),'Teacher action-time alignment')
        require(row['native_count']+n<=(500 if row['task']=='place_shoe' else 400),'Teacher changes native budget')
        require(np.isfinite(command[valid]).all(),'Teacher nonfinite valid command')
        require(np.array_equal(command[valid],y['target'][old,valid]) and np.array_equal(valid,y['valid'][old]),
            'Surviving absolute teacher command/valid differs from original training label')
        for arm,start in enumerate((0,7)):
            bounds=joint_bounds(robots[arm]);q=command[valid,start:start+6]
            require(not ((q<bounds[:,0])|(q>bounds[:,1])).any(),'Teacher violates URDF command bounds')
        require(not ((command[valid][:,[6,13]]<0)|(command[valid][:,[6,13]]>1)).any(),'Teacher gripper target outside native range')
        m=residual_metrics(actual_base,command,valid);old_m=residual_metrics(y['base'][old],command,valid)
        old_delta=command-y['base'][old];new_delta=command-actual_base
        scalar=np.dot(old_delta[valid].astype(float).ravel(),new_delta[valid].astype(float).ravel())
        norm=np.linalg.norm(old_delta[valid])*np.linalg.norm(new_delta[valid])
        rows.append(dict(cache_row=index,original_training_row=old,task=row['task'],seed=row['seed'],
            native_count=row['native_count'],current_signature_matched=True,actual_base_rebased=True,
            target_alignment_and_native_domain_passed=True,teacher_source_RGB_available=False,
            new_base_metrics=m,old_cache_base_metrics=old_m,
            old_vs_rebased_delta_cosine=float(scalar/norm) if norm else None,
            bounded_physical_benefit_verified=False,approved_for_new_recovery_training=False,
            source_teacher_commands_are_absolute=True,stage_plan_ids_offline_only=target['plan_ids'][valid].tolist()))
        payloads.append(dict(base=actual_base,target=command,valid=valid,source_native_counts=counts,
            coordinate_clip=actual_base+np.clip(new_delta,-CAPS,CAPS),
            arm_direction_projection=arm_direction_projection(actual_base,command)))
        sources[str(path)]=digest(path);sources[row['base']]=digest(Path(row['base']))
    require(len(rows)==11,'Expected11 entry candidates')
    # Carry forward the existing one-point physical evidence, not re-prove it.
    old_dir=ROOT/'outputs/reports/audits/teacher_common_prefix_stage_20261001_122721_988239'
    old_report=read(old_dir/'summary.json');require(old_report['audit_passed'],'Historical summary certificate missing')
    sources[str(old_dir/'summary.json')]=digest(old_dir/'summary.json')
    known=dict(task='place_shoe',seed=320006,native_count=64,
        scope='Previously accepted one-point diagnostic summary; original raw projection run has since been retired',
        outcomes={x['branch']:x['success'] for x in old_report['records']},
        repeated_this_round=False,not_applied_to_new421xxx_scenes=True)
    out=new_run('native_targets',group=('reports','audits'))
    np.savez_compressed(out/'teacher_entry_candidates_NOT_TRAINING.npz',
        **{key:np.stack([x[key] for x in payloads]) for key in payloads[0]},
        cache_rows=np.asarray([x['cache_row'] for x in rows]),
        recovery_training_approved=np.zeros(11,bool))
    report=dict(status='complete_alignment_only',passed=True,cache_run=str(run),rows=rows,
        alignment_native_domain_passed=11,bounded_physical_benefit_qualified=0,new_recovery_training_allowed=False,
        valid_teacher_actions=sum(x['new_base_metrics']['valid_actions'] for x in rows),
        fully_inside_caps_actions=sum(x['new_base_metrics']['fully_inside_caps_actions'] for x in rows),
        previous_physical_evidence=known,source_sha256=sources,
        gpu_used=False,model_training_started=False,
        interpretation='Absolute target timing/domain is valid; changing the sampled base changes the residual. '
            'Static cap and cosine diagnostics are not bounded recovery certificates. Do not reuse olddelta on nativebase. '
            'No new training until selected native-fork benefit is verified; existing320006 result is reused, not rerun.')
    write_json(out/'result.json',report)
    print('TARGET_REPORT',out,flush=True)
    print('ALIGNED11 BOUNDED_BENEFIT_QUALIFIED0',flush=True)
    return out


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True)
    qualify(p.parse_args().run)


if __name__=='__main__':main()
