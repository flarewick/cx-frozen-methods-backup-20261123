"""Coverage and fixed-seed aggregate conclusions after the audited capacity gate."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_pose_capacity import numpy_fk,angles
from .paths import write_json
from .robot_kinematics import RobotFK

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();read=lambda f:json.loads(f.read_text());audit=read(run/'audit.json')
    if not audit['passed']:raise ValueError('Capacity audit required')
    protocol=read(run/'protocol.json');prepared=Path(protocol['prepared']);rows=protocol['rows'];config=read(Path(protocol['config']));robots=[RobotFK(config,s) for s in ('left','right')]
    with np.load(prepared/'data.npz') as z:data={k:z[k].copy() for k in z.files}
    def endpoints(x):return np.stack([numpy_fk(r,x[:,protocol['offsets'],start:start+6].astype(float)) for r,start in zip(robots,(0,7))],2)
    distance=angles(endpoints(data['base']),endpoints(data['target']));valid=np.broadcast_to(data['valid'][:,protocol['offsets'],None],distance.shape);coverage=[]
    for part in ('fit','inner_held'):
        for task in sorted({r['task'] for r in rows}):
            ids=[i for i,r in enumerate(rows) if r['pilot_split']==part and r['task']==task];mask=valid[ids];large=mask&(distance[ids]>=protocol['large_angle_threshold_deg']);changed=np.where(np.any(large,(1,2)))[0]
            coverage.append(dict(partition=part,task=task,episodes=len({rows[i]['group'] for i in ids}),rows=len(ids),valid_endpoints=int(mask.sum()),large_endpoints=int(large.sum()),large_rows=len(changed),large_groups=sorted({rows[ids[j]]['group'] for j in changed}),base_error_under_3deg_fraction=float((distance[ids][mask]<3).mean())))
    write_json(run/'coverage_audit.json',dict(records=coverage,selection='Fixed offsets and >=45deg threshold preregistered before training',scope='Native expert endpoint labels, not executed failure recovery labels',source_sha256={str(f):digest(f) for f in [prepared/'data.npz',prepared/'manifest.json',run/'protocol.json',Path(__file__)]}))
    independent=read(run/'independent_metrics.json');summary=[]
    for task in sorted({r['task'] for r in rows}):
        table=dict(task=task)
        for variant in protocol['variants']:
            ms=[r['metrics'][task] for r in independent['records'] if r['partition']=='inner_held' and r['variant']==variant]
            table[variant]={k:None if ms[0][k] is None else float(np.mean([v[k] for v in ms])) for k in ('base_angle_deg','angle_deg','translation_m','large_angle_deg','large_base_angle_deg','native_action_mse')}
        summary.append(table)
    support=[r for r in coverage if r['partition']=='fit'];large_count=sum(r['large_endpoints'] for r in support);total=sum(r['valid_endpoints'] for r in support)
    large=[r for r in summary if r['proposal_pose']['large_angle_deg'] is not None]
    decision=dict(status='do_not_promote',shared_adapter_parameters=audit['shared_parameter_counts'],complete_variants=9,expert_fit_scenes=48,expert_inner_held_scenes=12,training_large_orientation_support=dict(large_endpoints=large_count,total_endpoints=total,fraction=large_count/total,groups=sum(len(r['large_groups']) for r in support)),task_summary=summary,
        findings=['Official target/frame audit passed; the observed large correction is toe/heel heading, not a quaternion sign or permitted 180-degree symmetry.',
            'Fixed global RNG does not reset native planner internal state. First plans from three fresh identical scenes are array-exact; same-instance subsequent plans differ. Store and reuse a first plan only under a verified identical predecessor state; retain query order/version/state provenance.',
            'All four tasks have worse mean held-expert orientation error for all tested variants versus base. Pose loss reduces deterioration versus proposal BC, but does not create useful large-angle correction.',
            'Removing the small residual output bound alone did not fix the problem. Held large-angle endpoints occur only in one shoe episode (three rows, six correlated arm/offset endpoints), so no broad large-angle or cross-task capability conclusion is possible.',
            'The existing success demonstrations mostly require no or small orientation changes. This supports a supervision/coverage mismatch, not proof that adapters or LoRA intrinsically cannot learn rotation. Ten-epoch fixed pilot does not establish optimized architectural capacity.'],
        next_steps=['Reuse executed teacher traces and legal saved observations to build a phase/source inventory: verified held recovery and actual rotation-to-placement transitions; distinguish static planner endpoints from executed successful recovery labels.',
            'For genuinely missing decision images, compiled base actions or causal histories, authorize/perform only a bounded interface replay. Do not expand identical success-demo or +/-translation collection.',
            'Before broader collection, validate a shared orientation/action-center interface using a few actual recovery sequences, matched BC/rotation/preservation controls, a small-drift negative control and separate scenes. If even fit-state rotation is not learned, debug representation/loss/output capacity before new labels; do not interpret memorized recovery as held-out success.',
            'After the label/interface gate, preselect multiple scene/model/arm configurations and task-held failures plus successful controls. Validate native measured rotation, coupling, release, full outcome and zero/sham pairing under live official compiled inference.',
            'Only after shared offline recovery and preservation generalize, evaluate backbone LoRA or TTA as separate matched contributions.'],
        scope='Small offline shared external adapter capacity diagnostic, not injected backbone LoRA, full training, deployment, recovery success or adapter task holdout. No GT inputs or task-ID rules. Proposal variants deliberately change action support.',
        source_sha256={str(f):digest(f) for f in [run/'audit.json',run/'independent_metrics.json',run/'coverage_audit.json',Path(__file__)]})
    write_json(run/'decision.json',decision);print(json.dumps(dict(status=decision['status'],training_support=decision['training_large_orientation_support'],tasks=summary),indent=2))

if __name__=='__main__':main()
