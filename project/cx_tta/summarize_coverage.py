"""Describe all fixed S1v endpoints and failed gates without model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import write_json


def read(p):
    return json.loads(p.read_text())


def failures(summary):
    failed = []
    for group in summary['groups']:
        seam = group['seam_error_rmse_rad']
        if seam is not None and seam > .03:
            failed.append(dict(group=group['group'],arm='both',metric='seam_error_rmse_rad',value=seam,limit=.03))
        for side, arm in group['arms'].items():
            for key, limit in [('angle_deg',3),('position_m',.01),('arm_rmse_rad',.05),('gripper_mae',.05),
                               ('velocity_error_rmse_rad',.03),('gripper_outside_margin',0),('joint_limit_violations',0)]:
                if arm[key] is not None and arm[key] > limit:
                    failed.append(dict(group=group['group'],arm=side,metric=key,value=arm[key],limit=limit))
            key='pose_under_3deg_1cm_fraction'
            if arm[key] < .9:
                failed.append(dict(group=group['group'],arm=side,metric=key,value=arm[key],minimum=.9))
    return failed


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve()
    if not read(run/'audit.json')['passed'] or not read(run/'coverage_audit.json')['passed']:
        raise ValueError('Completed independent audits required')
    p=read(run/'protocol.json');r=read(run/'result.json');d=read(run/'decision.json')
    final=[x for x in r['reports'] if x['update']==max(p['updates'])]
    conditions=[]
    for condition in p['conditions']:
        reports=[x for x in final if x['condition']==condition]
        decision=next(x for x in d['conditions'] if x['condition']==condition)
        groups={}
        for group in reports[0]['metrics']['recovery']['groups']:
            name=group['group'];matching=[next(g for g in x['metrics']['recovery']['groups'] if g['group']==name) for x in reports]
            groups[name]=dict(
                seams_rad=[g['seam_error_rmse_rad'] for g in matching],
                arms={side:{key:[g['arms'][side][key] for g in matching] for key in
                    ('pose_under_3deg_1cm_fraction','angle_deg','position_m','gripper_mae','gripper_outside_margin')}
                    for side in ('left','right')})
        conditions.append(dict(condition=condition,tasks=decision['aggregate'],gates=decision['seeds'],
            recovery_groups=groups,failed_recovery_gates=[dict(seed=x['seed'],failures=failures(x['metrics']['recovery'])) for x in reports]))
    references=read(Path(p['reference_source'])/'analysis.json')
    result=dict(references=[c for c in references['conditions'] if c['condition'] in p['reference_conditions']],status='complete',fixed_final_update=max(p['updates']),conditions=conditions,
        elapsed_seconds=r['elapsed_seconds'],promote=False,
        scope='Wide expert48-scene coverage plus two already studied recovery scenes; three initialization seeds. Report all predeclared conditions; '
              'no model selection, independent scene evidence or closed-loop success claim.',
        source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',run/'audit.json',
            run/'coverage_audit.json',run/'decision.json',Path(p['reference_source'])/'analysis.json',Path(__file__)]})
    write_json(run/'analysis.json',result)
    for c in conditions:
        print(c['condition'],json.dumps(dict(recovery=c['tasks']['recovery'],
            anchor_held=c['tasks']['anchor_held'],gates=c['gates'],groups=c['recovery_groups'])))


if __name__=='__main__':
    main()
