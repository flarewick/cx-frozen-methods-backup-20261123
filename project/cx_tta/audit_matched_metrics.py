"""Recompute paired recovery metrics from saved native commands, including seams."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .paths import write_json
from .audit import digest
from .probe_candidates import read,verify
from .matched_recovery_data import load
from .robot_kinematics import RobotFK
from .recovery_fit_metrics import metrics,recovery_gate
from .candidate_diagnostic import fixed_actions
from .component_selector import assemble


def main(run):
    p,a,_,_=load(run);analysis=read(run/'analysis.json');result=read(run/'result.json')
    verify(read(run/'evaluation_protocol.json'))
    ix=np.array([i for i,r in enumerate(p['rows']) if r['kind']=='recovery']);rows=[p['rows'][i] for i in ix]
    base,target,valid=(a[k][ix] for k in ('base','target','valid'))
    robots=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')]
    max_error=0.;count=0
    for source,report in zip(result['reports'],analysis['reports']):
        if any(source[k]!=report[k] for k in ('condition','encoding','seed')):raise ValueError('Model alignment failed')
        with np.load(source['artifact']) as z:pred=z['predicted'][ix].copy()
        fixed=fixed_actions(base,pred)
        variants=[('raw',fixed[3],report['parts']['raw']['train_recovery']),
                  ('capped',fixed[6],report['parts']['capped']['train_recovery'])]
        for key,item in report['recovery_oracles'].items():
            support=key.split('_')[0];action=assemble(fixed,support,np.asarray(item['choices']))
            variants.append((key,action,item['summary']))
        for name,action,saved in variants:
            independent,_=metrics(action,base.astype(float),target,valid,rows,robots)
            assert recovery_gate(independent)==saved['recovery_fit_gate']
            for expected,actual in zip(independent['groups'],saved['groups']):
                assert expected['group']==actual['group'] and expected['valid_actions']==actual['valid_actions']
                for side in ('left','right'):
                    for k in ('angle_deg','position_m','arm_rmse_rad','gripper_mae','pose_under_3deg_1cm_fraction',
                              'velocity_error_rmse_rad','joint_limit_violations','gripper_outside_margin'):
                        x,y=expected['arms'][side][k],actual['arms'][side][k]
                        if x is None or y is None:assert x==y;continue
                        error=abs(x-y);max_error=max(error,max_error)
                        if error>1e-7:raise ValueError(f'Metric mismatch {name}/{k}: {error}')
                x,y=expected['seam_error_rmse_rad'],actual['seam_error_rmse_rad']
                if x is not None:
                    error=abs(x-y);max_error=max(max_error,error);assert error<=1e-7
            count+=1
    aggregates=[]
    for encoding in p['encodings']:
        for support in ('raw','capped'):
            for condition in ('omit_shoe','omit_switch'):
                pairs=[r for r in analysis['paired'] if (r['encoding'],r['support'],r['condition'],r['excluded_this_scene'])==(encoding,support,condition,True)]
                assert len(pairs)==3
                def mean_for(variant,key):return float(np.mean([np.mean([a[key][variant] for a in r['arms'].values()]) for r in pairs]))
                other=[r for r in analysis['paired'] if (r['encoding'],r['support'],r['condition'],r['excluded_this_scene'])==(encoding,support,condition,False)]
                aggregates.append(dict(encoding=encoding,support=support,omitted_group=p['omitted_groups'][condition],
                    both_pass=sum(r['both_gate'] for r in pairs),omission_pass=sum(r['omission_gate'] for r in pairs),
                    repeats=3,independent_omitted_scenes=1,other_included_scene_pass=sum(r['omission_gate'] for r in other),
                    averages={variant:{k:mean_for(variant,k) for k in ('angle_deg','position_m','arm_rmse_rad','gripper_mae','score')}
                              for variant in ('both','omission')}))
    verify(p)
    write_json(run/'summary.json',dict(status='complete',paired_aggregates=aggregates,
        independent_recovery_scenes=2,model_initializations_per_encoding=3,
        scope='Means over both arms and three initialization repeats; gates tested independently per scene and arm. '
              'Three repeats are not three independent recovery scenes. Same scene with/without its own supervision.'))
    write_json(run/'metrics_audit.json',dict(passed=True,native_command_variant_checks=count,
        maximum_metric_difference=max_error,gates_match=True,
        source_sha256={str(f):digest(f) for f in (run/'analysis.json',run/'evaluation_protocol.json',Path(__file__))}))
    print('METRICS_AUDIT_PASSED',count,max_error,flush=True)
    print('PAIRED',aggregates,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    threadpool_limits(2);main(parser.parse_args().run.resolve())
