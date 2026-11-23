"""Independent reconstruction of all frozen counterfactuals and oracle choices."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .audit import digest
from .paths import write_json
from .probe_candidates import read,verify,load_candidate
from .robot_kinematics import RobotFK
from .recovery_fit_metrics import metrics as reference_metrics
from .audit_time_encoding_dense import strict_fit,preservation
from .audit_pose_capacity import numpy_fk


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();threadpool_limits(2)
    p=read(run/'protocol.json');r=read(run/'result.json');verify(p)
    if r['status']!='complete' or len(r['reports'])!=12:raise ValueError('Complete12-model diagnostic required')
    if {(x['condition'],x['seed']) for x in r['reports']}!={(x['condition'],x['seed']) for x in p['models']}:
        raise ValueError('Model subset changed')
    with np.load(p['data']) as z:arrays={k:z[k].copy() for k in z.files}
    base=arrays['base'].astype(float);target=arrays['target'].astype(float);valid=arrays['valid'];rows=p['rows']
    robots=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')]
    caps=np.array([.1]*6+[.2]+[.1]*6+[.2]);max_metric=0.;max_scalar=0.;checked=0;files=[]
    samples=sorted({next(i for i,row in enumerate(rows) if row['group']==g) for g in {row['group'] for row in rows}}|set(p['parts']['recovery']))
    for report,model in zip(r['reports'],p['models']):
        if (report['condition'],report['seed'])!=(model['condition'],model['seed']):raise ValueError('Wrong model order')
        file=Path(report['panel']);files.append(file)
        if digest(file)!=report['panel_sha256']:raise ValueError('Changed panel')
        with np.load(file) as z:panel={k:z[k].copy() for k in z.files}
        pred=load_candidate(model,p,arrays).astype(float);delta=pred-base
        # Independent formula, including projection order and bit-exact raw endpoints.
        bounded=np.minimum(np.maximum(delta,-caps),caps)
        fixed=[base,base+.25*delta,base+.5*delta,pred,base+.25*bounded,base+.5*bounded,base+bounded]
        for k in range(7):
            if not np.all(panel['choices'][k]==k):raise ValueError('Fixed-action choices changed')
        for vi,name in enumerate(p['variants']):
            selected=panel['choices'][vi]
            if vi>=7:
                support,mode=name.split('_',1);pool=[0,1,2,3] if support=='raw' else [0,4,5,6]
                if mode=='binary_chunk':pool=[pool[0],pool[-1]]
                for i in range(len(rows)):
                    if mode.endswith('chunk'):
                        best=min(pool,key=lambda c:float(panel['score'][c,i].mean()))
                        if not np.all(selected[i]==best):raise ValueError('Wrong chunk oracle')
                    else:
                        for arm in (0,1):
                            allowed=pool
                            if mode=='guarded_arm':
                                allowed=[c for c in pool if all(panel[k][c,i,arm]<=panel[k][0,i,arm]+tol
                                    for k,tol in p['guard_tolerance'].items()) and all(panel[k][c,i,arm]<=panel[k][0,i,arm]
                                    for k in ('joint_limit_violations','gripper_outside_margin'))]
                            best=min(allowed,key=lambda c:float(panel['score'][c,i,arm]))
                            if selected[i,arm]!=best:raise ValueError('Wrong per-arm oracle')
                action=base.copy()
                for c in pool:
                    for arm,start in enumerate((0,7)):
                        active=selected[:,arm]==c;action[active,:,start:start+7]=fixed[c][active,:,start:start+7]
            else:action=fixed[vi]
            if name.startswith('capped') and np.max(np.abs(action-base)-caps)>1e-12:
                raise ValueError('Original residual support exceeded')
            grip=action[...,[6,13]];bg=base[...,[6,13]]
            if ((grip<np.minimum(bg,0)-1e-12)|(grip>np.maximum(bg,1)+1e-12)).any():
                raise ValueError('Gripper envelope changed')
            ref,geom=reference_metrics(action,base,target,valid,rows,robots)
            ordinary=p['parts']['ordinary'];oi=np.asarray(ordinary)
            ordinary_ref,_=reference_metrics(action[oi],base[oi],target[oi],valid[oi],[rows[i] for i in oi],robots)
            lookup={g['group']:g for g in ref['groups']}
            mean=lambda value:(value*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
            # Separate dense-score expression: SO(3) chordal from geodesic angle.
            diff=(action-target).reshape(len(rows),32,2,7)
            expected_score=mean(np.mean((diff/caps.reshape(2,7))**2,axis=-1)+
                4*np.sin(np.radians(geom['angle'])/2)**2/.1**2+geom['position']**2/.02**2)
            for key,expected in [('score',expected_score),('angle_deg',mean(geom['angle'])),
                ('position_m',mean(geom['position'])),('gripper_mae',mean(np.abs(diff[...,6])))]:
                err=float(np.abs(panel[key][vi]-expected).max());max_metric=max(max_metric,err)
                if err>1e-8:raise ValueError('Independent row metric mismatch '+key)
            for part,summary in report['summaries'][name].items():
                expected_groups=ordinary_ref['groups'] if part=='ordinary' else [lookup[g['group']] for g in summary['groups']]
                for actual,expected in zip(summary['groups'],expected_groups):
                    for arm in ('left','right'):
                        for key,value in actual['arms'][arm].items():
                            if key in ('score','base_score'):continue
                            other=expected['arms'][arm][key]
                            if other is None:
                                if value is not None:raise ValueError('Empty metric mismatch')
                            elif abs(value-other)>1e-8:raise ValueError('Independent group metric mismatch '+key)
                    if actual['seam_error_rmse_rad']!=expected['seam_error_rmse_rad']:
                        raise ValueError('Actual routed seam not recomputed')
                gate_input=dict(groups=expected_groups)
                if strict_fit(gate_input)!=summary['recovery_fit_gate'] or preservation(gate_input,anchor=True)!=summary['preservation_gate']:
                    raise ValueError('Gate mismatch')
                for task,values in summary['tasks'].items():
                    for key,value in values.items():
                        vals=[arm[key] for g in summary['groups'] if g['task']==task for arm in g['arms'].values() if arm[key] is not None]
                        expected=None if not vals else float(np.mean(vals))
                        if value!=expected:raise ValueError('Task aggregation mismatch')
            # Sample both boundary poses from every scene against scalar URDF FK.
            for side,start in enumerate((0,7)):
                q=np.stack([action[i,t,start:start+6] for i in samples for t in (0,int(valid[i].sum())-1)])
                vector=numpy_fk(robots[side],q)
                for j,command in enumerate(q):
                    err=float(np.abs(robots[side].pose(command)-vector[j]).max());max_scalar=max(max_scalar,err)
                    if err>1e-10:raise ValueError('Scalar FK disagreement')
            checked+=1
        for support in ('raw','capped'):
            at=lambda name:panel['score'][p['variants'].index(support+'_'+name)]
            if np.max(at('grid_chunk').mean(1)-at('binary_chunk').mean(1))>1e-9:raise ValueError('Grid oracle worse than binary')
            if np.max(at('grid_arm').mean(1)-at('grid_chunk').mean(1))>1e-9:raise ValueError('Arm oracle worse than chunk')
            if np.max(at('guarded_arm')-panel['score'][0])>1e-9:raise ValueError('Guarded oracle worse than base score')
        print('AUDITED',report['condition'],report['seed'],flush=True)
    verify(p)
    write_json(run/'audit.json',dict(passed=True,models=12,variant_model_checks=checked,
        independent_metric_max_error=max_metric,scalar_fk_max_error=max_scalar,
        source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',Path(__file__),*files]},
        checks=['Frozen source and complete final models','Independent command and finite-grid oracle reconstruction',
            'Every valid-step group metric via original independent metric implementation',
            'Row scores recomputed from geodesic/chordal identity','Scalar FK per-scene boundary poses',
            'Actual selected-action velocity/seams and all original fit/preservation gates','Original caps and base-compatible gripper range'],
        scope='Offline label-privileged diagnostics only, not learned routing or physical task success.'))
    print('PASSED',checked,max_metric,max_scalar,flush=True)

if __name__=='__main__':main()
