"""S1w: freeze all S1v final candidates and separate quality from selection."""
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .audit import digest
from .paths import ROOT,new_run,write_json,replace_json
from .robot_kinematics import RobotFK
from .recovery_fit_metrics import endpoints
from .candidate_diagnostic import (CAPS,ALPHAS,FIXED,SUPPORTS,ORACLES,VARIANTS,TOL,
    fixed_actions,row_metrics,oracle_indices,mix_arms,summarize)

SOURCE=ROOT/'outputs/effect_learning/coverage_20261002_092902_991797'
def read(p):return json.loads(p.read_text())


def verify(p):
    for name,sha in p['source_sha256'].items():
        if digest(Path(name))!=sha:raise ValueError('Changed frozen input '+name)


def load_candidate(report,p,arrays,*,return_overlap=False):
    pred=np.zeros_like(arrays['base']);seen=np.zeros(len(pred),bool);overlap=0.
    # The same17 ordinary rows were saved using a different inference batch size.
    # Use the complete expert file consistently; retain/check the subset evidence.
    for entry in sorted(report['files'],key=lambda e:e['part']=='ordinary'):
        with np.load(entry['path']) as z:
            ids=z['indices']
            if ids.tolist()!=p['parts'][entry['part']]:raise ValueError('Partition changed')
            if any(not np.array_equal(z[k],arrays[k][ids]) for k in ('base','target','valid')):
                raise ValueError('Changed paired data')
            if seen[ids].any():
                overlap=max(overlap,float(np.abs(pred[ids][seen[ids]]-z['predicted'][seen[ids]]).max()))
                if overlap>2e-5:raise ValueError('Overlapping predictions exceed prior checkpoint replay tolerance')
            fresh=~seen[ids];pred[ids[fresh]]=z['predicted'][fresh];seen[ids]=True
    if not seen.all() or not np.isfinite(pred).all():raise ValueError('Incomplete final predictions')
    return (pred,overlap) if return_overlap else pred


def prepare(source):
    prior=read(source/'protocol.json');result=read(source/'result.json')
    for name in ('audit.json','coverage_audit.json'):
        if not read(source/name)['passed']:raise ValueError('Completed audited S1v source required')
    if result['status']!='complete':raise ValueError('Incomplete source')
    final=[r for r in result['reports'] if r['update']==1400]
    if len(final)!=12 or {(r['condition'],r['seed']) for r in final}!={(c,s) for c in prior['conditions'] for s in prior['seeds']}:
        raise ValueError('All four conditions and three seeds required')
    files=[source/name for name in ('protocol.json','result.json','audit.json','coverage_audit.json','data.npz')]
    files += [ROOT/'cx_tta'/name for name in ('candidate_diagnostic.py','probe_candidates.py','audit_candidates.py',
        'summarize_candidates.py','recovery_fit_metrics.py','audit_pose_capacity.py','audit_local_geometry.py',
        'robot_kinematics.py','audit_time_encoding_dense.py','paths.py','audit.py')]
    files += [Path(prior['config'])]
    files += [RobotFK(read(Path(prior['config'])),side).urdf for side in ('left','right')]
    for report in final:
        if digest(Path(report['checkpoint']))!=report['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        files.append(Path(report['checkpoint']))
        for entry in report['files']:
            if digest(Path(entry['path']))!=entry['sha256']:raise ValueError('Prediction changed')
            files.append(Path(entry['path']))
    if digest(source/'data.npz')!=prior['data_sha256']:raise ValueError('Data changed')
    with np.load(source/'data.npz') as z:
        arrays={k:z[k].copy() for k in z.files}
        valid=z['valid'];n,h,_=z['base'].shape
        if h!=32 or np.any(np.diff(valid.astype(int),axis=1)>0) or np.any(valid.sum(1)==0):
            raise ValueError('Unexpected action horizon/valid tail')
    overlap=[]
    for report in final:
        _,error=load_candidate(report,prior,arrays,return_overlap=True)
        overlap.append(dict(condition=report['condition'],seed=report['seed'],max_native_error=error))
    run=new_run('candidates',group=('reports','analyses'))
    write_json(run/'protocol.json',dict(status='frozen_before_diagnostic',source=str(source),
        data=str(source/'data.npz'),config=prior['config'],rows=prior['rows'],parts=prior['parts'],
        models=final,variants=list(VARIANTS),alphas=list(ALPHAS),caps=CAPS.tolist(),
        prediction_identity='Complete expert-fit predictions are canonical for overlapping ordinary17 rows. '
            'The17-row separately saved predictions differ slightly due to batch size; preserve and compare '
            'with prior2e-5 command replay tolerance. No source prediction/model is modified.',
        overlap_checks=overlap,
        rows_count=n,source_sha256={str(f):digest(f) for f in sorted(set(files))},
        candidate_rule='Use all12 final1400 models. No checkpoint, seed, encoding, frequency or support selection. '
            'Raw: base+alpha*(saved_prediction-base). Capped: clip each residual to original arm0.1/grip0.2 '
            'BEFORE alpha scaling. This support projection is not a rerun of the original tanh head, '
            'and component clipping may change direction. No angle wrapping or action re-query.',
        score='Mean over valid steps per arm block of original dense BC+SE3: native7-channel normalized MSE '
            '(arm0.1/grip0.2), rotation chordal squared/(2*0.1^2), position squared/0.02^2. '
            'No time penalty in oracle selection; report actual velocity/seams separately. This surrogate is NOT success.',
        oracles='Label-privileged finite-grid diagnostics only. Per chunk: binary0/1 or grid0/.25/.5/1, '
            'same alpha both arms. Per-arm grid permits a distinct alpha per arm, constant within whole32-step block. '
            'Guarded-arm restricts mean angle/position/grip and range counts not worse than base before argmin. '
            'Exact ties choose base/lowest alpha. All use teacher future commands and are NOT deployable routers.',
        benefit='Descriptive material benefit: >max(1e-8,5% of base dense score) improvement, mean angle/position/grip '
            'nonworsening plus no increased URDF/grip range count. Temporal error is reported, NOT constrained by this label.',
        guard_tolerance=TOL,stability='Report all seeds, same-block/same-arm common-alpha benefit across all3 seeds; '
            'also distinguish any-alpha-per-seed from one common alpha. Frame/arm/seed counts are not independent episodes.',
        aggregation='Metric task means average scene then arm; within scene weight valid steps. Benefit fractions '
            'average row/arm then scene then task. Ordinary17 overlaps expert856 and never adds to unique counts.',
        scope='S1w offline diagnostic on fixed cached states and predictions. Expert-held/anchor-held are reused '
            'developer scenes; recovery scenes are training data. No new data, model inference/training, router training, '
            'GPU, simulation, TTA or closed-loop success claim. Finite-grid oracle is an attainable minimum of its '
            'surrogate over these choices, not an upper bound over arbitrary residual policies or physical success.'))
    print(run,flush=True)


def execute(run):
    p=read(run/'protocol.json');verify(p);started=time.monotonic();reports=[]
    if (run/'result.json').exists():raise ValueError('Fresh diagnostic run required')
    with np.load(p['data']) as z:arrays={k:z[k].copy() for k in z.files}
    target=arrays['target'].astype(float);valid=arrays['valid'];base=arrays['base'].astype(float)
    robots=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')];tp=endpoints(target,robots)
    base_metric=row_metrics(base,target,valid,robots,tp);parity=0.
    try:
        sample=np.array(sorted({ids[0] for ids in p['parts'].values()}|
            {int(np.flatnonzero(~valid.all(1))[0])}))
        sample_pred=load_candidate(p['models'][0],p,arrays)[sample]
        sample_actions=fixed_actions(base[sample],sample_pred)
        if not np.array_equal(sample_actions[0],base[sample]) or not np.array_equal(sample_actions[3],sample_pred):
            raise ValueError('Zero/full identity mismatch')
        original=row_metrics(sample_actions[3],target[sample],valid[sample],robots)
        changed_target=target[sample].copy();changed_action=sample_actions[3].copy()
        changed_target[~valid[sample]]=1000;changed_action[~valid[sample]]=-1000
        masked=row_metrics(changed_action,changed_target,valid[sample],robots)
        if not all(np.array_equal(original[k],masked[k]) for k in original):raise ValueError('Invalid tail leaked into diagnostic')
        write_json(run/'interface_check.json',dict(passed=True,actual_rows=sample.tolist(),
            invalid_steps=int((~valid[sample]).sum()),zero_full_identity_exact=True,
            invalid_tail_metrics_exact=True,max_source_overlap_error=max(x['max_native_error'] for x in p['overlap_checks'])))
        for index,report in enumerate(p['models']):
            original=load_candidate(report,p,arrays);actions=fixed_actions(base,original)
            mets=[base_metric]+[row_metrics(a,target,valid,robots,tp) for a in actions[1:]]
            choices=[np.full((len(base),2),i,np.int16) for i in range(len(FIXED))]
            for support in SUPPORTS:
                for mode in ORACLES:
                    selection=oracle_indices(mets[:len(FIXED)],support,mode)
                    selected=mix_arms(actions[:len(FIXED)],selection)
                    choices.append(selection);actions.append(selected);mets.append(row_metrics(selected,target,valid,robots,tp))
            panels={k:np.stack([m[k] for m in mets]) for k in base_metric}
            tag=f"{report['condition']}_s{report['seed']}";file=run/f'{tag}.npz'
            np.savez_compressed(file,**panels,choices=np.stack(choices))
            summaries={variant:{part:summarize(m,a,target,valid,p['rows'],ids,base_metric)
                for part,ids in p['parts'].items()} for variant,m,a in zip(VARIANTS,mets,actions)}
            # New row/scene aggregation must reproduce the already audited full candidate.
            for part in p['parts']:
                prior=report['metrics'][part];current=summaries['raw_100'][part]
                for before,after in zip(prior['groups'],current['groups']):
                    if before['group']!=after['group']:raise ValueError('Group mismatch')
                    for side in ('left','right'):
                        for k in ('angle_deg','position_m','gripper_mae','arm_rmse_rad','velocity_error_rmse_rad'):
                            if before['arms'][side][k] is None:continue
                            err=abs(before['arms'][side][k]-after['arms'][side][k]);parity=max(parity,err)
                            tolerance=(2e-4 if k=='angle_deg' else 2e-5) if part=='ordinary' else 2e-6
                            if err>tolerance:raise ValueError('Original full-candidate metric mismatch '+k)
            reports.append(dict(condition=report['condition'],seed=report['seed'],panel=str(file),
                panel_sha256=digest(file),summaries=summaries))
            replace_json(run/'progress.json',dict(models=index+1,total_models=12,elapsed_seconds=time.monotonic()-started))
            print('COMPLETE_MODEL',tag,flush=True)
        verify(p)
        write_json(run/'result.json',dict(status='complete',reports=reports,elapsed_seconds=time.monotonic()-started,
            original_metric_max_error=parity,scope=p['scope']))
        print('COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed_models=len(reports)));raise


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--run',type=Path)
    args=parser.parse_args();threadpool_limits(2)
    if args.prepare_only:prepare(args.source.resolve())
    elif args.run is None:parser.error('--run required')
    else:execute(args.run.resolve())

if __name__=='__main__':main()
