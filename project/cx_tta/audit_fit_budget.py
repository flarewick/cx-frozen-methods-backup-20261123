"""Independent partition, history, saved-command FK and budget checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_pose_capacity import numpy_fk, angles, episode_metrics
from .paths import write_json
from .robot_kinematics import RobotFK

def read(p):return json.loads(p.read_text())

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();p=read(run/'protocol.json');r=read(run/'result.json')
    if r['status']!='complete':raise ValueError('Completed diagnostic required')
    for path,sha in p['source_sha256'].items():
        if digest(Path(path))!=sha:raise ValueError('Changed source '+path)
    if digest(run/'data.npz')!=p['data_sha256']:raise ValueError('Changed diagnostic data')
    with np.load(run/'data.npz') as z: data={k:z[k].copy() for k in z.files}
    rows=p['rows'];fit=np.array([i for i,v in enumerate(rows) if v['pilot_split']=='fit'])
    for name,sha in p['preprocessing_sha256'].items():
        f=run/f'{name}_preprocess.npz'
        if digest(f)!=sha:raise ValueError('Changed preprocessing')
        with np.load(f) as z:
            if not np.array_equal(z['fit_indices'],fit):raise ValueError('Preprocessing leakage')
            hi=z['history_indices'];std=z['context_std'];mean=z['context_mean']
            for i,j in enumerate(hi):
                if j>=0 and (rows[i]['group']!=rows[j]['group'] or rows[j]['frame']>=rows[i]['frame']
                             or rows[j]['pilot_split']!=rows[i]['pilot_split']):
                    raise ValueError('Noncausal/cross-group history')
            unscaled=data[name]*std+mean
            if not np.allclose(unscaled[hi>=0,78:156],unscaled[hi[hi>=0],:78],atol=2e-5):
                raise ValueError('History features mismatch')
            if not np.allclose(unscaled[hi<0,78:156],0,atol=2e-5):raise ValueError('Invented history')
    if any(rows[i]['pilot_split']!='fit' for i in np.r_[data['rare'],data['low']]):raise ValueError('Held fit overlap')
    expected={(c,s,u) for c,(_,_,updates) in p['conditions'].items() for s in p['seeds'] for u in updates}
    actual={(x['condition'],x['seed'],x['update']) for x in r['reports']}
    if expected!=actual or len(actual)!=len(r['reports']):raise ValueError('Budget/seed coverage mismatch')
    bots=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')];off=p['offsets'];results=[];files=[];maxparity=0.
    for report in r['reports']:
        if report['parameters']!=40718 or report['reload']!='exact':raise ValueError('Capacity/reload mismatch')
        if digest(Path(report['checkpoint']))!=report['checkpoint_sha256']:raise ValueError('Changed weights')
        files.append(Path(report['checkpoint']))
        for f in report['files']:
            path=Path(f['path']);files.append(path)
            if digest(path)!=f['sha256']:raise ValueError('Changed prediction')
            part=path.stem.split(f"_u{report['update']}_",1)[1]
            with np.load(path) as z: saved={k:z[k].copy() for k in z.files}
            ix=saved['indices'];expected_ix=data[part] if part in ('rare','low') else np.array([i for i,x in enumerate(rows) if x['pilot_split']==part])
            if not np.array_equal(ix,expected_ix) or not all(np.array_equal(saved[k],data[k][ix]) for k in ('base','target','valid')):
                raise ValueError('Pairing/partition mismatch')
            if not np.isfinite(saved['predicted']).all():raise ValueError('Nonfinite commands')
            def endpoints(x):return np.stack([numpy_fk(bot,x[:,off,start:start+6].astype(float)) for bot,start in zip(bots,(0,7))],2)
            pp,bp,tp=[endpoints(saved[k]) for k in ('predicted','base','target')]
            a,b=angles(pp,tp),angles(bp,tp);m=np.broadcast_to(saved['valid'][:,off,None],a.shape);large=m&(b>=45)
            if not np.array_equal(m,saved['mask']) or not np.array_equal(large,saved['large']) or not np.allclose(a,saved['angle'],atol=1e-9):raise ValueError('FK/threshold mismatch')
            for bot,start in zip(bots,(0,7)):
                q=saved['predicted'][0,off[0],start:start+6].astype(float)
                reference=bot.pose(q);fast=numpy_fk(bot,q)
                err=float(np.max(np.abs(reference-fast)));maxparity=max(maxparity,err)
                if err>1e-10:raise ValueError('Independent scalar FK parity failed')
            translation=np.linalg.norm(pp[...,:3,3]-tp[...,:3,3],axis=-1)
            summary,groups=episode_metrics(a,b,translation,m,large,ix,rows,saved['predicted'],saved['target'],saved['valid'])
            for task,values in report['metrics'][part].items():
                expected_values=summary[task]
                for k in ('angle_deg','base_angle_deg','large_angle_deg','large_base_angle_deg'):
                    if values[k] is None:
                        if expected_values[k] is not None:raise ValueError('Empty aggregation mismatch')
                    elif abs(values[k]-expected_values[k])>1e-8:raise ValueError('Aggregation mismatch '+k)
                if abs(values['native_mse']-expected_values['native_action_mse'])>1e-8:raise ValueError('MSE mismatch')
                for threshold,key in ((3,'under_3deg'),(15,'under_15deg')):
                    bygroup=[]
                    for group in sorted({rows[i]['group'] for i in ix if rows[i]['task']==task}):
                        selected=np.array([j for j,i in enumerate(ix) if rows[i]['group']==group]);lm=large[selected]
                        if lm.any():bygroup.append(float((a[selected][lm]<threshold).mean()))
                    value=None if not bygroup else float(np.mean(bygroup))
                    if value!=values[key]:raise ValueError('Threshold aggregation mismatch')
            results.append(dict(condition=report['condition'],seed=report['seed'],update=report['update'],part=part,
                metrics=report['metrics'][part]))
    final=[]
    for report in r['reports']:
        if report['update']!=max(p['conditions'][report['condition']][2]):continue
        supported={t:v for t,v in report['metrics']['rare'].items() if v['large_angle_deg'] is not None}
        rare_pass=all(v['large_angle_deg']<=p['gates']['rare_mean_max_deg'] and v['under_3deg']>=p['gates']['rare_fraction_under_3deg_min'] for v in supported.values())
        low_pass=all(v['angle_deg']<=p['gates']['low_mean_max_deg'] for v in report['metrics']['low'].values())
        final.append(dict(condition=report['condition'],seed=report['seed'],rare_fit_pass=rare_pass,
            low_fit_pass=low_pass,rare_metrics=supported,low_metrics=report['metrics']['low'],
            held_metrics=report['metrics']['inner_held'],anchor_metrics=report['metrics']['anchor_held']))
    write_json(run/'independent_metrics.json',dict(records=results,final=final))
    write_json(run/'audit.json',dict(passed=True,snapshots=len(actual),rare_rows=len(data['rare']),
        low_rows=len(data['low']),rare_groups=len({rows[i]['group'] for i in data['rare']}),
        scalar_fk_parity=maxparity,source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',run/'data.npz',Path(__file__),*files]},
        checks=['Fit-only small sets and full prescribed budget/seed coverage','Fit-only preprocessing and causal history',
                'Same 40718 parameters and exact checkpoint reload','Actual source base/teacher/valid pairing',
                'Saved-command float64 FK, scalar FK parity and independent group aggregation'],scope=p['scope']))
    print(json.dumps(dict(passed=True,snapshots=len(actual),final=[{k:x[k] for k in ('condition','seed','rare_fit_pass','low_fit_pass')} for x in final]),indent=2))

if __name__=='__main__':main()
