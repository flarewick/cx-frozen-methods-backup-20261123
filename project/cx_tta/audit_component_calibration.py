"""Independent grid enumeration, saved-choice verification and fixed-dose controls."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .paths import write_json
from .audit import digest
from .probe_candidates import read


def weights(rows, ids):
    result=np.zeros(len(rows));use=[i for i in ids if rows[i]['kind']=='expert']
    tasks={rows[i]['task'] for i in use}
    for task in tasks:
        groups={rows[i]['group'] for i in use if rows[i]['task']==task}
        for group in groups:
            ix=[i for i in use if rows[i]['group']==group]
            result[ix]=1/(len(tasks)*len(groups)*len(ix))
    return result


def enumerate_options(pred, panel, score, w, anchor_ids):
    thresholds=[0.,.01,.05,.1,.2] if score=='gain' else [.5,.6,.7,.8,.9]
    head=0 if score=='gain' else 1
    options=[]
    for t,r in [(None,None)]+[(t,r) for t in thresholds for r in [.05,.1,.2]]:
        if t is None:choice=np.zeros(pred.shape[:-2],int)
        else:
            score_array=np.where((pred[...,head]>t)&(pred[...,2]<=r),pred[...,head],-np.inf)
            choice=score_array.argmax(-1)+1
            choice[np.isneginf(score_array).all(-1)]=0
        selected={k:np.take_along_axis(v,choice[...,None],-1)[...,0] for k,v in panel.items()}
        # Average candidates and sides first, then balanced scene rows; keep component separate.
        gain=np.einsum('nc,n->c',selected['gain'].mean((0,2)),w)
        harm=np.einsum('nc,n->c',selected['harm'].mean((0,2)),w)
        coverage=np.einsum('nc,n->c',(choice>0).mean((0,2)),w)
        anchor=selected['harm'][:,anchor_ids].any((0,1,2))
        options.append(dict(t=t,r=r,choice=choice,gain=gain,harm=harm,coverage=coverage,anchor=anchor))
    return options


def optimum(options,mode):
    def key(j,c=None):
        o=options[j];gain=o['gain'].mean() if c is None else o['gain'][c]
        coverage=o['coverage'].mean() if c is None else o['coverage'][c]
        return (gain,-coverage,0 if o['t'] is None else o['t'],0 if o['r'] is None else -o['r'])
    if mode=='split':
        return [max((j for j,o in enumerate(options) if o['harm'][c]<=.05 and not o['anchor'][c]),key=lambda j:key(j,c)) for c in range(2)]
    good=[j for j,o in enumerate(options) if not o['anchor'].any() and
          (o['harm'].mean()<=.05 if mode=='pooled_reference' else (o['harm']<=.05).all())]
    j=max(good,key=key);return [j,j]


def main(run):
    protocol=read(run/'protocol.json');rows=protocol['rows'];ids=protocol['source_indices']
    source=Path(protocol['source']);sr=read(source/'probe_result.json')['reports']
    reports=read(run/'result.json')['reports'];label_files=read(source/'probe_data_manifest.json')['records']
    labels={}
    for e in label_files:
        with np.load(e['file']) as z:labels[e['encoding'],e['support']]={k:z[k][:,ids].copy() for k in z.files}
    enumerations=0;choice_checks=0;baselines=[]
    for report,record in zip(reports,sr):
        tag=f"{record['encoding']}_{record['support']}_{record['architecture']}_s{record['seed']}"
        if any(report[k]!=record[k] for k in ('encoding','support','architecture','seed')):raise ValueError('Report order changed')
        with np.load(record['artifact']) as z:pred=z['prediction'][:,ids].copy()
        panel=labels[record['encoding'],record['support']]
        with np.load(run/(tag+'.npz')) as z:saved={k:z[k].copy() for k in z.files}
        for rule,values in report['rules'].items():
            score,mode=rule.split('_',1)
            for fold in values['folds']+[dict(held_scene=None,thresholds=values['full_calibration']['thresholds'])]:
                group=fold['held_scene'];train=[i for i,r in enumerate(rows) if r['group']!=group]
                anchors=[i for i in train if rows[i]['kind']=='anchor']
                opts=enumerate_options(pred,panel,score,weights(rows,train),anchors)
                selected=optimum(opts,mode)
                if selected!=[t['option'] for t in fold['thresholds']]:raise ValueError('Independent optimum mismatch')
                enumerations+=1
                if group is not None:
                    held=[i for i,r in enumerate(rows) if r['group']==group]
                    for c,j in enumerate(selected):
                        if not np.array_equal(saved[rule][:,held,:,c],opts[j]['choice'][:,held,:,c]):raise ValueError('Saved held choices mismatch')
                        choice_checks+=1
        w=weights(rows,list(range(len(rows))));fixed={}
        for c,name in enumerate(('arm','grip')):
            truth=panel['gain'][:,:,:,c,1:]
            mass=np.broadcast_to(w[None,:,None,None],truth.shape)
            diff=np.stack([truth[...,1]-truth[...,0],truth[...,2]-truth[...,0],truth[...,2]-truth[...,1]],-1)
            use=np.abs(diff)>1e-8;denom=mass[use].sum()
            fixed[name]=dict(always_larger_alpha_pair_accuracy=float(np.sum(mass[use]*(diff[use]>0))/denom),
                always_smaller_alpha_pair_accuracy=float(np.sum(mass[use]*(diff[use]<0))/denom),
                teacher_known_best_gain=report['within_state_ranking'][name]['teacher_known_best_including_off_gain'],
                fixed_gain=report['within_state_ranking'][name]['fixed_alpha_gain'])
        baselines.append(dict(tag=tag,components=fixed))
    aggregates=[]
    for encoding in ('repeat','fourier'):
        for support in ('raw','capped'):
            for architecture in ('linear','mlp32'):
                rr=[r for r in reports if (r['encoding'],r['support'],r['architecture'])==(encoding,support,architecture)]
                out=dict(encoding=encoding,support=support,architecture=architecture,rules={},ranking={})
                for name in rr[0]['rules']:
                    cc=[r['rules'][name]['cross_calibration'] for r in rr]
                    out['rules'][name]=dict(
                        mean_metrics={k:float(np.mean([x['pooled'][k] for x in cc])) for k in ('gain','harm','coverage')},
                        gain_per_probe_seed=[x['pooled']['gain'] for x in cc],
                        positive_scene_counts=[sum(f['metrics']['pooled']['gain']>0 for f in r['rules'][name]['folds']) for r in rr],
                        per_task_gain={t:[x['per_task'][t]['gain'] for x in cc] for t in cc[0]['per_task']},
                        full_calibration_gain=[r['rules'][name]['full_calibration']['metrics']['pooled']['gain'] for r in rr])
                for c in ('arm','grip'):
                    out['ranking'][c]={h:float(np.mean([r['within_state_ranking'][c][h]['pair_accuracy'] for r in rr])) for h in ('gain','benefit')}
                aggregates.append(out)
    write_json(run/'fixed_dose_controls.json',dict(reports=baselines,
        scope='Fixed ordering is diagnostic; ranking a binary strict-improvement probability does not optimize gain magnitude.'))
    write_json(run/'summary.json',dict(aggregates=aggregates,independent_expert_scenes=8,held_success_scenes=0,
        scope='Development leave-one-calibration-scene-out only. Seeds and frames do not increase independent scene count.'))
    if any(digest(Path(f))!=sha for f,sha in protocol['source_sha256'].items()):raise ValueError('Changed inputs')
    write_json(run/'independent_audit.json',dict(passed=True,independent_grid_enumerations=enumerations,
        saved_component_choice_checks=choice_checks,source_hashes_unchanged=True,
        auditor_sha256=digest(Path(__file__)),validation_rows_used=0))
    print(dict(passed=True,enumerations=enumerations,choice_checks=choice_checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    main(parser.parse_args().run.resolve())
